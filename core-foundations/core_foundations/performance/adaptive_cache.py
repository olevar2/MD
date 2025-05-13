\
"""
Section: 5.1. Adaptive Caching (Performance Optimization)
Purpose: Implement an intelligent caching mechanism that adapts its behavior
         based on usage patterns.
"""

import time
import collections
import threading
import sys
import heapq
from typing import Any, Optional, Dict, Tuple, Callable, List, Counter as CounterType

# Uncomment for Redis distributed cache coordination
import redis

class AdaptiveCache:
    """
    An intelligent caching mechanism that adapts its behavior based on usage patterns.

    Features:
    - Time-To-Live (TTL) management (basic implementation, adaptive logic TBD).
    - Memory-aware cache sizing and eviction policies (LRU implemented).
    - Analytics/metrics for cache hit/miss rates.
    - Basic cache invalidation.
    - Placeholder for distributed cache coordination.
    """

    def __init__(self, max_size: int = 1024, default_ttl: Optional[float] = 300, eviction_policy: str = 'lru',
                 adaptive_ttl: bool = False, redis_host: Optional[str] = None, redis_port: int = 6379):
        """
        Initialize the AdaptiveCache.

        Args:
            max_size: Maximum number of items in the cache.
            default_ttl: Default time-to-live for cache items in seconds. None means items don't expire by default.
            eviction_policy: Cache eviction policy ('lru', 'lfu').
            adaptive_ttl: Whether to use adaptive TTL management based on access patterns.
            redis_host: Redis host for distributed cache coordination. None disables distributed features.
            redis_port: Redis port for distributed cache coordination.
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if default_ttl is not None and default_ttl <= 0:
            raise ValueError("default_ttl must be positive or None")

        self.max_size = max_size
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy.lower()
        self.adaptive_ttl = adaptive_ttl
        self._cache: Dict[Any, Tuple[Any, float]] = {} # {key: (value, expiry_timestamp)}
        self._access_order = collections.OrderedDict() # For LRU
        self._access_frequency = collections.Counter() # For LFU and adaptive TTL
        self._frequency_recency = [] # For LFU with recency tiebreaker [(freq, timestamp, key)]
        self._access_timestamps = {} # For adaptive TTL {key: last_access_timestamp}
        self._lock = threading.RLock() # Thread safety

        # Metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        # Distributed cache client (Redis)
        self.distributed_client = None
        if redis_host:
            try:
                self.distributed_client = redis.Redis(host=redis_host, port=redis_port, db=0)
                # Test connection
                self.distributed_client.ping()
                # Subscribe to invalidation channel in a separate thread
                self._setup_invalidation_listener()
            except Exception as e:
                print(f"Warning: Failed to connect to Redis at {redis_host}:{redis_port} - {e}")
                self.distributed_client = None

        if self.eviction_policy not in ['lru', 'lfu']:
             raise ValueError(f"Unsupported eviction policy: {eviction_policy}. Supported: 'lru', 'lfu'")


    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve an item from the cache.

        Args:
            key: The key of the item to retrieve.

        Returns:
            The cached value if found and not expired, otherwise None.
        """
        with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                current_time = time.time()
                
                if expiry is None or current_time < expiry:
                    # Item found and valid
                    self._hits += 1
                    
                    # Update access patterns for LRU/LFU
                    if self.eviction_policy == 'lru':
                        self._access_order.move_to_end(key)
                    
                    # Update frequency for LFU and adaptive TTL
                    self._access_frequency[key] += 1
                    self._access_timestamps[key] = current_time
                    
                    # If using adaptive TTL, potentially update the expiry time
                    if self.adaptive_ttl and expiry is not None:
                        new_ttl = self._adapt_ttl(key)
                        if new_ttl:
                            # Update expiry based on current adaptive TTL logic
                            new_expiry = current_time + new_ttl
                            # Only extend TTL if the new expiry is later than the current one
                            if new_expiry > expiry:
                                self._cache[key] = (value, new_expiry)
                    
                    return value
                else:
                    # Item expired, remove it
                    self._evict(key)

            # Item not found or expired
            self._misses += 1
            return None

    def set(self, key: Any, value: Any, ttl: Optional[float] = 'default') -> None:
        """
        Add or update an item in the cache.

        Args:
            key: The key of the item to store.
            value: The value to store.
            ttl: Time-to-live in seconds for this specific item.
                 If 'default', uses the cache's default_ttl.
                 If None, the item never expires based on time.
                 If a positive number, sets the specific TTL.
        """
        with self._lock:
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._apply_eviction()

            # Determine TTL value
            if self.adaptive_ttl and key in self._access_frequency:
                # For existing keys, use adaptive TTL
                effective_ttl = self._adapt_ttl(key)
            else:
                # For new keys or if adaptive TTL is disabled, use specified ttl
                effective_ttl = self.default_ttl if ttl == 'default' else ttl
                
            expiry = time.time() + effective_ttl if effective_ttl is not None else None

            # Update the cache
            self._cache[key] = (value, expiry)
            
            # Update metadata for eviction policies
            if self.eviction_policy == 'lru':
                if key in self._access_order:
                    self._access_order.move_to_end(key)
                else:
                    self._access_order[key] = None
                    
            # Initialize or reset frequency for LFU
            current_time = time.time()
            self._access_timestamps[key] = current_time
            if key not in self._access_frequency:
                self._access_frequency[key] = 1
            # We don't reset frequency on set to maintain the "frequency" aspect


    def delete(self, key: Any) -> bool:
        """
        Remove an item from the cache explicitly.

        Args:
            key: The key of the item to remove.

        Returns:
            True if the item was found and removed, False otherwise.
        """
        with self._lock:
            success = self._evict(key) > 0
            if success and self.distributed_client:
                # Notify other caches about this invalidation
                self._coordinate_invalidation(key)
            return success


    def invalidate(self, key_pattern: Optional[str] = None, condition: Optional[Callable[[Any, Any], bool]] = None) -> int:
        """
        Intelligently invalidate cache entries.

        Args:
            key_pattern: A pattern to match keys for invalidation (e.g., prefix*, *suffix). Basic implementation.
            condition: A function `(key, value) -> bool` to determine if an entry should be invalidated.

        Returns:
            The number of items invalidated.
        """
        # Basic pattern matching, can be expanded (e.g., regex)
        # More sophisticated invalidation might involve dependency tracking or event-based triggers.
        count = 0
        keys_to_invalidate = []
        with self._lock:
            for key, (value, _) in self._cache.items():
                invalidate_flag = False
                if key_pattern:
                    # Simple prefix/suffix matching for demonstration
                    if (key_pattern.endswith('*') and str(key).startswith(key_pattern[:-1])) or \
                       (key_pattern.startswith('*') and str(key).endswith(key_pattern[1:])) or \
                       (str(key) == key_pattern):
                           invalidate_flag = True
                if condition and condition(key, value):
                    invalidate_flag = True

                if invalidate_flag:
                    keys_to_invalidate.append(key)

            for key in keys_to_invalidate:
                count += self._evict(key) # Use internal eviction
                if self.distributed_client:
                    # Notify other caches about each invalidation
                    self._coordinate_invalidation(key)

        return count


    def clear(self) -> None:
        """Remove all items from the cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._access_frequency.clear()
            self._access_timestamps.clear()
            self._frequency_recency = []
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            
            # Optionally notify distributed cache to clear as well
            if self.distributed_client:
                try:
                    self.distributed_client.publish('cache_clear', 'all')
                except Exception as e:
                    print(f"Error publishing cache clear: {e}")


    def _apply_eviction(self) -> None:
        """Apply the configured eviction policy to remove one item."""
        if self.eviction_policy == 'lru':
            try:
                # popitem(last=False) removes the least recently used item
                lru_key, _ = self._access_order.popitem(last=False)
                self._evict(lru_key)
            except KeyError:
                pass # Cache might be empty or key already removed
        elif self.eviction_policy == 'lfu':
            if not self._access_frequency:
                return  # Nothing to evict
                
            # Find the key with the lowest access frequency
            min_freq = min(self._access_frequency.values())
            min_freq_keys = [k for k, v in self._access_frequency.items() if v == min_freq]
            
            if len(min_freq_keys) == 1:
                # Only one item with lowest frequency
                self._evict(min_freq_keys[0])
            else:
                # Multiple items with same lowest frequency, use LRU as tiebreaker
                oldest_key = None
                oldest_time = float('inf')
                for key in min_freq_keys:
                    if key in self._access_timestamps and self._access_timestamps[key] < oldest_time:
                        oldest_time = self._access_timestamps[key]
                        oldest_key = key
                        
                if oldest_key:
                    self._evict(oldest_key)
        # Add other policies as needed


    def _evict(self, key: Any) -> int:
        """Internal helper to remove a specific key and update structures."""
        # Assumes lock is already held
        removed_count = 0
        if key in self._cache:
            del self._cache[key]
            removed_count = 1
            self._evictions += 1
            
            # Clean up metadata structures
            if self.eviction_policy == 'lru':
                if key in self._access_order:
                    del self._access_order[key]
                    
            # Clean up LFU and adaptive TTL metadata
            if key in self._access_frequency:
                del self._access_frequency[key]
            if key in self._access_timestamps:
                del self._access_timestamps[key]
                
        return removed_count


    def get_metrics(self) -> Dict[str, Any]:
        """
        Return enhanced cache performance metrics.
        
        Returns:
            A dictionary containing performance metrics about the cache.
        """
        with self._lock:
            metrics = {
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "current_size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": (self._hits / (self._hits + self._misses)) if (self._hits + self._misses) > 0 else 0,
                "eviction_policy": self.eviction_policy,
                "adaptive_ttl": self.adaptive_ttl,
                "distributed_enabled": bool(self.distributed_client)
            }
            
            # Add memory usage estimation
            try:
                # Rough estimation of memory usage (in bytes)
                key_size = sum(sys.getsizeof(key) for key in self._cache)
                value_size = sum(sys.getsizeof(value) for value, _ in self._cache.values())
                metadata_size = sys.getsizeof(self._access_order) + sys.getsizeof(self._access_frequency) + sys.getsizeof(self._access_timestamps)
                metrics["estimated_memory_usage"] = key_size + value_size + metadata_size
            except:
                # In case of any error estimating size
                metrics["estimated_memory_usage"] = "unknown"
                
            return metrics
            
    def configure_distributed(self, redis_host: str, redis_port: int = 6379) -> bool:
        """
        Configure the distributed cache coordination after initialization.
        
        Args:
            redis_host: Redis host address for distributed cache coordination.
            redis_port: Redis port for distributed cache coordination.
            
        Returns:
            True if successful, False otherwise.
        """
        with self._lock:
            try:
                self.distributed_client = redis.Redis(host=redis_host, port=redis_port, db=0)
                # Test connection
                self.distributed_client.ping()
                # Set up listener
                self._setup_invalidation_listener()
                return True
            except Exception as e:
                print(f"Failed to configure distributed cache: {e}")
                self.distributed_client = None
                return False

    # --- Placeholder for Adaptive Behavior ---
    def _adapt_ttl(self, key: Any) -> Optional[float]:
        """
        Determine adaptive TTL based on access patterns.
        
        The adaptive logic increases TTL for frequently accessed items and
        decreases TTL for infrequently accessed ones. This optimizes cache efficiency
        by keeping frequently used items longer and evicting rarely used items faster.
        
        Args:
            key: The key to calculate an adaptive TTL for.
            
        Returns:
            The calculated TTL in seconds, or None if the item should never expire.
        """
        if not self.adaptive_ttl or self.default_ttl is None:
            return self.default_ttl
            
        # Get frequency and access patterns
        frequency = self._access_frequency.get(key, 0)
        
        # Calculate time since first access (approx, as we only store last access)
        current_time = time.time()
        time_since_access = 0
        if key in self._access_timestamps:
            time_since_access = current_time - self._access_timestamps[key]
        
        # Base TTL is the default
        ttl = self.default_ttl
        
        # Adjust based on frequency - higher frequency = longer TTL
        if frequency > 20:
            # Very frequently accessed items get much longer TTL
            ttl = self.default_ttl * 4
        elif frequency > 10:
            # Moderately frequently accessed items get longer TTL
            ttl = self.default_ttl * 2
        elif frequency > 5:
            # Somewhat frequently accessed items get slightly longer TTL
            ttl = self.default_ttl * 1.5
        elif frequency < 2:
            # Very infrequently accessed items get shorter TTL
            ttl = self.default_ttl * 0.5
            
        # Further adjust based on recency - more recently accessed = longer TTL
        # This prevents items that were once popular but now unused from staying too long
        if time_since_access > self.default_ttl * 2:
            # If it hasn't been accessed in a while, reduce TTL
            ttl = ttl * 0.75
        
        # Ensure we return a sensible value (minimum 10% of default_ttl)
        return max(self.default_ttl * 0.1, ttl)

    # --- Placeholder for Distributed Coordination ---
    def _coordinate_invalidation(self, key: Any):
        """
        Notify other cache instances about invalidation using Redis Pub/Sub.
        
        This enables distributed cache coordination across multiple instances
        of the application that share the same Redis instance.
        
        Args:
            key: The key that was invalidated locally and should be invalidated in other instances.
        """
        if not self.distributed_client:
            return
            
        try:
            # Publish invalidation message
            self.distributed_client.publish('cache_invalidation', str(key))
        except Exception as e:
            print(f"Error publishing invalidation for key {key}: {e}")
            
    def _setup_invalidation_listener(self):
        """Set up a listener for distributed cache invalidation events."""
        if not self.distributed_client:
            return
            
        def invalidation_listener():
    """
    Invalidation listener.
    
    """

            pubsub = self.distributed_client.pubsub()
            pubsub.subscribe('cache_invalidation')
            
            for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        key = message['data'].decode('utf-8')
                        # Skip if we originated this invalidation
                        if not message.get('_from_self', False):
                            with self._lock:
                                self._evict(key)
                    except Exception as e:
                        print(f"Error processing invalidation message: {e}")
        
        # Start listener thread
        thread = threading.Thread(target=invalidation_listener, daemon=True)
        thread.start()

# Example Usage (Optional - for testing/demonstration)
if __name__ == "__main__":
    cache = AdaptiveCache(max_size=3, default_ttl=5)

    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    print(f"Cache size: {len(cache)}")
    print(f"Metrics: {cache.get_metrics()}")

    print(f"Get 'a': {cache.get('a')}") # Hit, makes 'a' most recently used
    time.sleep(1)
    cache.set("d", 4) # Evicts 'b' (least recently used)
    print(f"Cache size after adding 'd': {len(cache)}")
    print(f"Contains 'b': {'b' in cache}") # Miss
    print(f"Get 'b': {cache.get('b')}") # Miss
    print(f"Metrics: {cache.get_metrics()}")

    print(f"Get 'c': {cache.get('c')}") # Hit
    print(f"Metrics: {cache.get_metrics()}")

    print("Waiting for TTL expiry...")
    time.sleep(5)
    print(f"Get 'a' after TTL: {cache.get('a')}") # Miss (expired)
    print(f"Get 'd' after TTL: {cache.get('d')}") # Miss (expired)
    print(f"Contains 'c': {'c' in cache}") # Miss (expired)
    print(f"Cache size after TTL: {len(cache)}")
    print(f"Metrics: {cache.get_metrics()}")

    cache.set("e", 5, ttl=None) # No expiry
    cache.set("f", 6)
    cache.invalidate(key_pattern='f')
    print(f"Contains 'f' after invalidation: {'f' in cache}")
    print(f"Metrics: {cache.get_metrics()}")
    cache.clear()
    print(f"Cache size after clear: {len(cache)}")
    print(f"Metrics: {cache.get_metrics()}")

