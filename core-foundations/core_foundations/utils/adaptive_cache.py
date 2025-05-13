"""
Adaptive cache module.

This module provides functionality for...
"""

import time
import threading
import pickle # Added for Redis serialization
from collections import OrderedDict
from heapq import heappush, heappop
from typing import Any, Callable, Dict, Optional, Tuple, Union

# Import Redis client library
import redis

# Import the base metrics collector for type hinting
from core_foundations.monitoring.base_collector import BaseMetricsCollector

class AdaptiveCache:
    """
    Implements an intelligent caching mechanism that adapts to usage patterns.

    Features:
    - Adaptive Time-To-Live (TTL) based on access frequency.
    - Adaptive strategy selection (LRU/LFU).
    - Basic event-based invalidation mechanism.
    - Thread-safe operations.
    - Placeholders for monitoring and distributed cache integration.
    """

    def __init__(self, max_size: int = 1024, default_ttl: int = 300, strategy: str = 'adaptive',
                 redis_client: Optional[redis.Redis] = None,
                 metrics_collector: Optional[BaseMetricsCollector] = None # Updated type hint
                 ):
        """
        Initializes the AdaptiveCache.

        Args:
            max_size: Maximum number of items in the cache.
            default_ttl: Default time-to-live for cache items in seconds.
            strategy: Caching strategy ('lru', 'lfu', 'adaptive').
            redis_client: Optional Redis client instance for distributed caching.
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if default_ttl <= 0:
            raise ValueError("default_ttl must be positive")
        if strategy not in ['lru', 'lfu', 'adaptive']:
            raise ValueError("strategy must be 'lru', 'lfu', or 'adaptive'")

        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy
        self.redis_client = redis_client # Uncommented
        self.metrics_collector = metrics_collector # Store metrics collector

        self._cache: Dict[Any, Tuple[Any, float, int]] = {}  # key -> (value, expiry_time, access_count)
        self._lru_order = OrderedDict() # For LRU strategy
        self._lfu_heap = [] # Min-heap (access_count, timestamp, key) for LFU strategy
        self._lock = threading.RLock()
        self._event_listeners: Dict[str, list[Callable]] = {} # event_name -> list of callbacks

        # Monitoring placeholders
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        # Adaptive strategy state (example: switch based on hit rate)
        self._adaptive_threshold = 0.8 # Example threshold for switching
        self._current_strategy = 'lru' if strategy == 'adaptive' else strategy # Start with LRU if adaptive

    def _get_current_time(self) -> float:
        """Returns the current time."""
        return time.time()

    def _is_expired(self, key: Any) -> bool:
        """Checks if a cache item is expired."""
        if key not in self._cache:
            return True
        _, expiry_time, _ = self._cache[key]
        return self._get_current_time() > expiry_time

    def _evict(self):
        """Evicts items based on the current strategy."""
        if self.strategy == 'adaptive':
            self._update_adaptive_strategy()
            eviction_strategy = self._current_strategy
        else:
            eviction_strategy = self.strategy

        if eviction_strategy == 'lru':
            self._evict_lru()
        elif eviction_strategy == 'lfu':
            self._evict_lfu()

    def _evict_lru(self):
        """Evicts the least recently used item."""
        if not self._lru_order:
            return
        key, _ = self._lru_order.popitem(last=False)
        if key in self._cache:
            del self._cache[key]
            self._evictions += 1
            if self.metrics_collector: # Check if collector exists
                self.metrics_collector.increment('cache_evictions', labels={'strategy': 'lru'}) # Monitoring

    def _evict_lfu(self):
        """Evicts the least frequently used item."""
        while self._lfu_heap:
            _, _, key = heappop(self._lfu_heap)
            # Check if the key is still valid (might have been updated or removed)
            if key in self._cache:
                # Ensure the popped item corresponds to the current state in cache
                # This check is needed because heap might contain stale entries
                # A more robust LFU might need a direct link or versioning
                # For simplicity, we assume if key exists, it's the one to evict
                del self._cache[key]
                self._evictions += 1
                if self.metrics_collector: # Check if collector exists
                    self.metrics_collector.increment('cache_evictions', labels={'strategy': 'lfu'}) # Monitoring
                # Clean up LRU order as well if the key exists there
                if key in self._lru_order:
                    del self._lru_order[key]
                return # Evicted one item

    def _update_adaptive_strategy(self):
        """Dynamically adjusts the caching strategy based on performance metrics."""
        total_accesses = self._hits + self._misses
        if total_accesses > 100: # Only adapt after sufficient data
            hit_rate = self._hits / total_accesses if total_accesses > 0 else 0
            # Example logic: Switch to LFU if hit rate is high, else LRU
            if hit_rate > self._adaptive_threshold and self._current_strategy == 'lru':
                print("Adaptive Cache: Switching strategy to LFU")
                self._current_strategy = 'lfu'
                # Potentially rebuild LFU heap from current cache state if needed
            elif hit_rate <= self._adaptive_threshold and self._current_strategy == 'lfu':
                print("Adaptive Cache: Switching strategy to LRU")
                self._current_strategy = 'lru'
                # Potentially rebuild LRU order if needed

    def _calculate_adaptive_ttl(self, access_count: int) -> int:
        """Calculates TTL based on access frequency (simple example)."""
        # Increase TTL for frequently accessed items, up to a max limit
        base_ttl = self.default_ttl
        bonus_ttl = min(access_count * 10, self.default_ttl * 2) # Example: +10s per access, max 2x base
        return base_ttl + bonus_ttl

    def get(self, key: Any) -> Optional[Any]:
        """
        Retrieves an item from the cache.

        Args:
            key: The key of the item to retrieve.

        Returns:
            The cached value if found and not expired, otherwise None.
        """
        # Potential distributed cache check first
        if self.redis_client:
            try:
                redis_key = f"adaptive_cache:{str(key)}" # Add prefix for namespacing
                val = self.redis_client.get(redis_key)
                if val is not None:
                    self._hits += 1
                    if self.metrics_collector: # Check if collector exists
                        self.metrics_collector.increment('cache_hits', labels={'source': 'redis'}) # Monitoring
                    # TODO: Consider updating local cache metadata if needed (e.g., access count, expiry)
                    # For now, just return the value from Redis
                    return pickle.loads(val) # Assuming stored pickled
            except redis.RedisError as e:
                print(f"Redis GET error for key '{key}': {e}") # Add proper logging
            except pickle.PickleError as e:
                 print(f"Redis deserialization error for key '{key}': {e}") # Add proper logging

        with self._lock:
            if key not in self._cache:
                self._misses += 1
                if self.metrics_collector: # Check if collector exists
                    self.metrics_collector.increment('cache_misses') # Monitoring
                return None

            value, expiry_time, access_count = self._cache[key]

            if self._get_current_time() > expiry_time:
                self._misses += 1
                if self.metrics_collector: # Check if collector exists
                    self.metrics_collector.increment('cache_misses') # Monitoring
                # Evict expired item explicitly
                del self._cache[key]
                if key in self._lru_order:
                    del self._lru_order[key]
                # LFU heap removal is handled implicitly during eviction or update
                self._evictions += 1
                if self.metrics_collector: # Check if collector exists
                    self.metrics_collector.increment('cache_evictions', labels={'reason': 'expired'}) # Monitoring
                return None

            # Item found and valid (local cache hit)
            self._hits += 1
            if self.metrics_collector: # Check if collector exists
                self.metrics_collector.increment('cache_hits', labels={'source': 'local'}) # Monitoring

            # Update access metadata
            access_count += 1
            new_ttl = self._calculate_adaptive_ttl(access_count)
            new_expiry_time = self._get_current_time() + new_ttl
            self._cache[key] = (value, new_expiry_time, access_count)

            # Update LRU order
            if self.strategy == 'lru' or (self.strategy == 'adaptive' and self._current_strategy == 'lru'):
                 if key in self._lru_order:
                    self._lru_order.move_to_end(key)

            # Update LFU heap (add new entry, old one becomes stale)
            if self.strategy == 'lfu' or (self.strategy == 'adaptive' and self._current_strategy == 'lfu'):
                 # In a simple heap implementation, we add the updated entry.
                 # Stale entries (with lower access count for the same key)
                 # will eventually be popped and ignored during eviction.
                 heappush(self._lfu_heap, (access_count, self._get_current_time(), key))


            return value

    def set(self, key: Any, value: Any, ttl: Optional[int] = None):
        """
        Adds or updates an item in the cache.

        Args:
            key: The key of the item.
            value: The value to cache.
            ttl: Optional specific time-to-live for this item in seconds.
                 If None, uses adaptive TTL based on future access.
        """
        with self._lock:
            current_time = self._get_current_time()
            # Determine initial TTL and expiry
            initial_access_count = 1
            if ttl is None:
                # Start with default TTL, will adapt on subsequent gets
                expiry_time = current_time + self.default_ttl
            else:
                 if ttl <= 0:
                     raise ValueError("TTL must be positive")
                 expiry_time = current_time + ttl

            # Check if cache is full before adding a new item
            if key not in self._cache and len(self._cache) >= self.max_size:
                self._evict()

            # Store the item
            self._cache[key] = (value, expiry_time, initial_access_count)

            # Update LRU order
            if self.strategy == 'lru' or (self.strategy == 'adaptive' and self._current_strategy == 'lru'):
                if key in self._lru_order:
                    self._lru_order.move_to_end(key)
                else:
                    self._lru_order[key] = None # Value doesn't matter, just order

            # Update LFU heap
            if self.strategy == 'lfu' or (self.strategy == 'adaptive' and self._current_strategy == 'lfu'):
                 # Add the new item to the heap
                 heappush(self._lfu_heap, (initial_access_count, current_time, key))

            # Potential distributed cache set
            if self.redis_client:
                try:
                    # Use calculated expiry for Redis TTL
                    redis_ttl = max(1, int(expiry_time - current_time)) # Ensure TTL is at least 1 second
                    redis_key = f"adaptive_cache:{str(key)}" # Add prefix for namespacing
                    self.redis_client.set(redis_key, pickle.dumps(value), ex=redis_ttl)
                except redis.RedisError as e:
                    print(f"Redis SET error for key '{key}': {e}") # Add proper logging
                except pickle.PickleError as e:
                    print(f"Redis serialization error for key '{key}': {e}") # Add proper logging

    def invalidate(self, key: Any):
        """
        Manually invalidates (removes) an item from the cache.

        Args:
            key: The key of the item to invalidate.
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._lru_order:
                    del self._lru_order[key]
                # LFU heap removal is tricky without direct pointers;
                # it relies on checks during eviction/get.
                # For explicit invalidation, a full heap rebuild or
                # marking items as invalid might be needed for strict LFU.
                # Here, we just remove from the main dict and LRU list.
                print(f"Cache invalidated for key: {key}")
                # Potential distributed cache delete
                if self.redis_client:
                    try:
                        redis_key = f"adaptive_cache:{str(key)}" # Add prefix for namespacing
                        self.redis_client.delete(redis_key)
                    except redis.RedisError as e:
                        print(f"Redis DELETE error for key '{key}': {e}") # Add proper logging

    def subscribe_to_event(self, event_name: str, callback: Callable):
        """Subscribes a callback function to an invalidation event."""
        with self._lock:
            if event_name not in self._event_listeners:
                self._event_listeners[event_name] = []
            self._event_listeners[event_name].append(callback)
            print(f"Callback subscribed to event: {event_name}")

    def publish_event(self, event_name: str, *args, **kwargs):
        """
        Publishes an event, triggering subscribed callbacks (e.g., for invalidation).

        Example: publish_event('user_updated', user_id=123) could trigger
                 cache invalidation for keys related to user 123.
        """
        with self._lock:
            if event_name in self._event_listeners:
                print(f"Publishing event: {event_name}")
                for callback in self._event_listeners[event_name]:
                    try:
                        # Callbacks might perform invalidation based on args
                        callback(self, *args, **kwargs)
                    except Exception as e:
                        print(f"Error executing callback for event {event_name}: {e}") # Add proper logging

    def clear(self):
        """Clears the entire cache."""
        with self._lock:
            self._cache.clear()
            self._lru_order.clear()
            self._lfu_heap = []
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            print("Cache cleared.")
            # Potential distributed cache flush (use with caution)
            # A more targeted approach might be needed depending on Redis usage
            # if self.redis_client:
            #     try:
            #         # Example: Delete keys matching a pattern if namespaced properly
            #         for rkey in self.redis_client.scan_iter("adaptive_cache:*"):
            #             self.redis_client.delete(rkey)
            #         print("Redis keys matching 'adaptive_cache:*' cleared.")
            #     except redis.RedisError as e:
            #         print(f"Redis clear error: {e}") # Add proper logging

    def stats(self) -> Dict[str, Union[int, float, str]]:
        """Returns cache statistics."""
        with self._lock:
            total_accesses = self._hits + self._misses
            hit_rate = self._hits / total_accesses if total_accesses > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": f"{hit_rate:.2%}",
                "current_strategy": self._current_strategy if self.strategy == 'adaptive' else self.strategy
            }

# Example Usage & Event-based Invalidation Callback

# def invalidate_user_cache(cache_instance: AdaptiveCache, user_id: int):
    """
    Invalidate user cache.
    
    Args:
        cache_instance: Description of cache_instance
        user_id: Description of user_id
    
    """

#     """Callback function to invalidate cache entries related to a user."""
#     print(f"Event received: Invalidating cache for user_id: {user_id}")
#     # Example: Invalidate specific keys based on the event payload
#     cache_instance.invalidate(f"user_profile_{user_id}")
#     cache_instance.invalidate(f"user_orders_{user_id}")

# if __name__ == "__main__":
#     cache = AdaptiveCache(max_size=3, strategy='adaptive', default_ttl=5)

#     # Subscribe to an event
#     cache.subscribe_to_event('user_updated', invalidate_user_cache)

#     # Set items
#     cache.set("a", 1)
#     cache.set("b", 2)
#     cache.set("user_profile_123", {"name": "Alice"})
#     print(f"Cache state: {cache.stats()}")

#     # Access items to influence LRU/LFU/Adaptive
#     cache.get("a")
#     time.sleep(1)
#     cache.get("a")
#     cache.get("b")
#     print(f"Cache state after gets: {cache.stats()}")

#     # Add another item, potentially causing eviction
#     cache.set("c", 3)
#     print(f"Cache state after adding 'c': {cache.stats()}")
#     cache.set("d", 4) # This should cause eviction based on strategy
#     print(f"Cache state after adding 'd': {cache.stats()}")

#     # Test expiration
#     print(f"Getting 'a' (should exist): {cache.get('a')}")
#     print("Waiting for TTL expiration...")
#     time.sleep(6)
#     print(f"Getting 'a' after TTL (should be None): {cache.get('a')}")
#     print(f"Cache state after expiration: {cache.stats()}")

#     # Test event-based invalidation
#     cache.set("user_profile_123", {"name": "Alice Updated"})
#     cache.set("user_orders_123", [101, 102])
#     print(f"Cache state before event: {cache.stats()}")
#     print(f"Getting user profile: {cache.get('user_profile_123')}")
#     cache.publish_event('user_updated', user_id=123)
#     print(f"Getting user profile after event (should be None): {cache.get('user_profile_123')}")
#     print(f"Getting user orders after event (should be None): {cache.get('user_orders_123')}")
#     print(f"Cache state after event: {cache.stats()}")
