"""
Adaptive Cache Manager

This module provides an advanced caching system with adaptive TTL and analytics.
It is designed to optimize memory usage and cache hit rates for performance-critical
operations across the forex trading platform.

Features:
- Adaptive TTL based on access patterns
- Cache analytics and statistics
- Memory-efficient storage
- Thread-safe operations
- Automatic cleanup of expired entries
- Predictive precomputation
- Priority-based eviction
- Redis integration for distributed caching
"""

import threading
import time
import sys
import logging
import json
import hashlib
import concurrent.futures
from typing import Any, Dict, Tuple, Optional, List, Hashable, Union, Callable, Set, TypeVar, cast
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import inspect
import functools

try:
    import redis
    from redis.exceptions import RedisError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from common_lib.errors.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    BaseError,
    ServiceError
)

from common_lib.monitoring.performance_monitoring import (
    track_operation
)

# Create logger
logger = logging.getLogger(__name__)

# Type variable for function
F = TypeVar('F', bound=Callable[..., Any])


class CacheEntry:
    """
    Cache entry with metadata.
    
    Attributes:
        value: Cached value
        timestamp: Creation timestamp
        access_count: Number of times the entry has been accessed
        last_access: Last access timestamp
        ttl: Time-to-live in seconds
        priority: Priority for eviction (lower values are evicted first)
        metadata: Additional metadata
    """
    
    def __init__(
        self,
        value: Any,
        ttl: Optional[int] = None,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a cache entry.
        
        Args:
            value: Cached value
            ttl: Time-to-live in seconds
            priority: Priority for eviction (lower values are evicted first)
            metadata: Additional metadata
        """
        self.value = value
        self.timestamp = time.time()
        self.access_count = 0
        self.last_access = self.timestamp
        self.ttl = ttl
        self.priority = priority
        self.metadata = metadata or {}
    
    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """
        Check if the entry is expired.
        
        Args:
            current_time: Current time (defaults to time.time())
            
        Returns:
            True if the entry is expired, False otherwise
        """
        if self.ttl is None:
            return False
        
        current = current_time or time.time()
        return current - self.timestamp > self.ttl
    
    def access(self) -> None:
        """Update access metadata."""
        self.access_count += 1
        self.last_access = time.time()


class AdaptiveCacheManager:
    """
    Advanced caching system with adaptive TTL and analytics.
    
    Features:
    - Adaptive TTL based on access patterns
    - Cache analytics and statistics
    - Memory-efficient storage
    - Thread-safe operations
    - Automatic cleanup of expired entries
    - Redis integration for distributed caching
    """

    def __init__(
        self,
        default_ttl_seconds: int = 300,
        max_size: int = 1000,
        cleanup_interval_seconds: int = 60,
        adaptive_ttl: bool = True,
        redis_url: Optional[str] = None,
        service_name: str = "forex-service",
        enable_metrics: bool = True
    ):
        """
        Initialize the cache manager.
        
        Args:
            default_ttl_seconds: Default time-to-live for cache entries in seconds
            max_size: Maximum number of entries in the cache
            cleanup_interval_seconds: Interval for automatic cache cleanup
            adaptive_ttl: Whether to use adaptive TTL based on access patterns
            redis_url: Redis URL for distributed caching
            service_name: Service name for metrics
            enable_metrics: Whether to collect metrics
        """
        self.default_ttl = default_ttl_seconds
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval_seconds
        self.adaptive_ttl = adaptive_ttl
        self.service_name = service_name
        self.enable_metrics = enable_metrics
        
        # Initialize local cache
        self.cache: Dict[Any, CacheEntry] = {}
        self.lock = threading.RLock()
        
        # Initialize metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0
        self.last_cleanup_time = time.time()
        
        # Initialize Redis client
        self.redis_url = redis_url
        self.redis_client = None
        
        if redis_url and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info(f"Successfully connected to Redis at {redis_url}")
            except (RedisError, AttributeError) as e:
                logger.warning(f"Failed to connect to Redis at {redis_url}: {e}")
                logger.warning("Falling back to local memory cache only")
        
        logger.debug(
            f"AdaptiveCacheManager initialized with TTL={default_ttl_seconds}s, "
            f"max_size={max_size}, cleanup_interval={cleanup_interval_seconds}s, "
            f"adaptive_ttl={adaptive_ttl}, redis_url={redis_url}"
        )
    
    @track_operation("caching", "get")
    def get(self, key: Any) -> Tuple[bool, Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Tuple of (hit, value) where hit is True if the key was found and not expired
        """
        # Try local cache first
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                current_time = time.time()
                
                # Check if the entry is expired
                if entry.is_expired(current_time):
                    self.cache.pop(key)
                    self.expirations += 1
                    self.misses += 1
                    return False, None
                
                # Update access metadata
                entry.access()
                
                # Update TTL if adaptive
                if self.adaptive_ttl:
                    self._update_adaptive_ttl(entry)
                
                self.hits += 1
                
                # Return a copy for mutable objects
                value = entry.value
                if isinstance(value, dict):
                    return True, value.copy()
                elif isinstance(value, list):
                    return True, value.copy()
                else:
                    return True, value
            
            # Try Redis if available
            if self.redis_client:
                try:
                    redis_key = f"cache:{self.service_name}:{key}"
                    redis_value = self.redis_client.get(redis_key)
                    
                    if redis_value:
                        try:
                            value = json.loads(redis_value)
                            
                            # Store in local cache
                            self.set(key, value)
                            
                            self.hits += 1
                            return True, value
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to decode Redis value for key {key}")
                except RedisError as e:
                    logger.warning(f"Redis error while getting key {key}: {e}")
            
            self.misses += 1
            return False, None
    
    @track_operation("caching", "set")
    def set(
        self,
        key: Any,
        value: Any,
        ttl: Optional[int] = None,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional custom TTL in seconds
            priority: Priority for eviction (lower values are evicted first)
            metadata: Additional metadata
        """
        with self.lock:
            # Check if cleanup is needed
            current_time = time.time()
            if current_time - self.last_cleanup_time > self.cleanup_interval:
                self._clean_cache()
            
            # Check if eviction is needed
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_one()
            
            # Create a copy for mutable objects
            if isinstance(value, dict):
                cached_value = value.copy()
            elif isinstance(value, list):
                cached_value = value.copy()
            else:
                cached_value = value
            
            # Create cache entry
            entry = CacheEntry(
                value=cached_value,
                ttl=ttl or self.default_ttl,
                priority=priority,
                metadata=metadata
            )
            
            # Store in local cache
            self.cache[key] = entry
            
            # Store in Redis if available
            if self.redis_client:
                try:
                    redis_key = f"cache:{self.service_name}:{key}"
                    redis_value = json.dumps(value)
                    
                    if ttl:
                        self.redis_client.setex(redis_key, ttl, redis_value)
                    else:
                        self.redis_client.setex(redis_key, self.default_ttl, redis_value)
                except (RedisError, TypeError) as e:
                    logger.warning(f"Redis error while setting key {key}: {e}")
    
    @track_operation("caching", "delete")
    def delete(self, key: Any) -> bool:
        """
        Delete a key from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if the key was found and deleted, False otherwise
        """
        with self.lock:
            found = key in self.cache
            if found:
                self.cache.pop(key)
            
            # Delete from Redis if available
            if self.redis_client:
                try:
                    redis_key = f"cache:{self.service_name}:{key}"
                    self.redis_client.delete(redis_key)
                except RedisError as e:
                    logger.warning(f"Redis error while deleting key {key}: {e}")
            
            return found
    
    @track_operation("caching", "clear")
    def clear(self) -> None:
        """Clear the entire cache."""
        with self.lock:
            self.cache.clear()
            
            # Clear Redis if available
            if self.redis_client:
                try:
                    pattern = f"cache:{self.service_name}:*"
                    keys = self.redis_client.keys(pattern)
                    if keys:
                        self.redis_client.delete(*keys)
                except RedisError as e:
                    logger.warning(f"Redis error while clearing cache: {e}")
    
    @track_operation("caching", "get_or_set")
    def get_or_set(
        self,
        key: Any,
        value_func: Callable[[], Any],
        ttl: Optional[int] = None,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Get a value from the cache or compute and store it if not available.
        
        Args:
            key: Cache key
            value_func: Function to compute the value
            ttl: Optional custom TTL in seconds
            priority: Priority for eviction (lower values are evicted first)
            metadata: Additional metadata
            
        Returns:
            Cached or computed value
        """
        hit, cached_value = self.get(key)
        if hit:
            return cached_value
        
        value = value_func()
        self.set(key, value, ttl, priority, metadata)
        return value
    
    def _clean_cache(self) -> None:
        """Clean expired entries from the cache."""
        with self.lock:
            current_time = time.time()
            expired_keys = []
            
            for key, entry in list(self.cache.items()):
                if entry.is_expired(current_time):
                    expired_keys.append(key)
            
            for key in expired_keys:
                self.cache.pop(key)
                self.expirations += 1
            
            self.last_cleanup_time = current_time
            
            if expired_keys:
                logger.debug(f"Cleaned {len(expired_keys)} expired entries from cache")
    
    def _evict_one(self) -> None:
        """
        Evict one item from the cache using adaptive policy.
        
        Uses a scoring system that considers:
        - Access frequency
        - Recency of access
        - Age relative to TTL
        - Priority
        """
        if not self.cache:
            return
        
        current_time = time.time()
        scores = {}
        
        for k, entry in self.cache.items():
            # Calculate score components
            age_factor = (current_time - entry.timestamp) / max(1, entry.ttl or self.default_ttl)
            recency_factor = (current_time - entry.last_access) / max(1, self.default_ttl)
            popularity_factor = 1.0 / max(1, entry.access_count)
            priority_factor = 1.0 / max(1, entry.priority + 1)  # +1 to avoid division by zero
            
            # Combine factors into a score
            scores[k] = (
                age_factor * 0.3 +
                recency_factor * 0.3 +
                popularity_factor * 0.2 +
                priority_factor * 0.2
            )
        
        if scores:
            # Evict the entry with the highest score
            worst_key = max(scores.items(), key=lambda x: x[1])[0]
            self.cache.pop(worst_key)
            self.evictions += 1
            logger.debug(f"Evicted cache entry with key: {worst_key}")
    
    def _update_adaptive_ttl(self, entry: CacheEntry) -> None:
        """
        Update TTL based on access patterns.
        
        Args:
            entry: Cache entry
        """
        if not self.adaptive_ttl:
            return
        
        # Adjust TTL based on access count and recency
        access_factor = min(5, entry.access_count) / 5  # Cap at 5 accesses
        recency_factor = min(1.0, (time.time() - entry.timestamp) / self.default_ttl)
        
        # Increase TTL for frequently accessed entries
        new_ttl = int(self.default_ttl * (1 + access_factor * (1 - recency_factor)))
        entry.ttl = new_ttl
    
    @track_operation("caching", "get_stats")
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            stats = {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "expirations": self.expirations,
                "hit_rate": hit_rate,
                "has_redis": self.redis_client is not None
            }
            
            return stats


# Create a decorator for caching function results
def cached(
    ttl: Optional[int] = None,
    key_prefix: Optional[str] = None,
    cache_manager: Optional[AdaptiveCacheManager] = None,
    key_func: Optional[Callable[..., Any]] = None
) -> Callable[[F], F]:
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time-to-live in seconds
        key_prefix: Prefix for cache keys
        cache_manager: Cache manager to use
        key_func: Function to generate cache keys
        
    Returns:
        Decorated function
    
    Usage:
        @cached(ttl=60)
        def expensive_calculation(param1, param2):
            # ...calculations...
            return result
    """
    def decorator(func: F) -> F:
    """
    Decorator.
    
    Args:
        func: Description of func
    
    Returns:
        F: Description of return value
    
    """

        # Get or create cache manager
        nonlocal cache_manager
        if cache_manager is None:
            cache_manager = AdaptiveCacheManager()
        
        # Get function name and module for key generation
        func_name = func.__name__
        module_name = func.__module__
        prefix = key_prefix or f"{module_name}.{func_name}"
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Convert args and kwargs to a string
                args_str = str(args)
                kwargs_str = str(sorted(kwargs.items()))
                
                # Generate a hash
                key_str = f"{prefix}:{args_str}:{kwargs_str}"
                cache_key = hashlib.md5(key_str.encode()).hexdigest()
            
            # Get or compute value
            return cache_manager.get_or_set(
                key=cache_key,
                value_func=lambda: func(*args, **kwargs),
                ttl=ttl
            )
        
        return cast(F, wrapper)
    
    return decorator


# Create singleton instance
_default_cache_manager = AdaptiveCacheManager()


def get_cache_manager() -> AdaptiveCacheManager:
    """
    Get the default cache manager.
    
    Returns:
        Default cache manager
    """
    return _default_cache_manager
