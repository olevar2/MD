"""
Feature Cache Module

This module provides caching functionality for feature store data to reduce
load on the feature store service and improve performance.
"""
import time
import threading
import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from datetime import datetime, timedelta
from core_foundations.utils.logger import get_logger
try:
    import redis
    has_redis = True
except ImportError:
    has_redis = False


from strategy_execution_engine.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class CacheEntry:
    """
    A cache entry with expiration.
    
    Attributes:
        value: The cached value
        expiry: Expiration timestamp
    """

    def __init__(self, value: Any, ttl: int=300):
        """
        Initialize a cache entry.
        
        Args:
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        self.value = value
        self.expiry = time.time() + ttl

    def is_expired(self) ->bool:
        """Check if the entry has expired."""
        return time.time() > self.expiry


class FeatureCache:
    """
    Cache for feature store data.
    
    This class provides an in-memory cache with optional Redis-based
    distributed caching for feature store data.
    
    Attributes:
        logger: Logger instance
        local_cache: Dictionary of local cache entries
        redis_client: Redis client for distributed caching (if available)
        cleanup_interval: Interval for cache cleanup in seconds
        max_size: Maximum number of entries in the local cache
        stats: Cache statistics
    """

    @with_exception_handling
    def __init__(self, redis_url: Optional[str]=None, cleanup_interval: int
        =300, max_size: int=1000, default_ttl: int=300):
        """
        Initialize the feature cache.
        
        Args:
            redis_url: URL for Redis connection (if None, only local cache is used)
            cleanup_interval: Interval for cache cleanup in seconds
            max_size: Maximum number of entries in the local cache
            default_ttl: Default time-to-live for cache entries in seconds
        """
        self.logger = get_logger('feature_cache')
        self.local_cache: Dict[str, CacheEntry] = {}
        self.cleanup_interval = cleanup_interval
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.stats = {'hits': 0, 'misses': 0, 'size': 0, 'evictions': 0}
        self._lock = threading.RLock()
        self.redis_client = None
        if redis_url and has_redis:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.logger.info('Redis cache initialized')
            except Exception as e:
                self.logger.warning(
                    f'Failed to initialize Redis cache: {str(e)}')
        self._start_cleanup_thread()
        self.logger.info(
            f'Feature cache initialized with max size: {max_size}, TTL: {default_ttl}s'
            )

    @with_exception_handling
    def get(self, key: str) ->Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value, or None if not found or expired
        """
        if self.redis_client:
            try:
                redis_value = self.redis_client.get(f'feature_cache:{key}')
                if redis_value:
                    import pickle
                    value = pickle.loads(redis_value)
                    with self._lock:
                        self.stats['hits'] += 1
                    self.logger.debug(f'Redis cache hit for {key}')
                    return value
            except Exception as e:
                self.logger.warning(f'Redis cache error: {str(e)}')
        with self._lock:
            entry = self.local_cache.get(key)
            if entry and not entry.is_expired():
                self.stats['hits'] += 1
                self.logger.debug(f'Local cache hit for {key}')
                return entry.value
            if entry:
                del self.local_cache[key]
                self.stats['size'] -= 1
            self.stats['misses'] += 1
            return None

    @with_exception_handling
    def set(self, key: str, value: Any, ttl: Optional[int]=None) ->None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (if None, uses default TTL)
        """
        ttl = ttl or self.default_ttl
        if self.redis_client:
            try:
                import pickle
                serialized_value = pickle.dumps(value)
                self.redis_client.setex(f'feature_cache:{key}', ttl,
                    serialized_value)
            except Exception as e:
                self.logger.warning(f'Redis cache error: {str(e)}')
        with self._lock:
            if len(self.local_cache
                ) >= self.max_size and key not in self.local_cache:
                self._evict_entries()
            if key in self.local_cache:
                self.local_cache[key] = CacheEntry(value, ttl)
            else:
                self.local_cache[key] = CacheEntry(value, ttl)
                self.stats['size'] += 1

    @with_exception_handling
    def invalidate(self, key: str) ->None:
        """
        Invalidate a cache entry.
        
        Args:
            key: Cache key to invalidate
        """
        if self.redis_client:
            try:
                self.redis_client.delete(f'feature_cache:{key}')
            except Exception as e:
                self.logger.warning(f'Redis cache error: {str(e)}')
        with self._lock:
            if key in self.local_cache:
                del self.local_cache[key]
                self.stats['size'] -= 1

    @with_exception_handling
    def invalidate_pattern(self, pattern: str) ->None:
        """
        Invalidate all cache entries matching a pattern.
        
        Args:
            pattern: Pattern to match (e.g., "ohlcv_*")
        """
        if self.redis_client:
            try:
                keys = self.redis_client.keys(f'feature_cache:{pattern}')
                if keys:
                    self.redis_client.delete(*keys)
            except Exception as e:
                self.logger.warning(f'Redis cache error: {str(e)}')
        with self._lock:
            keys_to_delete = [k for k in self.local_cache.keys() if pattern in
                k]
            for key in keys_to_delete:
                del self.local_cache[key]
                self.stats['size'] -= 1

    @with_exception_handling
    def clear(self) ->None:
        """Clear the entire cache."""
        if self.redis_client:
            try:
                keys = self.redis_client.keys('feature_cache:*')
                if keys:
                    self.redis_client.delete(*keys)
            except Exception as e:
                self.logger.warning(f'Redis cache error: {str(e)}')
        with self._lock:
            self.local_cache.clear()
            self.stats['size'] = 0

    def get_stats(self) ->Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'
                ] / total_requests if total_requests > 0 else 0
            stats = {'hits': self.stats['hits'], 'misses': self.stats[
                'misses'], 'hit_rate': hit_rate, 'size': self.stats['size'],
                'max_size': self.max_size, 'evictions': self.stats[
                'evictions'], 'has_redis': self.redis_client is not None}
            return stats

    def _evict_entries(self) ->None:
        """
        Evict entries from the cache when it's full.
        
        This method uses a simple LRU-like approach by evicting expired entries first,
        then the oldest entries if needed.
        """
        with self._lock:
            current_time = time.time()
            expired_keys = [k for k, v in self.local_cache.items() if 
                current_time > v.expiry]
            for key in expired_keys:
                del self.local_cache[key]
                self.stats['size'] -= 1
            if len(self.local_cache) >= self.max_size:
                sorted_entries = sorted(self.local_cache.items(), key=lambda
                    x: x[1].expiry)
                num_to_remove = max(1, int(self.max_size * 0.1))
                for i in range(num_to_remove):
                    if i < len(sorted_entries):
                        key, _ = sorted_entries[i]
                        del self.local_cache[key]
                        self.stats['evictions'] += 1
                        self.stats['size'] -= 1

    def _cleanup_expired(self) ->None:
        """Remove all expired entries from the cache."""
        with self._lock:
            current_time = time.time()
            expired_keys = [k for k, v in self.local_cache.items() if 
                current_time > v.expiry]
            for key in expired_keys:
                del self.local_cache[key]
                self.stats['size'] -= 1
            if expired_keys:
                self.logger.debug(
                    f'Cleaned up {len(expired_keys)} expired cache entries')

    @with_exception_handling
    def _start_cleanup_thread(self) ->None:
        """Start a background thread for periodic cache cleanup."""

        @with_exception_handling
        def cleanup_task():
    """
    Cleanup task.
    
    """

            while True:
                time.sleep(self.cleanup_interval)
                try:
                    self._cleanup_expired()
                except Exception as e:
                    self.logger.error(f'Error in cache cleanup: {str(e)}')
        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()
        self.logger.debug(
            f'Started cache cleanup thread (interval: {self.cleanup_interval}s)'
            )
