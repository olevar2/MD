"""
Memory cache implementation (L1) for the feature store caching system.
"""
import time
import re
import sys
import asyncio
import threading
from typing import Dict, Any, Optional, Union, List, Pattern, Tuple
from collections import OrderedDict
from datetime import datetime, timedelta
import pandas as pd

from .base_cache import BaseCache
from .cache_key import CacheKey


class LRUCache(BaseCache):
    """
    LRU Memory Cache implementation (L1 cache).
    
    This is the fastest cache tier that uses an in-memory LRU (Least Recently Used)
    eviction policy to maintain a maximum size. Items are stored in memory for
    fastest access but limited by available RAM.
    
    Features:
    - Thread-safe operations
    - LRU eviction policy
    - TTL (Time to Live) support
    - Size-based constraints
    - Pattern-based invalidation
    """
    
    def __init__(self, max_size: int = 100_000_000, default_ttl_seconds: int = 300):
        """
        Initialize the memory cache.
        
        Args:
            max_size: Maximum cache size in bytes (default: 100MB)
            default_ttl_seconds: Default time-to-live for cache entries in seconds (default: 5 minutes)
        """
        self._cache = OrderedDict()  # {key_str: (expires_at, size_bytes, value)}
        self._max_size = max_size
        self._current_size = 0
        self._default_ttl_seconds = default_ttl_seconds
        self._lock = threading.RLock()
    
    async def get(self, key: Union[str, CacheKey]) -> Optional[Any]:
        """
        Retrieve an item from the memory cache.
        
        Args:
            key: Cache key as string or CacheKey object
            
        Returns:
            The cached data if found and valid, None otherwise
        """
        key_str = key.to_string() if isinstance(key, CacheKey) else key
        
        with self._lock:
            if key_str not in self._cache:
                return None
                
            expires_at, size_bytes, value = self._cache.pop(key_str)
            
            # Check if expired
            if expires_at is not None and datetime.now() > expires_at:
                self._current_size -= size_bytes
                return None
                
            # Move to the end (most recently used)
            self._cache[key_str] = (expires_at, size_bytes, value)
            return value
    
    async def put(self, key: Union[str, CacheKey], value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """
        Store an item in the memory cache.
        
        Args:
            key: Cache key as string or CacheKey object
            value: Data to cache
            ttl_seconds: Time-to-live in seconds, uses default if not specified
            
        Returns:
            True if the operation was successful, False otherwise
        """
        key_str = key.to_string() if isinstance(key, CacheKey) else key
        
        # Calculate expiration time
        if ttl_seconds is not None:
            expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
        elif self._default_ttl_seconds is not None:
            expires_at = datetime.now() + timedelta(seconds=self._default_ttl_seconds)
        else:
            expires_at = None
        
        # Estimate size in bytes
        if isinstance(value, pd.DataFrame):
            size_bytes = value.memory_usage(deep=True).sum()
        elif isinstance(value, (bytes, bytearray)):
            size_bytes = len(value)
        else:
            # Rough estimate for other objects
            size_bytes = sys.getsizeof(value)
        
        with self._lock:
            # If key exists, remove its size from current size
            if key_str in self._cache:
                _, old_size, _ = self._cache[key_str]
                self._current_size -= old_size
            
            # Check if we need to make space
            while self._current_size + size_bytes > self._max_size and self._cache:
                # Remove least recently used item
                removed_key, (_, removed_size, _) = self._cache.popitem(last=False)
                self._current_size -= removed_size
            
            # If we still can't fit the item, return False
            if size_bytes > self._max_size:
                return False
            
            # Add the item
            self._cache[key_str] = (expires_at, size_bytes, value)
            self._current_size += size_bytes
            return True
    
    async def invalidate(self, key_pattern: Union[str, Pattern]) -> int:
        """
        Invalidate cache entries matching a pattern.
        
        Args:
            key_pattern: String pattern or regex pattern to match cache keys
            
        Returns:
            Number of invalidated cache entries
        """
        if isinstance(key_pattern, str):
            pattern = re.compile(key_pattern)
        else:
            pattern = key_pattern
        
        invalidated_count = 0
        with self._lock:
            keys_to_remove = []
            
            # Find keys to remove
            for key_str in self._cache:
                if pattern.search(key_str):
                    keys_to_remove.append(key_str)
            
            # Remove keys and update size
            for key_str in keys_to_remove:
                _, size_bytes, _ = self._cache.pop(key_str)
                self._current_size -= size_bytes
                invalidated_count += 1
        
        return invalidated_count
    
    async def clear(self) -> int:
        """
        Clear all items from the cache.
        
        Returns:
            Number of cleared cache entries
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._current_size = 0
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            expired_count = 0
            now = datetime.now()
            
            for expires_at, _, _ in self._cache.values():
                if expires_at is not None and now > expires_at:
                    expired_count += 1
            
            return {
                "size_bytes": self._current_size,
                "max_size_bytes": self._max_size,
                "utilization": self._current_size / self._max_size if self._max_size > 0 else 0,
                "item_count": len(self._cache),
                "expired_count": expired_count
            }
    
    @property
    def size(self) -> int:
        """
        Get the current size of the cache in bytes.
        
        Returns:
            Size in bytes
        """
        return self._current_size
    
    @property
    def max_size(self) -> int:
        """
        Get the maximum size of the cache in bytes.
        
        Returns:
            Maximum size in bytes
        """
        return self._max_size
    
    @property
    def item_count(self) -> int:
        """
        Get the number of items in the cache.
        
        Returns:
            Number of items
        """
        with self._lock:
            return len(self._cache)
    
    def cleanup_expired(self) -> int:
        """
        Remove expired items from the cache.
        
        Returns:
            Number of removed items
        """
        now = datetime.now()
        removed_count = 0
        
        with self._lock:
            keys_to_remove = []
            
            for key_str, (expires_at, _, _) in self._cache.items():
                if expires_at is not None and now > expires_at:
                    keys_to_remove.append(key_str)
            
            for key_str in keys_to_remove:
                _, size_bytes, _ = self._cache.pop(key_str)
                self._current_size -= size_bytes
                removed_count += 1
        
        return removed_count
