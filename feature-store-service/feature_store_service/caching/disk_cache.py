"""
Disk cache implementation (L2) for the feature store caching system.
"""
import os
import re
import json
import shutil
import pickle
import hashlib
import asyncio
import time
from typing import Dict, Any, Optional, Union, List, Pattern, Tuple
from datetime import datetime, timedelta
import pandas as pd
from .base_cache import BaseCache
from .cache_key import CacheKey


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class DiskCache(BaseCache):
    """
    Disk Cache implementation (L2 cache).
    
    This is the second-tier cache that stores data on disk for larger capacity
    but medium-speed access compared to the memory cache. It's used for recent
    but not immediate data that may be needed again.
    
    Features:
    - File-based storage with organized directory structure
    - Background maintenance processes
    - TTL (Time to Live) support
    - Size-based constraints
    - Pattern-based invalidation
    """

    def __init__(self, directory: str='./cache', max_size: int=10000000000,
        default_ttl_seconds: int=86400):
        """
        Initialize the disk cache.
        
        Args:
            directory: Base directory for cache storage
            max_size: Maximum cache size in bytes (default: 10GB)
            default_ttl_seconds: Default time-to-live for cache entries (default: 24 hours)
        """
        self._base_dir = os.path.abspath(directory)
        self._data_dir = os.path.join(self._base_dir, 'data')
        self._index_path = os.path.join(self._base_dir, 'index.json')
        self._max_size = max_size
        self._current_size = 0
        self._default_ttl_seconds = default_ttl_seconds
        self._index = {}
        os.makedirs(self._data_dir, exist_ok=True)
        self._load_index()
        self._update_current_size()

    @with_exception_handling
    def _load_index(self):
        """Load the cache index from disk."""
        if os.path.exists(self._index_path):
            try:
                with open(self._index_path, 'r') as f:
                    loaded_index = json.load(f)
                self._index = {}
                for key_str, (expires_at_str, file_path, size_bytes
                    ) in loaded_index.items():
                    if expires_at_str:
                        expires_at = datetime.fromisoformat(expires_at_str
                            ).timestamp()
                    else:
                        expires_at = None
                    self._index[key_str] = expires_at, file_path, size_bytes
            except Exception as e:
                print(f'Error loading disk cache index: {e}')
                self._index = {}

    @with_exception_handling
    def _save_index(self):
        """Save the cache index to disk."""
        try:
            serializable_index = {}
            for key_str, (expires_at, file_path, size_bytes
                ) in self._index.items():
                if expires_at:
                    expires_at_str = datetime.fromtimestamp(expires_at
                        ).isoformat()
                else:
                    expires_at_str = None
                serializable_index[key_str
                    ] = expires_at_str, file_path, size_bytes
            with open(self._index_path, 'w') as f:
                json.dump(serializable_index, f)
        except Exception as e:
            print(f'Error saving disk cache index: {e}')

    def _update_current_size(self):
        """Calculate the current cache size from the index."""
        self._current_size = sum(size for _, _, size in self._index.values())

    def _get_hash_path(self, key_str: str) ->str:
        """
        Generate a file path for a cache key using hashing to avoid path length issues.
        
        Args:
            key_str: The cache key string
            
        Returns:
            Relative path for storing the cache file
        """
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        dir_name = key_hash[:2]
        file_name = key_hash[2:] + '.pickle'
        dir_path = os.path.join(self._data_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_name, file_name)

    @async_with_exception_handling
    async def get(self, key: Union[str, CacheKey]) ->Optional[Any]:
        """
        Retrieve an item from the disk cache.
        
        Args:
            key: Cache key as string or CacheKey object
            
        Returns:
            The cached data if found and valid, None otherwise
        """
        key_str = key.to_string() if isinstance(key, CacheKey) else key
        if key_str not in self._index:
            return None
        expires_at, file_path, _ = self._index[key_str]
        if expires_at is not None and time.time() > expires_at:
            await self._remove_item(key_str)
            return None
        full_path = os.path.join(self._data_dir, file_path)
        if not os.path.exists(full_path):
            await self._remove_item(key_str)
            return None
        try:
            loop = asyncio.get_event_loop()
            value = await loop.run_in_executor(None, lambda : pickle.load(
                open(full_path, 'rb')))
            return value
        except Exception as e:
            print(f'Error reading cache file {full_path}: {e}')
            await self._remove_item(key_str)
            return None

    @async_with_exception_handling
    async def put(self, key: Union[str, CacheKey], value: Any, ttl_seconds:
        Optional[int]=None) ->bool:
        """
        Store an item in the disk cache.
        
        Args:
            key: Cache key as string or CacheKey object
            value: Data to cache
            ttl_seconds: Time-to-live in seconds, uses default if not specified
            
        Returns:
            True if the operation was successful, False otherwise
        """
        key_str = key.to_string() if isinstance(key, CacheKey) else key
        if ttl_seconds is not None:
            expires_at = time.time() + ttl_seconds
        elif self._default_ttl_seconds is not None:
            expires_at = time.time() + self._default_ttl_seconds
        else:
            expires_at = None
        rel_path = self._get_hash_path(key_str)
        full_path = os.path.join(self._data_dir, rel_path)
        if key_str in self._index:
            await self._remove_item(key_str)
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda : pickle.dump(value,
                open(full_path, 'wb')))
            size_bytes = os.path.getsize(full_path)
            if self._current_size + size_bytes > self._max_size:
                await self._make_space(size_bytes)
            if size_bytes > self._max_size:
                os.remove(full_path)
                return False
            self._index[key_str] = expires_at, rel_path, size_bytes
            self._current_size += size_bytes
            self._save_index()
            return True
        except Exception as e:
            print(f'Error writing cache file {full_path}: {e}')
            if os.path.exists(full_path):
                os.remove(full_path)
            return False

    async def _make_space(self, required_bytes: int) ->None:
        """
        Make space in the cache by removing expired and least recently used items.
        
        Args:
            required_bytes: Number of bytes to free up
        """
        await self._remove_expired()
        if self._current_size + required_bytes > self._max_size:
            sorted_items = sorted(self._index.items(), key=lambda x: x[1][0
                ] if x[1][0] is not None else float('inf'))
            for key_str, _ in sorted_items:
                if self._current_size + required_bytes <= self._max_size:
                    break
                await self._remove_item(key_str)

    async def _remove_expired(self) ->int:
        """
        Remove all expired items from the cache.
        
        Returns:
            Number of removed items
        """
        now = time.time()
        expired_keys = [key_str for key_str, (expires_at, _, _) in self.
            _index.items() if expires_at is not None and now > expires_at]
        for key_str in expired_keys:
            await self._remove_item(key_str)
        return len(expired_keys)

    @async_with_exception_handling
    async def _remove_item(self, key_str: str) ->bool:
        """
        Remove an item from the cache.
        
        Args:
            key_str: Cache key string
            
        Returns:
            True if the item was removed, False otherwise
        """
        if key_str not in self._index:
            return False
        expires_at, file_path, size_bytes = self._index.pop(key_str)
        self._current_size -= size_bytes
        full_path = os.path.join(self._data_dir, file_path)
        try:
            if os.path.exists(full_path):
                os.remove(full_path)
            return True
        except Exception as e:
            print(f'Error removing cache file {full_path}: {e}')
            return False

    async def invalidate(self, key_pattern: Union[str, Pattern]) ->int:
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
        keys_to_remove = [key_str for key_str in self._index.keys() if
            pattern.search(key_str)]
        invalidated_count = 0
        for key_str in keys_to_remove:
            if await self._remove_item(key_str):
                invalidated_count += 1
        if invalidated_count > 0:
            self._save_index()
        return invalidated_count

    @async_with_exception_handling
    async def clear(self) ->int:
        """
        Clear all items from the cache.
        
        Returns:
            Number of cleared cache entries
        """
        count = len(self._index)
        self._index = {}
        self._current_size = 0
        self._save_index()
        try:
            shutil.rmtree(self._data_dir)
            os.makedirs(self._data_dir, exist_ok=True)
        except Exception as e:
            print(f'Error clearing disk cache: {e}')
        return count

    def get_stats(self) ->Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        expired_count = 0
        now = time.time()
        for expires_at, _, _ in self._index.values():
            if expires_at is not None and now > expires_at:
                expired_count += 1
        return {'size_bytes': self._current_size, 'max_size_bytes': self.
            _max_size, 'utilization': self._current_size / self._max_size if
            self._max_size > 0 else 0, 'item_count': len(self._index),
            'expired_count': expired_count, 'directory': self._base_dir}

    @property
    def size(self) ->int:
        """
        Get the current size of the cache in bytes.
        
        Returns:
            Size in bytes
        """
        return self._current_size

    @property
    def max_size(self) ->int:
        """
        Get the maximum size of the cache in bytes.
        
        Returns:
            Maximum size in bytes
        """
        return self._max_size

    @property
    def item_count(self) ->int:
        """
        Get the number of items in the cache.
        
        Returns:
            Number of items
        """
        return len(self._index)
