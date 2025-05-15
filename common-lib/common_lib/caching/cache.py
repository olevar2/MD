"""
Cache interfaces and implementations.

This module provides the base interfaces and implementations for caching
in the forex trading platform.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, TypeVar, Generic, Union
import json
import logging
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar('T')

class CacheKey:
    """
    Utility class for generating cache keys.
    """
    @staticmethod
    def generate(prefix: str, *args: Any) -> str:
        """
        Generate a cache key.
        
        Args:
            prefix: The prefix for the key
            *args: The arguments to include in the key
            
        Returns:
            The generated cache key
        """
        key_parts = [prefix]
        for arg in args:
            if isinstance(arg, dict):
                # Sort dictionary keys for consistent key generation
                key_parts.append(json.dumps(arg, sort_keys=True))
            elif isinstance(arg, (list, tuple)):
                key_parts.append(json.dumps(arg))
            elif isinstance(arg, (datetime, timedelta)):
                key_parts.append(str(arg))
            else:
                key_parts.append(str(arg))
        
        return ":".join(key_parts)


class Cache(Generic[T], ABC):
    """
    Base interface for cache implementations.
    """
    @abstractmethod
    async def get(self, key: str) -> Optional[T]:
        """
        Get a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cached value or None if not found
        """
        pass
    
    @abstractmethod
    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: Time to live in seconds (optional)
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """
        Delete a value from the cache.
        
        Args:
            key: The cache key
        """
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache.
        
        Args:
            key: The cache key
            
        Returns:
            True if the key exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """
        Clear the cache.
        """
        pass


class InMemoryCache(Cache[T]):
    """
    In-memory cache implementation.
    """
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    async def get(self, key: str) -> Optional[T]:
        """
        Get a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cached value or None if not found
        """
        if key not in self._cache:
            return None
        
        cache_entry = self._cache[key]
        
        # Check if the entry has expired
        if 'expiry' in cache_entry and cache_entry['expiry'] < time.time():
            # Entry has expired, remove it
            del self._cache[key]
            return None
        
        return cache_entry['value']
    
    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: Time to live in seconds (optional)
        """
        cache_entry = {'value': value}
        
        if ttl is not None:
            cache_entry['expiry'] = time.time() + ttl
        
        self._cache[key] = cache_entry
    
    async def delete(self, key: str) -> None:
        """
        Delete a value from the cache.
        
        Args:
            key: The cache key
        """
        if key in self._cache:
            del self._cache[key]
    
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache.
        
        Args:
            key: The cache key
            
        Returns:
            True if the key exists, False otherwise
        """
        if key not in self._cache:
            return False
        
        cache_entry = self._cache[key]
        
        # Check if the entry has expired
        if 'expiry' in cache_entry and cache_entry['expiry'] < time.time():
            # Entry has expired, remove it
            del self._cache[key]
            return False
        
        return True
    
    async def clear(self) -> None:
        """
        Clear the cache.
        """
        self._cache.clear()


class RedisCache(Cache[T]):
    """
    Redis cache implementation.
    """
    def __init__(self, redis_client):
        """
        Initialize the Redis cache.
        
        Args:
            redis_client: The Redis client
        """
        self._redis = redis_client
    
    async def get(self, key: str) -> Optional[T]:
        """
        Get a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cached value or None if not found
        """
        value = await self._redis.get(key)
        if value is None:
            return None
        
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode cached value for key {key}")
            return None
    
    async def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: Time to live in seconds (optional)
        """
        serialized_value = json.dumps(value)
        if ttl is not None:
            await self._redis.setex(key, ttl, serialized_value)
        else:
            await self._redis.set(key, serialized_value)
    
    async def delete(self, key: str) -> None:
        """
        Delete a value from the cache.
        
        Args:
            key: The cache key
        """
        await self._redis.delete(key)
    
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache.
        
        Args:
            key: The cache key
            
        Returns:
            True if the key exists, False otherwise
        """
        return await self._redis.exists(key) > 0
    
    async def clear(self) -> None:
        """
        Clear the cache.
        """
        # This is a potentially dangerous operation, so we'll just log a warning
        logger.warning("RedisCache.clear() is not implemented for safety reasons")


class CacheFactory:
    """
    Factory for creating cache instances.
    """
    @staticmethod
    def create_memory_cache() -> InMemoryCache:
        """
        Create an in-memory cache.
        
        Returns:
            An in-memory cache instance
        """
        return InMemoryCache()
    
    @staticmethod
    def create_redis_cache(redis_url: str) -> RedisCache:
        """
        Create a Redis cache.
        
        Args:
            redis_url: The Redis URL
            
        Returns:
            A Redis cache instance
        """
        import redis.asyncio as redis
        redis_client = redis.from_url(redis_url)
        return RedisCache(redis_client)