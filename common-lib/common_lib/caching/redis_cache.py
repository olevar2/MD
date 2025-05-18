"""
Redis cache implementation for the forex trading platform.
"""

import logging
import json
import pickle
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
import asyncio
import redis
from redis.asyncio import Redis as AsyncRedis

from common_lib.errors.decorators import (
    with_exception_handling,
    async_with_exception_handling
)

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis cache implementation.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "cache:",
        ttl: int = 3600,
        serializer: str = "json"
    ):
        """
        Initialize the Redis cache.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database
            password: Redis password
            prefix: Key prefix
            ttl: Default TTL in seconds
            serializer: Serializer to use (json or pickle)
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.prefix = prefix
        self.ttl = ttl
        self.serializer = serializer

        # Initialize Redis clients
        self.redis = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=(serializer == "json")
        )
        self.async_redis = AsyncRedis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=(serializer == "json")
        )

        logger.info(f"Initialized Redis cache at {host}:{port}/{db}")

    def _get_key(self, key: str) -> str:
        """
        Get the full key with prefix.

        Args:
            key: Key to get

        Returns:
            Full key with prefix
        """
        return f"{self.prefix}{key}"

    def _serialize(self, value: Any) -> Union[str, bytes]:
        """
        Serialize a value.

        Args:
            value: Value to serialize

        Returns:
            Serialized value
        """
        if self.serializer == "json":
            return json.dumps(value)
        else:
            return pickle.dumps(value)

    def _deserialize(self, value: Union[str, bytes]) -> Any:
        """
        Deserialize a value.

        Args:
            value: Value to deserialize

        Returns:
            Deserialized value
        """
        if value is None:
            return None

        if self.serializer == "json":
            return json.loads(value)
        else:
            return pickle.loads(value)

    @with_exception_handling
    def get(self, key: str) -> Any:
        """
        Get a value from the cache.

        Args:
            key: Key to get

        Returns:
            Value from the cache
        """
        full_key = self._get_key(key)
        value = self.redis.get(full_key)
        return self._deserialize(value)

    @with_exception_handling
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in the cache.

        Args:
            key: Key to set
            value: Value to set
            ttl: TTL in seconds

        Returns:
            True if the value was set
        """
        full_key = self._get_key(key)
        serialized_value = self._serialize(value)
        return self.redis.set(full_key, serialized_value, ex=(ttl or self.ttl))

    @with_exception_handling
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.

        Args:
            key: Key to delete

        Returns:
            True if the value was deleted
        """
        full_key = self._get_key(key)
        return bool(self.redis.delete(full_key))

    @with_exception_handling
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache.

        Args:
            key: Key to check

        Returns:
            True if the key exists
        """
        full_key = self._get_key(key)
        return bool(self.redis.exists(full_key))

    @with_exception_handling
    def increment(self, key: str, amount: int = 1) -> int:
        """
        Increment a value in the cache.

        Args:
            key: Key to increment
            amount: Amount to increment by

        Returns:
            New value
        """
        full_key = self._get_key(key)
        return self.redis.incrby(full_key, amount)

    @with_exception_handling
    def decrement(self, key: str, amount: int = 1) -> int:
        """
        Decrement a value in the cache.

        Args:
            key: Key to decrement
            amount: Amount to decrement by

        Returns:
            New value
        """
        full_key = self._get_key(key)
        return self.redis.decrby(full_key, amount)

    @with_exception_handling
    def expire(self, key: str, ttl: int) -> bool:
        """
        Set the TTL for a key.

        Args:
            key: Key to set TTL for
            ttl: TTL in seconds

        Returns:
            True if the TTL was set
        """
        full_key = self._get_key(key)
        return bool(self.redis.expire(full_key, ttl))

    @with_exception_handling
    def ttl(self, key: str) -> int:
        """
        Get the TTL for a key.

        Args:
            key: Key to get TTL for

        Returns:
            TTL in seconds
        """
        full_key = self._get_key(key)
        return self.redis.ttl(full_key)

    @with_exception_handling
    def keys(self, pattern: str = "*") -> List[str]:
        """
        Get keys matching a pattern.

        Args:
            pattern: Pattern to match

        Returns:
            List of keys
        """
        full_pattern = self._get_key(pattern)
        keys = self.redis.keys(full_pattern)
        
        # Remove prefix from keys
        if isinstance(keys[0], bytes):
            return [key.decode("utf-8")[len(self.prefix):] for key in keys]
        else:
            return [key[len(self.prefix):] for key in keys]

    @with_exception_handling
    def flush(self) -> bool:
        """
        Flush all keys with the prefix.

        Returns:
            True if the keys were flushed
        """
        keys = self.redis.keys(self._get_key("*"))
        if keys:
            return bool(self.redis.delete(*keys))
        return True

    @with_exception_handling
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values from the cache.

        Args:
            keys: Keys to get

        Returns:
            Dictionary of keys and values
        """
        full_keys = [self._get_key(key) for key in keys]
        values = self.redis.mget(full_keys)
        
        result = {}
        for i, key in enumerate(keys):
            result[key] = self._deserialize(values[i])
        
        return result

    @with_exception_handling
    def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Set multiple values in the cache.

        Args:
            mapping: Dictionary of keys and values
            ttl: TTL in seconds

        Returns:
            True if the values were set
        """
        full_mapping = {}
        for key, value in mapping.items():
            full_mapping[self._get_key(key)] = self._serialize(value)
        
        pipeline = self.redis.pipeline()
        pipeline.mset(full_mapping)
        
        if ttl is not None or self.ttl is not None:
            for key in mapping.keys():
                pipeline.expire(self._get_key(key), ttl or self.ttl)
        
        pipeline.execute()
        return True

    @with_exception_handling
    def delete_many(self, keys: List[str]) -> int:
        """
        Delete multiple values from the cache.

        Args:
            keys: Keys to delete

        Returns:
            Number of keys deleted
        """
        full_keys = [self._get_key(key) for key in keys]
        return self.redis.delete(*full_keys)

    @async_with_exception_handling
    async def async_get(self, key: str) -> Any:
        """
        Get a value from the cache asynchronously.

        Args:
            key: Key to get

        Returns:
            Value from the cache
        """
        full_key = self._get_key(key)
        value = await self.async_redis.get(full_key)
        return self._deserialize(value)

    @async_with_exception_handling
    async def async_set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in the cache asynchronously.

        Args:
            key: Key to set
            value: Value to set
            ttl: TTL in seconds

        Returns:
            True if the value was set
        """
        full_key = self._get_key(key)
        serialized_value = self._serialize(value)
        return await self.async_redis.set(full_key, serialized_value, ex=(ttl or self.ttl))

    @async_with_exception_handling
    async def async_delete(self, key: str) -> bool:
        """
        Delete a value from the cache asynchronously.

        Args:
            key: Key to delete

        Returns:
            True if the value was deleted
        """
        full_key = self._get_key(key)
        return bool(await self.async_redis.delete(full_key))

    @async_with_exception_handling
    async def async_exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache asynchronously.

        Args:
            key: Key to check

        Returns:
            True if the key exists
        """
        full_key = self._get_key(key)
        return bool(await self.async_redis.exists(full_key))

    @async_with_exception_handling
    async def async_increment(self, key: str, amount: int = 1) -> int:
        """
        Increment a value in the cache asynchronously.

        Args:
            key: Key to increment
            amount: Amount to increment by

        Returns:
            New value
        """
        full_key = self._get_key(key)
        return await self.async_redis.incrby(full_key, amount)

    @async_with_exception_handling
    async def async_decrement(self, key: str, amount: int = 1) -> int:
        """
        Decrement a value in the cache asynchronously.

        Args:
            key: Key to decrement
            amount: Amount to decrement by

        Returns:
            New value
        """
        full_key = self._get_key(key)
        return await self.async_redis.decrby(full_key, amount)

    @async_with_exception_handling
    async def async_expire(self, key: str, ttl: int) -> bool:
        """
        Set the TTL for a key asynchronously.

        Args:
            key: Key to set TTL for
            ttl: TTL in seconds

        Returns:
            True if the TTL was set
        """
        full_key = self._get_key(key)
        return bool(await self.async_redis.expire(full_key, ttl))

    @async_with_exception_handling
    async def async_ttl(self, key: str) -> int:
        """
        Get the TTL for a key asynchronously.

        Args:
            key: Key to get TTL for

        Returns:
            TTL in seconds
        """
        full_key = self._get_key(key)
        return await self.async_redis.ttl(full_key)

    @async_with_exception_handling
    async def async_keys(self, pattern: str = "*") -> List[str]:
        """
        Get keys matching a pattern asynchronously.

        Args:
            pattern: Pattern to match

        Returns:
            List of keys
        """
        full_pattern = self._get_key(pattern)
        keys = await self.async_redis.keys(full_pattern)
        
        # Remove prefix from keys
        if isinstance(keys[0], bytes):
            return [key.decode("utf-8")[len(self.prefix):] for key in keys]
        else:
            return [key[len(self.prefix):] for key in keys]

    @async_with_exception_handling
    async def async_flush(self) -> bool:
        """
        Flush all keys with the prefix asynchronously.

        Returns:
            True if the keys were flushed
        """
        keys = await self.async_redis.keys(self._get_key("*"))
        if keys:
            return bool(await self.async_redis.delete(*keys))
        return True

    @async_with_exception_handling
    async def async_get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values from the cache asynchronously.

        Args:
            keys: Keys to get

        Returns:
            Dictionary of keys and values
        """
        full_keys = [self._get_key(key) for key in keys]
        values = await self.async_redis.mget(full_keys)
        
        result = {}
        for i, key in enumerate(keys):
            result[key] = self._deserialize(values[i])
        
        return result

    @async_with_exception_handling
    async def async_set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Set multiple values in the cache asynchronously.

        Args:
            mapping: Dictionary of keys and values
            ttl: TTL in seconds

        Returns:
            True if the values were set
        """
        full_mapping = {}
        for key, value in mapping.items():
            full_mapping[self._get_key(key)] = self._serialize(value)
        
        pipeline = self.async_redis.pipeline()
        pipeline.mset(full_mapping)
        
        if ttl is not None or self.ttl is not None:
            for key in mapping.keys():
                pipeline.expire(self._get_key(key), ttl or self.ttl)
        
        await pipeline.execute()
        return True

    @async_with_exception_handling
    async def async_delete_many(self, keys: List[str]) -> int:
        """
        Delete multiple values from the cache asynchronously.

        Args:
            keys: Keys to delete

        Returns:
            Number of keys deleted
        """
        full_keys = [self._get_key(key) for key in keys]
        return await self.async_redis.delete(*full_keys)