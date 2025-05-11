import redis
import json
import functools
import hashlib
import inspect
from typing import Callable, Any, Optional, Union
from datetime import timedelta
from analysis_engine.core.config import settings
from analysis_engine.core.logging import logger

class CachingService:
    """Provides caching functionality using Redis."""

    def __init__(self, redis_url: str=settings.REDIS_URL):
        """Initializes the CachingService.

        Args:
            redis_url: The connection URL for the Redis instance.
        """
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info(f'Successfully connected to Redis at {redis_url}')
        except redis.exceptions.ConnectionError as e:
            logger.error(f'Failed to connect to Redis at {redis_url}: {e}')
            self.redis_client = None

    def _serialize(self, value: Any) -> str:
        """Serializes a Python object to a JSON string for storage in Redis."""
        try:
            return json.dumps(value)
        except TypeError as e:
            logger.warning(f'Could not serialize value for caching: {e}. Value type: {type(value)}')
            raise

    def _deserialize(self, value: Optional[str]) -> Any:
        """Deserializes a JSON string from Redis back into a Python object."""
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            logger.error(f'Could not deserialize cached value: {e}. Value: {value[:100]}...', exc_info=True)
            return None

    def get(self, key: str) -> Any:
        """Retrieves an item from the cache.

        Args:
            key: The cache key.

        Returns:
            The deserialized cached item, or None if not found or on error.
        """
        if not self.redis_client:
            logger.warning('Redis client not available. Skipping cache get.')
            return None
        try:
            cached_value = self.redis_client.get(key)
            if cached_value:
                logger.debug(f'Cache HIT for key: {key}')
                return self._deserialize(cached_value)
            else:
                logger.debug(f'Cache MISS for key: {key}')
                return None
        except redis.exceptions.RedisError as e:
            logger.error(f"Redis error getting key '{key}': {e}", exc_info=True)
            return None

    def set(self, key: str, value: Any, ttl: Optional[Union[int, timedelta]]=None) -> bool:
        """Stores an item in the cache.

        Args:
            key: The cache key.
            value: The Python object to cache.
            ttl: Time-to-live in seconds or as a timedelta. If None, cache indefinitely.

        Returns:
            True if the item was successfully set, False otherwise.
        """
        if not self.redis_client:
            logger.warning('Redis client not available. Skipping cache set.')
            return False
        try:
            serialized_value = self._serialize(value)
            if ttl:
                result = self.redis_client.setex(key, ttl, serialized_value)
            else:
                result = self.redis_client.set(key, serialized_value)
            logger.debug(f'Cache SET for key: {key} with TTL: {ttl}')
            return result
        except (redis.exceptions.RedisError, TypeError, json.JSONDecodeError) as e:
            logger.error(f"Redis error setting key '{key}': {e}", exc_info=True)
            return False

    def delete(self, key: str) -> bool:
        """Deletes an item from the cache.

        Args:
            key: The cache key.

        Returns:
            True if the key was deleted, False otherwise.
        """
        if not self.redis_client:
            logger.warning('Redis client not available. Skipping cache delete.')
            return False
        try:
            result = self.redis_client.delete(key)
            logger.debug(f'Cache DELETE for key: {key}')
            return bool(result)
        except redis.exceptions.RedisError as e:
            logger.error(f"Redis error deleting key '{key}': {e}", exc_info=True)
            return False

    def clear_prefix(self, prefix: str) -> int:
        """Deletes all keys matching a given prefix.

        Args:
            prefix: The prefix to match (e.g., 'analysis:indicator:*').

        Returns:
            The number of keys deleted.
        """
        if not self.redis_client:
            logger.warning('Redis client not available. Skipping cache clear_prefix.')
            return 0
        deleted_count = 0
        try:
            if not prefix.endswith('*'):
                prefix += '*'
            keys_to_delete = list(self.redis_client.scan_iter(match=prefix))
            if keys_to_delete:
                deleted_count = self.redis_client.delete(*keys_to_delete)
                logger.info(f'Cleared {deleted_count} cache keys with prefix: {prefix}')
            else:
                logger.debug(f'No cache keys found with prefix: {prefix}')
            return deleted_count
        except redis.exceptions.RedisError as e:
            logger.error(f"Redis error clearing prefix '{prefix}': {e}", exc_info=True)
            return 0
caching_service = CachingService()

def generate_cache_key(func: Callable, args: tuple, kwargs: dict) -> str:
    """Generates a unique cache key based on function and arguments."""
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    key_parts = [func.__module__, func.__name__]
    for k, v in bound_args.arguments.items():
        try:
            arg_repr = repr(v)
        except Exception:
            arg_repr = str(v)
        key_parts.append(f'{k}={arg_repr}')
    key_string = ':'.join(key_parts)
    return f'cache:{func.__module__}.{func.__name__}:{hashlib.sha256(key_string.encode()).hexdigest()}'

def cache_result(ttl: Optional[Union[int, timedelta]]=None):
    """Decorator to cache the result of a function.

    Args:
        ttl: Time-to-live for the cached item (in seconds or timedelta).
             If None, caches indefinitely (use with caution).
    """

    def decorator(func: Callable):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not caching_service.redis_client:
                logger.debug(f'Caching disabled for {func.__name__}, executing function directly.')
                return func(*args, **kwargs)
            cache_key = generate_cache_key(func, args, kwargs)
            cached_result = caching_service.get(cache_key)
            if cached_result is not None:
                return cached_result
            else:
                result = func(*args, **kwargs)
                caching_service.set(cache_key, result, ttl=ttl)
                return result
        wrapper._is_cached = True
        wrapper._cache_ttl = ttl
        wrapper._original_func = func
        return wrapper
    return decorator