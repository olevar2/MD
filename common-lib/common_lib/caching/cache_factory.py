"""
Cache factory for the forex trading platform.
"""

import logging
import os
from typing import Any, Dict, Optional, Type, Union

from common_lib.caching.cache_interface import CacheInterface
from common_lib.caching.memory_cache import MemoryCache
from common_lib.caching.redis_cache import RedisCache

logger = logging.getLogger(__name__)


class CacheFactory:
    """
    Cache factory for creating cache instances.
    """

    @staticmethod
    def create_cache(
        cache_type: str = "memory",
        service_name: str = "default",
        prefix: Optional[str] = None,
        ttl: int = 3600,
        **kwargs: Any
    ) -> CacheInterface:
        """
        Create a cache instance.

        Args:
            cache_type: Type of cache to create (memory or redis)
            service_name: Name of the service
            prefix: Key prefix
            ttl: Default TTL in seconds
            **kwargs: Additional arguments for the cache

        Returns:
            Cache instance
        """
        # Set default prefix if not provided
        if prefix is None:
            prefix = f"{service_name}:"

        # Create cache based on type
        if cache_type.lower() == "memory":
            logger.info(f"Creating in-memory cache for {service_name}")
            return MemoryCache(
                prefix=prefix,
                ttl=ttl,
                **kwargs
            )
        elif cache_type.lower() == "redis":
            logger.info(f"Creating Redis cache for {service_name}")
            
            # Get Redis configuration from environment variables or kwargs
            host = kwargs.get("host") or os.environ.get("REDIS_HOST", "localhost")
            port = int(kwargs.get("port") or os.environ.get("REDIS_PORT", "6379"))
            db = int(kwargs.get("db") or os.environ.get("REDIS_DB", "0"))
            password = kwargs.get("password") or os.environ.get("REDIS_PASSWORD")
            
            return RedisCache(
                host=host,
                port=port,
                db=db,
                password=password,
                prefix=prefix,
                ttl=ttl,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")


def create_cache(
    cache_type: str = "memory",
    service_name: str = "default",
    prefix: Optional[str] = None,
    ttl: int = 3600,
    **kwargs: Any
) -> CacheInterface:
    """
    Create a cache instance.

    Args:
        cache_type: Type of cache to create (memory or redis)
        service_name: Name of the service
        prefix: Key prefix
        ttl: Default TTL in seconds
        **kwargs: Additional arguments for the cache

    Returns:
        Cache instance
    """
    return CacheFactory.create_cache(
        cache_type=cache_type,
        service_name=service_name,
        prefix=prefix,
        ttl=ttl,
        **kwargs
    )