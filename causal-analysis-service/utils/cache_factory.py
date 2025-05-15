"""
Cache factory for the Causal Analysis Service.

This module provides a factory for creating cache instances.
"""
import logging
from typing import Optional

from common_lib.caching.cache import Cache, InMemoryCache, RedisCache
from causal_analysis_service.config.redis_config import redis_config

logger = logging.getLogger(__name__)

class CacheFactory:
    """
    Factory for creating cache instances.
    """
    _instance: Optional[CacheFactory] = None
    _cache: Optional[Cache] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CacheFactory, cls).__new__(cls)
        return cls._instance
    
    def get_cache(self) -> Cache:
        """
        Get a cache instance.
        
        Returns:
            A cache instance
        """
        if self._cache is None:
            if redis_config.enabled:
                try:
                    import redis.asyncio as redis
                    redis_client = redis.from_url(redis_config.url)
                    self._cache = RedisCache(redis_client)
                    logger.info(f"Using Redis cache at {redis_config.host}:{redis_config.port}")
                except ImportError:
                    logger.warning("Redis package not installed, falling back to in-memory cache")
                    self._cache = InMemoryCache()
                except Exception as e:
                    logger.warning(f"Failed to connect to Redis: {e}, falling back to in-memory cache")
                    self._cache = InMemoryCache()
            else:
                logger.info("Redis disabled, using in-memory cache")
                self._cache = InMemoryCache()
        
        return self._cache


# Create a singleton instance
cache_factory = CacheFactory()