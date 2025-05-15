"""
Cache Factory for backtesting-service

This module provides a factory for creating cache instances.
"""
import logging
from typing import Optional

from common_lib.caching.cache import Cache
from common_lib.caching.redis_cache import RedisCache
from common_lib.config.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class CacheFactory:
    """
    Factory for creating cache instances.
    """
    _instance: Optional[Cache] = None
    
    def get_cache(self) -> Cache:
        """
        Get a cache instance.
        
        Returns:
            A cache instance
        """
        if self._instance is None:
            config = ConfigManager().get_config()
            cache_config = config.get("cache", {})
            
            # Use Redis cache if configured
            if cache_config.get("type") == "redis":
                redis_config = cache_config.get("redis", {})
                host = redis_config.get("host", "localhost")
                port = redis_config.get("port", 6379)
                db = redis_config.get("db", 0)
                
                logger.info(f"Creating Redis cache with host={host}, port={port}, db={db}")
                self._instance = RedisCache(host=host, port=port, db=db)
            else:
                # Use in-memory cache by default
                logger.info("Creating in-memory cache")
                from common_lib.caching.memory_cache import MemoryCache
                self._instance = MemoryCache()
        
        return self._instance

# Create a singleton instance
cache_factory = CacheFactory()
