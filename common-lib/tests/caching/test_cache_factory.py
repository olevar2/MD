"""
Tests for the cache factory module.
"""
import unittest
from unittest.mock import patch, MagicMock
import os

from common_lib.caching.cache_factory import CacheFactory, create_cache
from common_lib.caching.memory_cache import MemoryCache
from common_lib.caching.redis_cache import RedisCache


class TestCacheFactory(unittest.TestCase):
    """Tests for the CacheFactory class."""
    
    def test_create_memory_cache(self):
        """Test creating a memory cache."""
        # Create a memory cache
        cache = create_cache(
            cache_type="memory",
            service_name="test_service",
            ttl=3600
        )
        
        # Check that it's a MemoryCache
        self.assertIsInstance(cache, MemoryCache)
    
    @patch("common_lib.caching.redis_cache.redis.Redis")
    @patch("common_lib.caching.redis_cache.AsyncRedis")
    def test_create_redis_cache(self, mock_async_redis, mock_redis):
        """Test creating a Redis cache."""
        # Create a Redis cache
        cache = create_cache(
            cache_type="redis",
            service_name="test_service",
            ttl=3600
        )
        
        # Check that it's a RedisCache
        self.assertIsInstance(cache, RedisCache)
        
        # Check that Redis was initialized
        mock_redis.assert_called_once()
        mock_async_redis.assert_called_once()
    
    @patch.dict(os.environ, {"REDIS_HOST": "redis.example.com", "REDIS_PORT": "6380", "REDIS_DB": "1"})
    @patch("common_lib.caching.redis_cache.redis.Redis")
    @patch("common_lib.caching.redis_cache.AsyncRedis")
    def test_create_redis_cache_from_env(self, mock_async_redis, mock_redis):
        """Test creating a Redis cache from environment variables."""
        # Create a Redis cache
        cache = create_cache(
            cache_type="redis",
            service_name="test_service",
            ttl=3600
        )
        
        # Check that it's a RedisCache
        self.assertIsInstance(cache, RedisCache)
        
        # Check that Redis was initialized with the correct parameters
        mock_redis.assert_called_once_with(
            host="redis.example.com",
            port=6380,
            db=1,
            password=None,
            decode_responses=True
        )
    
    def test_create_unknown_cache(self):
        """Test creating an unknown cache type."""
        # Try to create an unknown cache type
        with self.assertRaises(ValueError):
            create_cache(
                cache_type="unknown",
                service_name="test_service",
                ttl=3600
            )


if __name__ == "__main__":
    unittest.main()