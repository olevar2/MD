"""
Tests for the cache implementations.
"""
import unittest
import asyncio
import time
from unittest.mock import MagicMock, patch, AsyncMock

from common_lib.caching.cache import CacheKey, InMemoryCache, RedisCache, CacheFactory


class TestCacheKey(unittest.TestCase):
    """
    Tests for the CacheKey class.
    """
    
    def test_generate_with_simple_args(self):
        """Test generating a cache key with simple arguments."""
        key = CacheKey.generate("test", 1, "abc")
        self.assertEqual(key, "test:1:abc")
    
    def test_generate_with_dict(self):
        """Test generating a cache key with a dictionary."""
        key = CacheKey.generate("test", {"a": 1, "b": 2})
        self.assertEqual(key, 'test:{"a": 1, "b": 2}')
    
    def test_generate_with_list(self):
        """Test generating a cache key with a list."""
        key = CacheKey.generate("test", [1, 2, 3])
        self.assertEqual(key, 'test:[1, 2, 3]')
    
    def test_generate_with_mixed_args(self):
        """Test generating a cache key with mixed arguments."""
        key = CacheKey.generate("test", 1, {"a": 1}, [1, 2, 3])
        self.assertEqual(key, 'test:1:{"a": 1}:[1, 2, 3]')


class TestInMemoryCache(unittest.TestCase):
    """
    Tests for the InMemoryCache class.
    """
    
    def setUp(self):
        """Set up the test case."""
        self.cache = InMemoryCache()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Tear down the test case."""
        self.loop.close()
    
    def test_get_nonexistent_key(self):
        """Test getting a nonexistent key."""
        result = self.loop.run_until_complete(self.cache.get("nonexistent"))
        self.assertIsNone(result)
    
    def test_set_and_get(self):
        """Test setting and getting a value."""
        self.loop.run_until_complete(self.cache.set("test", "value"))
        result = self.loop.run_until_complete(self.cache.get("test"))
        self.assertEqual(result, "value")
    
    def test_set_with_ttl_and_get_before_expiry(self):
        """Test setting a value with TTL and getting it before it expires."""
        self.loop.run_until_complete(self.cache.set("test", "value", ttl=10))
        result = self.loop.run_until_complete(self.cache.get("test"))
        self.assertEqual(result, "value")
    
    def test_set_with_ttl_and_get_after_expiry(self):
        """Test setting a value with TTL and getting it after it expires."""
        self.loop.run_until_complete(self.cache.set("test", "value", ttl=1))
        time.sleep(1.1)  # Wait for the value to expire
        result = self.loop.run_until_complete(self.cache.get("test"))
        self.assertIsNone(result)
    
    def test_delete(self):
        """Test deleting a value."""
        self.loop.run_until_complete(self.cache.set("test", "value"))
        self.loop.run_until_complete(self.cache.delete("test"))
        result = self.loop.run_until_complete(self.cache.get("test"))
        self.assertIsNone(result)
    
    def test_exists(self):
        """Test checking if a key exists."""
        self.loop.run_until_complete(self.cache.set("test", "value"))
        result = self.loop.run_until_complete(self.cache.exists("test"))
        self.assertTrue(result)
        
        result = self.loop.run_until_complete(self.cache.exists("nonexistent"))
        self.assertFalse(result)
    
    def test_clear(self):
        """Test clearing the cache."""
        self.loop.run_until_complete(self.cache.set("test1", "value1"))
        self.loop.run_until_complete(self.cache.set("test2", "value2"))
        self.loop.run_until_complete(self.cache.clear())
        
        result1 = self.loop.run_until_complete(self.cache.get("test1"))
        result2 = self.loop.run_until_complete(self.cache.get("test2"))
        
        self.assertIsNone(result1)
        self.assertIsNone(result2)


class TestRedisCache(unittest.TestCase):
    """
    Tests for the RedisCache class.
    """
    
    def setUp(self):
        """Set up the test case."""
        self.redis_client = MagicMock()
        # Configure the mock to return awaitable values
        self.redis_client.get = AsyncMock()
        self.redis_client.set = AsyncMock()
        self.redis_client.setex = AsyncMock()
        self.redis_client.delete = AsyncMock()
        self.redis_client.exists = AsyncMock()
        
        self.cache = RedisCache(self.redis_client)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Tear down the test case."""
        self.loop.close()
    
    def test_get_nonexistent_key(self):
        """Test getting a nonexistent key."""
        self.redis_client.get.return_value = None
        result = self.loop.run_until_complete(self.cache.get("nonexistent"))
        self.assertIsNone(result)
        self.redis_client.get.assert_called_once_with("nonexistent")
    
    def test_get_existing_key(self):
        """Test getting an existing key."""
        self.redis_client.get.return_value = '{"key": "value"}'
        result = self.loop.run_until_complete(self.cache.get("test"))
        self.assertEqual(result, {"key": "value"})
        self.redis_client.get.assert_called_once_with("test")
    
    def test_get_invalid_json(self):
        """Test getting a value that is not valid JSON."""
        self.redis_client.get.return_value = "not json"
        result = self.loop.run_until_complete(self.cache.get("test"))
        self.assertIsNone(result)
        self.redis_client.get.assert_called_once_with("test")
    
    def test_set_without_ttl(self):
        """Test setting a value without TTL."""
        self.loop.run_until_complete(self.cache.set("test", {"key": "value"}))
        self.redis_client.set.assert_called_once_with("test", '{"key": "value"}')
    
    def test_set_with_ttl(self):
        """Test setting a value with TTL."""
        self.loop.run_until_complete(self.cache.set("test", {"key": "value"}, ttl=60))
        self.redis_client.setex.assert_called_once_with("test", 60, '{"key": "value"}')
    
    def test_delete(self):
        """Test deleting a value."""
        self.loop.run_until_complete(self.cache.delete("test"))
        self.redis_client.delete.assert_called_once_with("test")
    
    def test_exists(self):
        """Test checking if a key exists."""
        self.redis_client.exists.return_value = 1
        result = self.loop.run_until_complete(self.cache.exists("test"))
        self.assertTrue(result)
        self.redis_client.exists.assert_called_once_with("test")
        
        self.redis_client.exists.reset_mock()
        self.redis_client.exists.return_value = 0
        result = self.loop.run_until_complete(self.cache.exists("nonexistent"))
        self.assertFalse(result)
        self.redis_client.exists.assert_called_once_with("nonexistent")
    
    def test_clear(self):
        """Test clearing the cache."""
        with self.assertLogs(level='WARNING') as cm:
            self.loop.run_until_complete(self.cache.clear())
            self.assertIn("RedisCache.clear() is not implemented for safety reasons", cm.output[0])


class TestCacheFactory(unittest.TestCase):
    """
    Tests for the CacheFactory class.
    """
    
    def test_create_memory_cache(self):
        """Test creating an in-memory cache."""
        cache = CacheFactory.create_memory_cache()
        self.assertIsInstance(cache, InMemoryCache)
    
    @patch('redis.asyncio.from_url')
    def test_create_redis_cache(self, mock_from_url):
        """Test creating a Redis cache."""
        mock_redis_client = MagicMock()
        mock_from_url.return_value = mock_redis_client
        
        cache = CacheFactory.create_redis_cache("redis://localhost:6379/0")
        
        self.assertIsInstance(cache, RedisCache)
        mock_from_url.assert_called_once_with("redis://localhost:6379/0")


if __name__ == '__main__':
    unittest.main()