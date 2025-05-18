"""
Tests for the cache decorators.
"""
import unittest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

from common_lib.caching.decorators import cached, invalidate_cache
from common_lib.caching.cache import Cache


class TestCachedDecorator(unittest.TestCase):
    """
    Tests for the cached decorator.
    """
    
    def setUp(self):
        """Set up the test case."""
        self.cache = MagicMock(spec=Cache)
        # Configure the mock to return awaitable values
        self.cache.get = AsyncMock()
        self.cache.set = AsyncMock()
        self.cache.delete = AsyncMock()
        self.cache.exists = AsyncMock()
        
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Tear down the test case."""
        self.loop.close()
    
    def test_cached_async_function_cache_hit(self):
        """Test caching an asynchronous function with a cache hit."""
        # Set up the cache to return a value
        self.cache.get.return_value = "cached_value"
        
        # Define an async function to cache
        @cached(self.cache, "test")
        async def test_function(arg1, arg2):
            return f"{arg1}_{arg2}"
        
        # Call the function
        result = self.loop.run_until_complete(test_function("a", "b"))
        
        # Check that the cache was queried
        self.cache.get.assert_called_once()
        
        # Check that the function returned the cached value
        self.assertEqual(result, "cached_value")
        
        # Check that the function was not called (since the value was in the cache)
        self.cache.set.assert_not_called()
    
    def test_cached_async_function_cache_miss(self):
        """Test caching an asynchronous function with a cache miss."""
        # Set up the cache to return None (cache miss)
        self.cache.get.return_value = None
        
        # Define an async function to cache
        @cached(self.cache, "test")
        async def test_function(arg1, arg2):
            return f"{arg1}_{arg2}"
        
        # Call the function
        result = self.loop.run_until_complete(test_function("a", "b"))
        
        # Check that the cache was queried
        self.cache.get.assert_called_once()
        
        # Check that the function returned the correct value
        self.assertEqual(result, "a_b")
        
        # Check that the result was cached
        self.cache.set.assert_called_once()
    
    def test_cached_async_function_with_ttl(self):
        """Test caching an asynchronous function with a TTL."""
        # Set up the cache to return None (cache miss)
        self.cache.get.return_value = None
        
        # Define an async function to cache
        @cached(self.cache, "test", ttl=60)
        async def test_function(arg1, arg2):
            return f"{arg1}_{arg2}"
        
        # Call the function
        result = self.loop.run_until_complete(test_function("a", "b"))
        
        # Check that the cache was queried
        self.cache.get.assert_called_once()
        
        # Check that the function returned the correct value
        self.assertEqual(result, "a_b")
        
        # Check that the result was cached with the correct TTL
        self.cache.set.assert_called_once()
        self.assertEqual(self.cache.set.call_args[0][2], 60)
    
    def test_cached_async_function_with_key_generator(self):
        """Test caching an asynchronous function with a custom key generator."""
        # Set up the cache to return None (cache miss)
        self.cache.get.return_value = None
        
        # Define a key generator
        def key_generator(arg1, arg2):
            return f"custom:{arg1}:{arg2}"
        
        # Define an async function to cache
        @cached(self.cache, "test", key_generator=key_generator)
        async def test_function(arg1, arg2):
            return f"{arg1}_{arg2}"
        
        # Call the function
        result = self.loop.run_until_complete(test_function("a", "b"))
        
        # Check that the cache was queried with the custom key
        self.cache.get.assert_called_once_with("custom:a:b")
        
        # Check that the function returned the correct value
        self.assertEqual(result, "a_b")
        
        # Check that the result was cached with the custom key
        self.cache.set.assert_called_once()
        self.assertEqual(self.cache.set.call_args[0][0], "custom:a:b")


class TestInvalidateCacheDecorator(unittest.TestCase):
    """
    Tests for the invalidate_cache decorator.
    """
    
    def setUp(self):
        """Set up the test case."""
        self.cache = MagicMock(spec=Cache)
        # Configure the mock to return awaitable values
        self.cache.get = AsyncMock()
        self.cache.set = AsyncMock()
        self.cache.delete = AsyncMock()
        self.cache.exists = AsyncMock()
        
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Tear down the test case."""
        self.loop.close()
    
    def test_invalidate_cache_async_function(self):
        """Test invalidating the cache for an asynchronous function."""
        # Define an async function to invalidate the cache
        @invalidate_cache(self.cache, "test")
        async def test_function(arg1, arg2):
            return f"{arg1}_{arg2}"
        
        # Call the function
        result = self.loop.run_until_complete(test_function("a", "b"))
        
        # Check that the function returned the correct value
        self.assertEqual(result, "a_b")
        
        # Check that the cache was invalidated
        self.cache.delete.assert_called_once()
    
    def test_invalidate_cache_async_function_with_key_generator(self):
        """Test invalidating the cache for an asynchronous function with a custom key generator."""
        # Define a key generator
        def key_generator(arg1, arg2):
            return f"custom:{arg1}:{arg2}"
        
        # Define an async function to invalidate the cache
        @invalidate_cache(self.cache, "test", key_generator=key_generator)
        async def test_function(arg1, arg2):
            return f"{arg1}_{arg2}"
        
        # Call the function
        result = self.loop.run_until_complete(test_function("a", "b"))
        
        # Check that the function returned the correct value
        self.assertEqual(result, "a_b")
        
        # Check that the cache was invalidated with the custom key
        self.cache.delete.assert_called_once_with("custom:a:b")


if __name__ == '__main__':
    unittest.main()