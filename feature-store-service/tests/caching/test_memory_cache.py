"""
Tests for the LRUCache class (memory cache) in the feature store caching system.
"""
import asyncio
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from feature_store_service.caching.memory_cache import LRUCache
from feature_store_service.caching.cache_key import CacheKey


class TestLRUCache(unittest.TestCase):
    """Test cases for LRUCache (memory cache) class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.cache = LRUCache(max_size=1000000, default_ttl_seconds=300)
        
        # Create a test key
        self.test_key = CacheKey(
            indicator_type="SMA",
            params={"window": 20},
            symbol="EURUSD",
            timeframe="1h",
            start_time=datetime(2025, 1, 1, 0, 0, 0),
            end_time=datetime(2025, 1, 1, 23, 59, 59)
        )
        
        # Create a test DataFrame
        self.test_data = pd.DataFrame({
            'open': np.random.rand(100),
            'high': np.random.rand(100),
            'low': np.random.rand(100),
            'close': np.random.rand(100),
            'volume': np.random.rand(100),
        })

    def test_put_and_get(self):
        """Test putting and getting items from the cache."""
        # Define test case function
        async def test_case():
            # Put item in cache
            success = await self.cache.put(self.test_key, self.test_data)
            self.assertTrue(success)
            
            # Get item from cache
            result = await self.cache.get(self.test_key)
            
            # Verify result
            self.assertIsNotNone(result)
            pd.testing.assert_frame_equal(result, self.test_data)
            
        # Run test case
        asyncio.run(test_case())

    def test_ttl_expiration(self):
        """Test that items expire based on TTL."""
        # Define test case function
        async def test_case():
            # Put item with a short TTL (1 second)
            await self.cache.put(self.test_key, self.test_data, ttl_seconds=1)
            
            # Verify it exists immediately
            result_before = await self.cache.get(self.test_key)
            self.assertIsNotNone(result_before)
            
            # Wait for expiration
            await asyncio.sleep(1.5)
            
            # Verify it's gone after expiration
            result_after = await self.cache.get(self.test_key)
            self.assertIsNone(result_after)
            
        # Run test case
        asyncio.run(test_case())

    def test_lru_eviction(self):
        """Test that least recently used items are evicted when cache is full."""
        # Define test case function
        async def test_case():
            # Create a tiny cache
            small_cache = LRUCache(max_size=1000)  # Very small cache
            
            # Create keys
            key1 = CacheKey(
                indicator_type="SMA",
                params={"window": 10},
                symbol="EURUSD",
                timeframe="1h",
                start_time=datetime(2025, 1, 1, 0, 0, 0),
                end_time=datetime(2025, 1, 1, 23, 59, 59)
            )
            
            key2 = CacheKey(
                indicator_type="SMA",
                params={"window": 20},
                symbol="EURUSD",
                timeframe="1h",
                start_time=datetime(2025, 1, 1, 0, 0, 0),
                end_time=datetime(2025, 1, 1, 23, 59, 59)
            )
            
            # Create DataFrames that will fill the cache
            df1 = pd.DataFrame({'data': np.random.rand(50)})
            df2 = pd.DataFrame({'data': np.random.rand(50)})
            
            # Put first item
            await small_cache.put(key1, df1)
            
            # Verify it exists
            self.assertIsNotNone(await small_cache.get(key1))
            
            # Put second item that should push out the first due to size constraints
            await small_cache.put(key2, df2)
            
            # First item should be evicted
            self.assertIsNone(await small_cache.get(key1))
            
            # Second item should exist
            self.assertIsNotNone(await small_cache.get(key2))
            
        # Run test case
        asyncio.run(test_case())

    def test_invalidate(self):
        """Test invalidating cache entries based on pattern."""
        # Define test case function
        async def test_case():
            # Create multiple keys
            key1 = CacheKey(
                indicator_type="SMA",
                params={"window": 10},
                symbol="EURUSD",
                timeframe="1h",
                start_time=datetime(2025, 1, 1, 0, 0, 0),
                end_time=datetime(2025, 1, 1, 23, 59, 59)
            )
            
            key2 = CacheKey(
                indicator_type="EMA",
                params={"window": 20},
                symbol="EURUSD",
                timeframe="1h",
                start_time=datetime(2025, 1, 1, 0, 0, 0),
                end_time=datetime(2025, 1, 1, 23, 59, 59)
            )
            
            key3 = CacheKey(
                indicator_type="RSI",
                params={"window": 14},
                symbol="EURUSD",
                timeframe="1h",
                start_time=datetime(2025, 1, 1, 0, 0, 0),
                end_time=datetime(2025, 1, 1, 23, 59, 59)
            )
            
            # Put items in cache
            await self.cache.put(key1, "data1")
            await self.cache.put(key2, "data2")
            await self.cache.put(key3, "data3")
            
            # Verify all items exist
            self.assertIsNotNone(await self.cache.get(key1))
            self.assertIsNotNone(await self.cache.get(key2))
            self.assertIsNotNone(await self.cache.get(key3))
            
            # Invalidate SMA entries
            count = await self.cache.invalidate("^SMA:")
            
            # Verify one item was invalidated
            self.assertEqual(count, 1)
            
            # Verify SMA is gone but others remain
            self.assertIsNone(await self.cache.get(key1))
            self.assertIsNotNone(await self.cache.get(key2))
            self.assertIsNotNone(await self.cache.get(key3))
            
            # Clear all
            count = await self.cache.clear()
            
            # Verify all items were cleared
            self.assertEqual(count, 2)
            self.assertIsNone(await self.cache.get(key2))
            self.assertIsNone(await self.cache.get(key3))
            
        # Run test case
        asyncio.run(test_case())

    def test_get_stats(self):
        """Test getting cache statistics."""
        # Define test case function
        async def test_case():
            # Put items in cache
            await self.cache.put(self.test_key, self.test_data)
            
            # Get stats
            stats = self.cache.get_stats()
            
            # Verify stats format
            self.assertIn('size_bytes', stats)
            self.assertIn('max_size_bytes', stats)
            self.assertIn('utilization', stats)
            self.assertIn('item_count', stats)
            
            # Verify item count
            self.assertEqual(stats['item_count'], 1)
            
        # Run test case
        asyncio.run(test_case())


if __name__ == "__main__":
    unittest.main()
