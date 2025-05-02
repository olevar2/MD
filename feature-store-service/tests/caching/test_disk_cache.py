"""
Tests for the DiskCache class in the feature store caching system.
"""
import asyncio
import unittest
import os
import shutil
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from feature_store_service.caching.disk_cache import DiskCache
from feature_store_service.caching.cache_key import CacheKey


class TestDiskCache(unittest.TestCase):
    """Test cases for DiskCache class."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for the disk cache
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize the disk cache with the temporary directory
        self.cache = DiskCache(
            directory=self.temp_dir,
            max_size=1_000_000_000,  # 1GB
            default_ttl_seconds=300  # 5 minutes
        )
        
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
            'sma_20': np.random.rand(100),
        })

    def tearDown(self):
        """Clean up after each test."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_put_and_get(self):
        """Test putting and getting items from the disk cache."""
        # Define test case function
        async def test_case():
            # Put item in cache
            await self.cache.put(self.test_key.to_string(), self.test_data)
            
            # Get item from cache
            result = await self.cache.get(self.test_key.to_string())
            
            # Verify result
            self.assertIsNotNone(result)
            pd.testing.assert_frame_equal(result, self.test_data)
            
            # Check that the file exists on disk
            cache_files = os.listdir(self.temp_dir)
            self.assertGreater(len(cache_files), 0)
            
        # Run test case
        asyncio.run(test_case())

    def test_ttl_expiration(self):
        """Test that items expire based on TTL."""
        # Define test case function
        async def test_case():
            # Put item with a short TTL (1 second)
            await self.cache.put(self.test_key.to_string(), self.test_data, ttl_seconds=1)
            
            # Verify it exists immediately
            result_before = await self.cache.get(self.test_key.to_string())
            self.assertIsNotNone(result_before)
            
            # Wait for expiration
            await asyncio.sleep(1.5)
            
            # Verify it's gone after expiration
            result_after = await self.cache.get(self.test_key.to_string())
            self.assertIsNone(result_after)
            
        # Run test case
        asyncio.run(test_case())

    def test_size_management(self):
        """Test that the disk cache manages its size correctly."""
        # Define test case function
        async def test_case():
            # Create a small disk cache
            small_cache = DiskCache(
                directory=os.path.join(self.temp_dir, "small_cache"),
                max_size=1000,  # Very small max size
                default_ttl_seconds=300
            )
            
            # Create a DataFrame that will exceed the cache size
            large_data = pd.DataFrame({
                'data': np.random.rand(1000)  # Should exceed 1000 bytes when serialized
            })
            
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
                indicator_type="SMA",
                params={"window": 20},
                symbol="EURUSD",
                timeframe="1h",
                start_time=datetime(2025, 1, 1, 0, 0, 0),
                end_time=datetime(2025, 1, 1, 23, 59, 59)
            )
            
            # Put first item in cache
            await small_cache.put(key1.to_string(), large_data)
            
            # Get size after first item
            size_after_first = small_cache.size
            
            # Put second item - should trigger eviction due to size constraint
            await small_cache.put(key2.to_string(), large_data)
            
            # Verify the cache size is managed
            self.assertLessEqual(small_cache.size, small_cache.max_size * 1.1)  # Allow 10% overflow during cleanup
            
            # First item should be evicted
            self.assertIsNone(await small_cache.get(key1.to_string()))
            
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
            
            # Put items in cache
            await self.cache.put(key1.to_string(), "data1")
            await self.cache.put(key2.to_string(), "data2")
            
            # Verify both exist
            self.assertIsNotNone(await self.cache.get(key1.to_string()))
            self.assertIsNotNone(await self.cache.get(key2.to_string()))
            
            # Invalidate by pattern (SMA entries)
            count = await self.cache.invalidate("SMA:")
            self.assertEqual(count, 1)
            
            # Verify SMA is gone but EMA remains
            self.assertIsNone(await self.cache.get(key1.to_string()))
            self.assertIsNotNone(await self.cache.get(key2.to_string()))
            
        # Run test case
        asyncio.run(test_case())

    def test_clear(self):
        """Test clearing all cache entries."""
        # Define test case function
        async def test_case():
            # Put some items in cache
            key1 = "test_key1"
            key2 = "test_key2"
            await self.cache.put(key1, "data1")
            await self.cache.put(key2, "data2")
            
            # Verify they exist
            self.assertIsNotNone(await self.cache.get(key1))
            self.assertIsNotNone(await self.cache.get(key2))
            
            # Clear cache
            count = await self.cache.clear()
            self.assertEqual(count, 2)
            
            # Verify all items are gone
            self.assertIsNone(await self.cache.get(key1))
            self.assertIsNone(await self.cache.get(key2))
            
            # Verify directory is empty or only has metadata files
            # Some implementations might keep metadata files, so we don't check for exact emptiness
            cache_files = [f for f in os.listdir(self.temp_dir) if not f.startswith('.')]
            self.assertEqual(len(cache_files), 0)
            
        # Run test case
        asyncio.run(test_case())

    def test_persistence(self):
        """Test that cached data persists after recreating the cache instance."""
        # Define test case function
        async def test_case():
            # Put item in cache
            await self.cache.put(self.test_key.to_string(), self.test_data)
            
            # Create a new cache instance with the same directory
            new_cache = DiskCache(
                directory=self.temp_dir,
                max_size=1_000_000_000,
                default_ttl_seconds=300
            )
            
            # Get item from new cache
            result = await new_cache.get(self.test_key.to_string())
            
            # Verify result
            self.assertIsNotNone(result)
            pd.testing.assert_frame_equal(result, self.test_data)
            
        # Run test case
        asyncio.run(test_case())


if __name__ == "__main__":
    unittest.main()
