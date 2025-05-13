"""
Tests for the CacheManager class in the feature store caching system.
"""
import asyncio
import unittest
import os
import shutil
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from common_lib.caching import AdaptiveCacheManager, cached, get_cache_manager
from core.cache_key import CacheKey


class TestCacheManager(unittest.TestCase):
    """Test cases for CacheManager class."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for disk cache
        self.temp_dir = tempfile.mkdtemp()
        
        # Configure cache
        self.config = {
            'memory_cache_size': 1000000,  # 1MB
            'memory_cache_ttl': 300,  # 5 minutes
            'use_disk_cache': True,
            'disk_cache_path': self.temp_dir,
            'disk_cache_size': 10000000,  # 10MB
            'disk_cache_ttl': 3600  # 1 hour
        }
        
        # Initialize cache manager
        self.cache_manager = CacheManager(self.config)
        
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

    def tearDown(self):
        """Clean up after each test."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

    def test_put_and_get(self):
        """Test putting and getting items from the cache manager."""
        # Define test case function
        async def test_case():
            # Put item in cache
            success = await self.cache_manager.put(self.test_key, self.test_data)
            self.assertTrue(success)
            
            # Get item from cache
            result = await self.cache_manager.get(self.test_key)
            
            # Verify result
            self.assertIsNotNone(result)
            pd.testing.assert_frame_equal(result, self.test_data)
            
        # Run test case
        asyncio.run(test_case())

    def test_tier_promotion(self):
        """Test that items are promoted between cache tiers."""
        # Define test case function
        async def test_case():
            # Put item in cache
            await self.cache_manager.put(self.test_key, self.test_data)
            
            # Clear memory cache to simulate the item being evicted from memory
            # but still present in disk cache
            await self.cache_manager.memory_cache.clear()
            
            # Verify memory cache is empty
            self.assertEqual(self.cache_manager.memory_cache.item_count, 0)
            
            # Get item from cache - should be pulled from disk and promoted to memory
            result = await self.cache_manager.get(self.test_key)
            
            # Verify result is correct
            self.assertIsNotNone(result)
            pd.testing.assert_frame_equal(result, self.test_data)
            
            # Verify item was promoted to memory cache
            self.assertEqual(self.cache_manager.memory_cache.item_count, 1)
            
        # Run test case
        asyncio.run(test_case())

    def test_invalidate_by_pattern(self):
        """Test invalidating cache entries based on pattern."""
        # Define test case function
        async def test_case():
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
                symbol="GBPUSD",  # Different symbol
                timeframe="1h",
                start_time=datetime(2025, 1, 1, 0, 0, 0),
                end_time=datetime(2025, 1, 1, 23, 59, 59)
            )
            
            key3 = CacheKey(
                indicator_type="RSI",  # Different indicator
                params={"window": 14},
                symbol="EURUSD",
                timeframe="1h",
                start_time=datetime(2025, 1, 1, 0, 0, 0),
                end_time=datetime(2025, 1, 1, 23, 59, 59)
            )
            
            # Put items in cache
            await self.cache_manager.put(key1, "data1")
            await self.cache_manager.put(key2, "data2")
            await self.cache_manager.put(key3, "data3")
            
            # Verify all items exist
            self.assertIsNotNone(await self.cache_manager.get(key1))
            self.assertIsNotNone(await self.cache_manager.get(key2))
            self.assertIsNotNone(await self.cache_manager.get(key3))
            
            # Invalidate by symbol
            count = await self.cache_manager.invalidate_by_symbol("EURUSD")
            
            # Verify correct items were invalidated
            self.assertEqual(count, 2)  # Should invalidate key1 and key3
            self.assertIsNone(await self.cache_manager.get(key1))
            self.assertIsNotNone(await self.cache_manager.get(key2))
            self.assertIsNone(await self.cache_manager.get(key3))
            
        # Run test case
        asyncio.run(test_case())

    def test_invalidate_by_indicator(self):
        """Test invalidating cache entries based on indicator type."""
        # Define test case function
        async def test_case():
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
                symbol="GBPUSD",
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
            await self.cache_manager.put(key1, "data1")
            await self.cache_manager.put(key2, "data2")
            await self.cache_manager.put(key3, "data3")
            
            # Verify all items exist
            self.assertIsNotNone(await self.cache_manager.get(key1))
            self.assertIsNotNone(await self.cache_manager.get(key2))
            self.assertIsNotNone(await self.cache_manager.get(key3))
            
            # Invalidate by indicator
            count = await self.cache_manager.invalidate_by_indicator("SMA")
            
            # Verify correct items were invalidated
            self.assertEqual(count, 2)  # Should invalidate key1 and key2
            self.assertIsNone(await self.cache_manager.get(key1))
            self.assertIsNone(await self.cache_manager.get(key2))
            self.assertIsNotNone(await self.cache_manager.get(key3))
            
        # Run test case
        asyncio.run(test_case())

    def test_get_metrics(self):
        """Test getting cache metrics."""
        # Define test case function
        async def test_case():
            # Put items in cache
            await self.cache_manager.put(self.test_key, self.test_data)
            
            # Get some items to record hits
            await self.cache_manager.get(self.test_key)
            await self.cache_manager.get(self.test_key)
            
            # Get nonexistent item to record miss
            await self.cache_manager.get("nonexistent_key")
            
            # Get metrics
            metrics = self.cache_manager.get_metrics()
            
            # Verify metrics structure
            self.assertIn('memory_cache', metrics)
            self.assertIn('disk_cache', metrics)
            self.assertIn('performance', metrics)
            
        # Run test case
        asyncio.run(test_case())

    def test_clear(self):
        """Test clearing all cache entries."""
        # Define test case function
        async def test_case():
            # Put items in cache
            await self.cache_manager.put(self.test_key, self.test_data)
            
            # Verify item exists
            self.assertIsNotNone(await self.cache_manager.get(self.test_key))
            
            # Clear cache
            count = await self.cache_manager.clear()
            
            # Verify cache is empty
            self.assertIsNone(await self.cache_manager.get(self.test_key))
            
        # Run test case
        asyncio.run(test_case())


if __name__ == "__main__":
    unittest.main()
