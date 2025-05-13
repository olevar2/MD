"""
Tests for the feature cache.
"""
import unittest
import time
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from repositories.feature_cache import FeatureCache, CacheEntry


class TestFeatureCache(unittest.TestCase):
    """Test suite for the feature cache."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache = FeatureCache(
            cleanup_interval=1,
            max_size=10,
            default_ttl=2
        )
        
        # Sample data for tests
        self.sample_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='1h'),
            'value': np.random.rand(5)
        })
    
    def tearDown(self):
        """Clean up after tests."""
        self.cache.clear()
    
    def test_init(self):
        """Test cache initialization."""
        cache = FeatureCache(
            cleanup_interval=60,
            max_size=100,
            default_ttl=300
        )
        
        self.assertEqual(cache.cleanup_interval, 60)
        self.assertEqual(cache.max_size, 100)
        self.assertEqual(cache.default_ttl, 300)
        self.assertEqual(cache.stats["size"], 0)
    
    def test_set_get(self):
        """Test setting and getting cache entries."""
        # Set a value
        self.cache.set("test_key", "test_value")
        
        # Get the value
        value = self.cache.get("test_key")
        
        # Check the value
        self.assertEqual(value, "test_value")
        
        # Check stats
        self.assertEqual(self.cache.stats["size"], 1)
        self.assertEqual(self.cache.stats["hits"], 1)
        self.assertEqual(self.cache.stats["misses"], 0)
    
    def test_get_missing(self):
        """Test getting a missing cache entry."""
        # Get a non-existent value
        value = self.cache.get("missing_key")
        
        # Check the value
        self.assertIsNone(value)
        
        # Check stats
        self.assertEqual(self.cache.stats["size"], 0)
        self.assertEqual(self.cache.stats["hits"], 0)
        self.assertEqual(self.cache.stats["misses"], 1)
    
    def test_expiration(self):
        """Test cache entry expiration."""
        # Set a value with a short TTL
        self.cache.set("test_key", "test_value", ttl=1)
        
        # Get the value immediately
        value1 = self.cache.get("test_key")
        self.assertEqual(value1, "test_value")
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Get the value after expiration
        value2 = self.cache.get("test_key")
        self.assertIsNone(value2)
        
        # Check stats
        self.assertEqual(self.cache.stats["hits"], 1)
        self.assertEqual(self.cache.stats["misses"], 1)
    
    def test_invalidate(self):
        """Test invalidating a cache entry."""
        # Set a value
        self.cache.set("test_key", "test_value")
        
        # Invalidate the value
        self.cache.invalidate("test_key")
        
        # Get the value
        value = self.cache.get("test_key")
        
        # Check the value
        self.assertIsNone(value)
        
        # Check stats
        self.assertEqual(self.cache.stats["size"], 0)
    
    def test_invalidate_pattern(self):
        """Test invalidating cache entries by pattern."""
        # Set multiple values
        self.cache.set("prefix_key1", "value1")
        self.cache.set("prefix_key2", "value2")
        self.cache.set("other_key", "value3")
        
        # Invalidate by pattern
        self.cache.invalidate_pattern("prefix_")
        
        # Get the values
        value1 = self.cache.get("prefix_key1")
        value2 = self.cache.get("prefix_key2")
        value3 = self.cache.get("other_key")
        
        # Check the values
        self.assertIsNone(value1)
        self.assertIsNone(value2)
        self.assertEqual(value3, "value3")
        
        # Check stats
        self.assertEqual(self.cache.stats["size"], 1)
    
    def test_clear(self):
        """Test clearing the cache."""
        # Set multiple values
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        
        # Clear the cache
        self.cache.clear()
        
        # Get the values
        value1 = self.cache.get("key1")
        value2 = self.cache.get("key2")
        
        # Check the values
        self.assertIsNone(value1)
        self.assertIsNone(value2)
        
        # Check stats
        self.assertEqual(self.cache.stats["size"], 0)
    
    def test_eviction(self):
        """Test cache eviction when full."""
        # Fill the cache to capacity
        for i in range(self.cache.max_size):
            self.cache.set(f"key{i}", f"value{i}")
        
        # Add one more entry to trigger eviction
        self.cache.set("new_key", "new_value")
        
        # Check that the cache size is still at max
        self.assertEqual(self.cache.stats["size"], self.cache.max_size)
        
        # Check that the new entry is in the cache
        self.assertEqual(self.cache.get("new_key"), "new_value")
        
        # Check that at least one eviction occurred
        self.assertGreater(self.cache.stats["evictions"], 0)
    
    def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        # Set entries with short TTL
        for i in range(5):
            self.cache.set(f"key{i}", f"value{i}", ttl=1)
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Trigger cleanup
        self.cache._cleanup_expired()
        
        # Check that all entries were removed
        self.assertEqual(self.cache.stats["size"], 0)
    
    def test_get_stats(self):
        """Test getting cache statistics."""
        # Set some values
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        
        # Get some values
        self.cache.get("key1")
        self.cache.get("missing_key")
        
        # Get stats
        stats = self.cache.get_stats()
        
        # Check stats
        self.assertEqual(stats["size"], 2)
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["hit_rate"], 0.5)
        self.assertEqual(stats["max_size"], 10)
        self.assertEqual(stats["evictions"], 0)
    
    def test_pandas_dataframe(self):
        """Test caching pandas DataFrames."""
        # Set a DataFrame
        self.cache.set("df_key", self.sample_df)
        
        # Get the DataFrame
        df = self.cache.get("df_key")
        
        # Check the DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        pd.testing.assert_frame_equal(df, self.sample_df)
    
    def test_cache_entry(self):
        """Test CacheEntry class."""
        # Create an entry
        entry = CacheEntry("test_value", ttl=1)
        
        # Check the value
        self.assertEqual(entry.value, "test_value")
        
        # Check that it's not expired initially
        self.assertFalse(entry.is_expired())
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Check that it's expired now
        self.assertTrue(entry.is_expired())


if __name__ == '__main__':
    unittest.main()
