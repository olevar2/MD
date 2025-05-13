"""
Tests for the CacheKey class in the feature store caching system.
"""
import unittest
from datetime import datetime, timezone
import json

from core.cache_key import CacheKey


class TestCacheKey(unittest.TestCase):
    """Test cases for the CacheKey class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.indicator_type = "SMA"
        self.params = {"window": 20}
        self.symbol = "EURUSD"
        self.timeframe = "1h"
        self.start_time = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        self.end_time = datetime(2025, 1, 1, 23, 59, 59, tzinfo=timezone.utc)
        
        self.cache_key = CacheKey(
            indicator_type=self.indicator_type,
            params=self.params,
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_time=self.start_time,
            end_time=self.end_time
        )

    def test_initialization(self):
        """Test that CacheKey initializes correctly."""
        self.assertEqual(self.cache_key.indicator_type, self.indicator_type)
        self.assertEqual(self.cache_key.params, self.params)
        self.assertEqual(self.cache_key.symbol, self.symbol)
        self.assertEqual(self.cache_key.timeframe, self.timeframe)
        self.assertEqual(self.cache_key.start_time, self.start_time)
        self.assertEqual(self.cache_key.end_time, self.end_time)

    def test_to_string(self):
        """Test converting CacheKey to string representation."""
        key_string = self.cache_key.to_string()
        
        # Verify key string format
        self.assertIn(self.indicator_type, key_string)
        self.assertIn(self.symbol, key_string)
        self.assertIn(self.timeframe, key_string)
        self.assertIn(json.dumps(self.params, sort_keys=True), key_string)
        self.assertIn(self.start_time.isoformat(), key_string)
        self.assertIn(self.end_time.isoformat(), key_string)
        
        # Verify exact string format
        expected = f"{self.indicator_type}:{json.dumps(self.params, sort_keys=True)}:{self.symbol}:{self.timeframe}:{self.start_time.isoformat()}:{self.end_time.isoformat()}"
        self.assertEqual(key_string, expected)

    def test_from_string(self):
        """Test creating CacheKey from string representation."""
        original_key = self.cache_key
        key_string = original_key.to_string()
        
        # Create a new key from the string
        parsed_key = CacheKey.from_string(key_string)
        
        # Verify all properties match
        self.assertEqual(parsed_key.indicator_type, original_key.indicator_type)
        self.assertEqual(parsed_key.params, original_key.params)
        self.assertEqual(parsed_key.symbol, original_key.symbol)
        self.assertEqual(parsed_key.timeframe, original_key.timeframe)
        self.assertEqual(parsed_key.start_time, original_key.start_time)
        self.assertEqual(parsed_key.end_time, original_key.end_time)

    def test_equality(self):
        """Test equality comparison between CacheKey instances."""
        # Create a key with identical values
        identical_key = CacheKey(
            indicator_type=self.indicator_type,
            params=self.params,
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_time=self.start_time,
            end_time=self.end_time
        )
        
        # Create a key with different values
        different_key = CacheKey(
            indicator_type=self.indicator_type,
            params={"window": 30},  # Different parameter
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_time=self.start_time,
            end_time=self.end_time
        )
        
        # Test equality
        self.assertEqual(self.cache_key, identical_key)
        self.assertNotEqual(self.cache_key, different_key)
        
    def test_hash(self):
        """Test that CacheKey objects are hashable and can be used in dictionaries."""
        # Create a key with identical values
        identical_key = CacheKey(
            indicator_type=self.indicator_type,
            params=self.params,
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_time=self.start_time,
            end_time=self.end_time
        )
        
        # Create a dictionary with CacheKey as key
        cache_dict = {self.cache_key: "test_value"}
        
        # Verify we can retrieve the value using an identical key
        self.assertEqual(cache_dict.get(identical_key), "test_value")
        
    def test_parameters_are_sorted(self):
        """Test that parameter order doesn't affect equality or string representation."""
        # Create keys with the same parameters but in different order
        params1 = {"window": 20, "alpha": 0.5}
        params2 = {"alpha": 0.5, "window": 20}
        
        key1 = CacheKey(
            indicator_type=self.indicator_type,
            params=params1,
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_time=self.start_time,
            end_time=self.end_time
        )
        
        key2 = CacheKey(
            indicator_type=self.indicator_type,
            params=params2,
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_time=self.start_time,
            end_time=self.end_time
        )
        
        # Test equality and string representation
        self.assertEqual(key1, key2)
        self.assertEqual(key1.to_string(), key2.to_string())


if __name__ == "__main__":
    unittest.main()
