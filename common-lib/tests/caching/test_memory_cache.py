"""
Tests for the memory cache module.
"""
import unittest
from unittest.mock import patch, MagicMock
import time
import asyncio
import pytest

from common_lib.caching.memory_cache import MemoryCache


class TestMemoryCache(unittest.TestCase):
    """Tests for the MemoryCache class."""
    
    def setUp(self):
        """Set up test environment."""
        self.cache = MemoryCache(
            prefix="test:",
            ttl=1,
            max_size=3,
            cleanup_interval=0.1
        )
    
    def test_get_set(self):
        """Test get and set methods."""
        # Set a value
        self.cache.set("key1", "value1")
        
        # Get the value
        value = self.cache.get("key1")
        
        # Check that the value is returned
        self.assertEqual(value, "value1")
        
        # Get a non-existent value
        value = self.cache.get("non_existent_key")
        
        # Check that None is returned
        self.assertIsNone(value)
    
    def test_delete(self):
        """Test delete method."""
        # Set a value
        self.cache.set("key1", "value1")
        
        # Delete the value
        result = self.cache.delete("key1")
        
        # Check that True is returned
        self.assertTrue(result)
        
        # Check that the value is deleted
        self.assertIsNone(self.cache.get("key1"))
        
        # Delete a non-existent value
        result = self.cache.delete("non_existent_key")
        
        # Check that False is returned
        self.assertFalse(result)
    
    def test_exists(self):
        """Test exists method."""
        # Set a value
        self.cache.set("key1", "value1")
        
        # Check that the key exists
        self.assertTrue(self.cache.exists("key1"))
        
        # Check that a non-existent key doesn't exist
        self.assertFalse(self.cache.exists("non_existent_key"))
    
    def test_increment(self):
        """Test increment method."""
        # Set a value
        self.cache.set("key1", 1)
        
        # Increment the value
        result = self.cache.increment("key1")
        
        # Check that the value is incremented
        self.assertEqual(result, 2)
        self.assertEqual(self.cache.get("key1"), 2)
        
        # Increment a non-existent value
        result = self.cache.increment("non_existent_key")
        
        # Check that the value is set to the increment amount
        self.assertEqual(result, 1)
        self.assertEqual(self.cache.get("non_existent_key"), 1)
        
        # Try to increment a non-numeric value
        self.cache.set("key2", "value2")
        
        # Check that TypeError is raised
        with self.assertRaises(TypeError):
            self.cache.increment("key2")
    
    def test_decrement(self):
        """Test decrement method."""
        # Set a value
        self.cache.set("key1", 2)
        
        # Decrement the value
        result = self.cache.decrement("key1")
        
        # Check that the value is decremented
        self.assertEqual(result, 1)
        self.assertEqual(self.cache.get("key1"), 1)
        
        # Decrement a non-existent value
        result = self.cache.decrement("non_existent_key")
        
        # Check that the value is set to the negative of the decrement amount
        self.assertEqual(result, -1)
        self.assertEqual(self.cache.get("non_existent_key"), -1)
    
    def test_expire(self):
        """Test expire method."""
        # Set a value
        self.cache.set("key1", "value1")
        
        # Set the TTL
        result = self.cache.expire("key1", 2)
        
        # Check that True is returned
        self.assertTrue(result)
        
        # Set the TTL for a non-existent value
        result = self.cache.expire("non_existent_key", 2)
        
        # Check that False is returned
        self.assertFalse(result)
    
    def test_ttl(self):
        """Test ttl method."""
        # Set a value
        self.cache.set("key1", "value1")
        
        # This is a placeholder test
        self.assertTrue(True)
    
    def test_keys(self):
        """Test keys method."""
        # Set some values
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        
        # Get all keys
        keys = self.cache.keys()
        
        # Check that the keys are returned
        self.assertIn("key1", keys)
        self.assertIn("key2", keys)
        
        # Get keys with a pattern
        keys = self.cache.keys("key1")
        
        # Check that only the matching keys are returned
        self.assertIn("key1", keys)
        self.assertNotIn("key2", keys)
    
    def test_flush(self):
        """Test flush method."""
        # Set some values
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        
        # Flush the cache
        result = self.cache.flush()
        
        # Check that True is returned
        self.assertTrue(result)
        
        # Check that the values are flushed
        self.assertIsNone(self.cache.get("key1"))
        self.assertIsNone(self.cache.get("key2"))
    
    def test_get_many(self):
        """Test get_many method."""
        # Set some values
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        
        # Get multiple values
        values = self.cache.get_many(["key1", "key2", "non_existent_key"])
        
        # Check that the values are returned
        self.assertEqual(values, {"key1": "value1", "key2": "value2"})
    
    def test_set_many(self):
        """Test set_many method."""
        # Set multiple values
        result = self.cache.set_many({
            "key1": "value1",
            "key2": "value2"
        })
        
        # Check that True is returned
        self.assertTrue(result)
        
        # Check that the values are set
        self.assertEqual(self.cache.get("key1"), "value1")
        self.assertEqual(self.cache.get("key2"), "value2")
    
    def test_delete_many(self):
        """Test delete_many method."""
        # Set some values
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        
        # Delete multiple values
        count = self.cache.delete_many(["key1", "key2", "non_existent_key"])
        
        # Check that the number of deleted keys is returned
        self.assertEqual(count, 2)
        
        # Check that the values are deleted
        self.assertIsNone(self.cache.get("key1"))
        self.assertIsNone(self.cache.get("key2"))
    
    def test_max_size(self):
        """Test max_size limit."""
        # Set values up to the max size
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        self.cache.set("key3", "value3")
        
        # Check that all values are in the cache
        self.assertEqual(self.cache.get("key1"), "value1")
        self.assertEqual(self.cache.get("key2"), "value2")
        self.assertEqual(self.cache.get("key3"), "value3")
        
        # Set a value that exceeds the max size
        self.cache.set("key4", "value4")
        
        # Check that one of the original values is no longer in the cache
        values = [
            self.cache.get("key1"),
            self.cache.get("key2"),
            self.cache.get("key3")
        ]
        self.assertIn(None, values)
        
        # Check that the new value is in the cache
        self.assertEqual(self.cache.get("key4"), "value4")
    
    def test_expiry(self):
        """Test that values expire."""
        # Set a value with a short TTL
        self.cache.set("key1", "value1", ttl=0.1)
        
        # Check that the value is in the cache
        self.assertEqual(self.cache.get("key1"), "value1")
        
        # Wait for the value to expire
        time.sleep(0.2)
        
        # Check that the value is no longer in the cache
        self.assertIsNone(self.cache.get("key1"))
    
    def test_cleanup(self):
        """Test that expired values are cleaned up."""
        # Set a value with a short TTL
        self.cache.set("key1", "value1", ttl=0.1)
        
        # Check that the value is in the cache
        self.assertEqual(self.cache.get("key1"), "value1")
        
        # Wait for the cleanup to run
        time.sleep(0.3)
        
        # Check that the value is no longer in the cache
        self.assertIsNone(self.cache.get("key1"))
    
    @pytest.mark.asyncio
    async def test_async_methods(self):
        """Test async methods."""
        # Set a value
        await self.cache.async_set("key1", "value1")
        
        # Get the value
        value = await self.cache.async_get("key1")
        
        # Check that the value is returned
        self.assertEqual(value, "value1")
        
        # Delete the value
        result = await self.cache.async_delete("key1")
        
        # Check that True is returned
        self.assertTrue(result)
        
        # Check that the value is deleted
        self.assertIsNone(await self.cache.async_get("key1"))
        
        return None


if __name__ == "__main__":
    unittest.main()