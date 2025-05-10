"""
Unit tests for the adaptive cache manager.

This module contains tests for the AdaptiveCacheManager class.
"""

import unittest
import time
import threading
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from analysis_engine.utils.adaptive_cache_manager import AdaptiveCacheManager
except ImportError as e:
    print(f"Error importing modules: {e}")
    try:
        # Try with the full path
        sys.path.insert(0, "D:\\MD\\forex_trading_platform")
        from analysis_engine.utils.adaptive_cache_manager import AdaptiveCacheManager
    except ImportError as e:
        print(f"Error importing modules with full path: {e}")
        sys.exit(1)


class TestAdaptiveCacheManager(unittest.TestCase):
    """Test the adaptive cache manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache_manager = AdaptiveCacheManager(
            default_ttl_seconds=1,  # Short TTL for testing
            max_size=5,  # Small size for testing eviction
            cleanup_interval_seconds=0.5  # Short interval for testing cleanup
        )
    
    def test_get_set_basic(self):
        """Test basic get and set operations."""
        # Set a value
        self.cache_manager.set("key1", "value1")
        
        # Get the value
        hit, value = self.cache_manager.get("key1")
        
        # Verify
        self.assertTrue(hit)
        self.assertEqual(value, "value1")
        
        # Get a non-existent key
        hit, value = self.cache_manager.get("non_existent")
        
        # Verify
        self.assertFalse(hit)
        self.assertIsNone(value)
    
    def test_ttl_expiration(self):
        """Test that entries expire after TTL."""
        # Set a value
        self.cache_manager.set("key1", "value1")
        
        # Verify it exists
        hit, value = self.cache_manager.get("key1")
        self.assertTrue(hit)
        
        # Wait for TTL to expire
        time.sleep(1.1)
        
        # Verify it's expired
        hit, value = self.cache_manager.get("key1")
        self.assertFalse(hit)
    
    def test_max_size_eviction(self):
        """Test that entries are evicted when max size is reached."""
        # Fill the cache
        for i in range(5):
            self.cache_manager.set(f"key{i}", f"value{i}")
        
        # Verify all entries exist
        for i in range(5):
            hit, value = self.cache_manager.get(f"key{i}")
            self.assertTrue(hit)
        
        # Add one more entry to trigger eviction
        self.cache_manager.set("key5", "value5")
        
        # Verify at least one entry was evicted
        evicted = False
        for i in range(5):
            hit, value = self.cache_manager.get(f"key{i}")
            if not hit:
                evicted = True
                break
        
        self.assertTrue(evicted)
        
        # Verify the new entry exists
        hit, value = self.cache_manager.get("key5")
        self.assertTrue(hit)
    
    def test_automatic_cleanup(self):
        """Test automatic cleanup of expired entries."""
        # Set some values
        for i in range(5):
            self.cache_manager.set(f"key{i}", f"value{i}")
        
        # Verify all entries exist
        for i in range(5):
            hit, value = self.cache_manager.get(f"key{i}")
            self.assertTrue(hit)
        
        # Wait for TTL to expire and cleanup to run
        time.sleep(1.6)  # TTL + cleanup interval
        
        # Verify all entries are expired
        for i in range(5):
            hit, value = self.cache_manager.get(f"key{i}")
            self.assertFalse(hit)
        
        # Verify cache is empty
        self.assertEqual(len(self.cache_manager.cache), 0)
    
    def test_thread_safety(self):
        """Test thread safety of the cache manager."""
        # Number of threads and operations
        num_threads = 10
        num_ops = 100
        
        # Shared counter for successful operations
        successful_ops = 0
        lock = threading.Lock()
        
        # Thread function
        def worker():
            nonlocal successful_ops
            for i in range(num_ops):
                # Set a value
                key = f"key{i % 10}"
                value = f"value{i}"
                self.cache_manager.set(key, value)
                
                # Get a value
                hit, _ = self.cache_manager.get(key)
                
                # Count successful operations
                if hit:
                    with lock:
                        successful_ops += 1
        
        # Create and start threads
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify that most operations were successful
        # Note: Some might fail due to eviction or expiration
        self.assertGreater(successful_ops, num_threads * num_ops * 0.8)
    
    def test_clear(self):
        """Test clearing the cache."""
        # Set some values
        for i in range(5):
            self.cache_manager.set(f"key{i}", f"value{i}")
        
        # Verify all entries exist
        for i in range(5):
            hit, value = self.cache_manager.get(f"key{i}")
            self.assertTrue(hit)
        
        # Clear the cache
        self.cache_manager.clear()
        
        # Verify all entries are gone
        for i in range(5):
            hit, value = self.cache_manager.get(f"key{i}")
            self.assertFalse(hit)
        
        # Verify cache is empty
        self.assertEqual(len(self.cache_manager.cache), 0)
    
    def test_get_stats(self):
        """Test getting cache statistics."""
        # Set some values
        for i in range(3):
            self.cache_manager.set(f"key{i}", f"value{i}")
        
        # Get some values (hits)
        for i in range(3):
            self.cache_manager.get(f"key{i}")
        
        # Get some non-existent values (misses)
        for i in range(3, 6):
            self.cache_manager.get(f"key{i}")
        
        # Get stats
        stats = self.cache_manager.get_stats()
        
        # Verify stats
        self.assertEqual(stats["size"], 3)
        self.assertEqual(stats["max_size"], 5)
        self.assertEqual(stats["hits"], 3)
        self.assertEqual(stats["misses"], 3)
        self.assertEqual(stats["hit_rate"], 0.5)
    
    def test_copy_on_get_set(self):
        """Test that values are copied on get and set."""
        # Set a mutable value
        original = {"key": "value"}
        self.cache_manager.set("key1", original)
        
        # Modify the original
        original["key"] = "modified"
        
        # Get the cached value
        hit, cached = self.cache_manager.get("key1")
        
        # Verify the cached value was not modified
        self.assertTrue(hit)
        self.assertEqual(cached["key"], "value")
        
        # Modify the cached value
        cached["key"] = "modified_cached"
        
        # Get the value again
        hit, cached2 = self.cache_manager.get("key1")
        
        # Verify the cached value was not modified
        self.assertTrue(hit)
        self.assertEqual(cached2["key"], "value")


if __name__ == "__main__":
    unittest.main()
