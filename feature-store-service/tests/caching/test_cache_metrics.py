"""
Tests for the CacheMetrics class in the feature store caching system.
"""
import unittest
import time
from datetime import datetime

from feature_store_service.caching.cache_metrics import CacheMetrics


class TestCacheMetrics(unittest.TestCase):
    """Test cases for CacheMetrics class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.metrics = CacheMetrics()

    def test_hit_miss_tracking(self):
        """Test that hit and miss counts are tracked correctly."""
        # Initial state
        self.assertEqual(self.metrics.hits, 0)
        self.assertEqual(self.metrics.misses, 0)
        self.assertEqual(self.metrics.hit_ratio, 0.0)
        
        # Record some hits
        for _ in range(5):
            self.metrics.record_hit('memory')
            
        # Record some misses
        for _ in range(3):
            self.metrics.record_miss()
            
        # Record more hits
        for _ in range(2):
            self.metrics.record_hit('disk')
            
        # Verify counts
        self.assertEqual(self.metrics.hits, 7)  # 5 memory + 2 disk
        self.assertEqual(self.metrics.misses, 3)
        self.assertEqual(self.metrics.memory_hits, 5)
        self.assertEqual(self.metrics.disk_hits, 2)
        
        # Verify hit ratio
        expected_ratio = 7 / 10  # 7 hits out of 10 total operations
        self.assertAlmostEqual(self.metrics.hit_ratio, expected_ratio)

    def test_timing_tracking(self):
        """Test that operation timing is tracked correctly."""
        # Record some timings for hits
        self.metrics.record_hit_time(0.01)  # 10ms
        self.metrics.record_hit_time(0.02)  # 20ms
        self.metrics.record_hit_time(0.03)  # 30ms
        
        # Record some timings for misses
        self.metrics.record_miss_time(0.05)  # 50ms
        self.metrics.record_miss_time(0.07)  # 70ms
        
        # Verify average times
        self.assertAlmostEqual(self.metrics.avg_hit_time, 0.02)  # (10+20+30)/3 = 20ms
        self.assertAlmostEqual(self.metrics.avg_miss_time, 0.06)  # (50+70)/2 = 60ms

    def test_data_volume_tracking(self):
        """Test that data volume is tracked correctly."""
        # Initial state
        self.assertEqual(self.metrics.total_bytes_stored, 0)
        self.assertEqual(self.metrics.total_bytes_retrieved, 0)
        
        # Record some data storage operations
        self.metrics.record_put(1000)  # 1KB
        self.metrics.record_put(2000)  # 2KB
        
        # Record some data retrieval operations
        self.metrics.record_get(500)   # 500B
        self.metrics.record_get(1500)  # 1.5KB
        
        # Verify volume tracking
        self.assertEqual(self.metrics.total_bytes_stored, 3000)
        self.assertEqual(self.metrics.total_bytes_retrieved, 2000)

    def test_reset(self):
        """Test that metrics reset works correctly."""
        # Record some metrics
        self.metrics.record_hit('memory')
        self.metrics.record_miss()
        self.metrics.record_put(1000)
        
        # Reset
        self.metrics.reset()
        
        # Verify all metrics are reset
        self.assertEqual(self.metrics.hits, 0)
        self.assertEqual(self.metrics.misses, 0)
        self.assertEqual(self.metrics.memory_hits, 0)
        self.assertEqual(self.metrics.disk_hits, 0)
        self.assertEqual(self.metrics.total_bytes_stored, 0)
        self.assertEqual(self.metrics.hit_ratio, 0.0)

    def test_interval_tracking(self):
        """Test that interval-based metrics are tracked correctly."""
        # Record some hits and misses
        self.metrics.record_hit('memory')
        self.metrics.record_hit('memory')
        self.metrics.record_miss()
        
        # Get metrics for this interval
        interval_metrics = self.metrics.get_interval_metrics()
        
        # Verify interval metrics
        self.assertEqual(interval_metrics['hits'], 2)
        self.assertEqual(interval_metrics['misses'], 1)
        self.assertEqual(interval_metrics['memory_hits'], 2)
        self.assertEqual(interval_metrics['disk_hits'], 0)
        self.assertAlmostEqual(interval_metrics['hit_ratio'], 2/3)
        
        # Record more operations
        self.metrics.record_hit('disk')
        self.metrics.record_miss()
        
        # Get new interval metrics
        new_interval_metrics = self.metrics.get_interval_metrics()
        
        # Verify new interval metrics only include operations since last call
        self.assertEqual(new_interval_metrics['hits'], 1)
        self.assertEqual(new_interval_metrics['misses'], 1)
        self.assertEqual(new_interval_metrics['memory_hits'], 0)
        self.assertEqual(new_interval_metrics['disk_hits'], 1)
        self.assertAlmostEqual(new_interval_metrics['hit_ratio'], 0.5)

    def test_invalidation_tracking(self):
        """Test that invalidation operations are tracked."""
        # Initial state
        self.assertEqual(self.metrics.invalidations, 0)
        self.assertEqual(self.metrics.invalidated_entries, 0)
        
        # Record some invalidations
        self.metrics.record_invalidation(5)  # 5 entries invalidated
        self.metrics.record_invalidation(3)  # 3 more entries invalidated
        
        # Verify counts
        self.assertEqual(self.metrics.invalidations, 2)  # 2 invalidation operations
        self.assertEqual(self.metrics.invalidated_entries, 8)  # 8 total entries invalidated

    def test_to_dict(self):
        """Test conversion of metrics to dictionary format."""
        # Record some metrics
        self.metrics.record_hit('memory')
        self.metrics.record_hit('disk')
        self.metrics.record_miss()
        self.metrics.record_hit_time(0.01)
        self.metrics.record_miss_time(0.02)
        self.metrics.record_put(1000)
        self.metrics.record_get(500)
        self.metrics.record_invalidation(2)
        
        # Convert to dict
        metrics_dict = self.metrics.to_dict()
        
        # Verify dict format and content
        self.assertIn('hits', metrics_dict)
        self.assertIn('misses', metrics_dict)
        self.assertIn('hit_ratio', metrics_dict)
        self.assertIn('memory_hits', metrics_dict)
        self.assertIn('disk_hits', metrics_dict)
        self.assertIn('avg_hit_time', metrics_dict)
        self.assertIn('avg_miss_time', metrics_dict)
        self.assertIn('total_bytes_stored', metrics_dict)
        self.assertIn('total_bytes_retrieved', metrics_dict)
        self.assertIn('invalidations', metrics_dict)
        self.assertIn('invalidated_entries', metrics_dict)
        self.assertIn('last_reset', metrics_dict)
        
        # Verify values
        self.assertEqual(metrics_dict['hits'], 2)
        self.assertEqual(metrics_dict['misses'], 1)
        self.assertAlmostEqual(metrics_dict['hit_ratio'], 2/3)
        self.assertEqual(metrics_dict['memory_hits'], 1)
        self.assertEqual(metrics_dict['disk_hits'], 1)


if __name__ == "__main__":
    unittest.main()
