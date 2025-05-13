"""
Tests for the resource management and optimization service.
"""
import unittest
import tempfile
import shutil
from datetime import datetime, timedelta
import time
import threading
from pathlib import Path
import json
import numpy as np

from feature_store_service.optimization.resource_manager import (
    ResourceMetrics,
    LoadBalancer,
    CacheManager,
    AdaptiveResourceManager
)

class TestResourceManagement(unittest.TestCase):
    """Test suite for resource management components."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.resource_manager = AdaptiveResourceManager(
            cache_dir=self.temp_dir,
            max_memory_size=256,  # Small size for testing
            max_cache_age=timedelta(minutes=5)
        )

    def tearDown(self):
        """Clean up after each test method."""
        self.resource_manager.cleanup()
        shutil.rmtree(self.temp_dir)

    def test_resource_metrics(self):
        """Test resource metrics collection."""
        metrics = ResourceMetrics()
        
        # Record some metrics
        for _ in range(3):
            metrics.record_metrics()
            time.sleep(0.1)  # Small delay between recordings
            
        summary = metrics.get_summary()
        
        self.assertIn('cpu', summary)
        self.assertIn('memory', summary)
        self.assertIn('threads', summary)
        self.assertIn('processes', summary)
        
        # Verify metric structure
        for category in ['cpu', 'memory', 'threads', 'processes']:
            self.assertIn('current', summary[category])
            self.assertIn('mean', summary[category])
            self.assertIn('max', summary[category])

    def test_load_balancer_task_submission(self):
        """Test task submission and execution."""
        load_balancer = LoadBalancer()
        
        def task_func(x):
    """
    Task func.
    
    Args:
        x: Description of x
    
    """

            return x * 2
            
        # Submit CPU-bound task
        future = load_balancer.submit_task(
            task_id="test_task",
            task_type="cpu_bound",
            func=task_func,
            args=(5,)
        )
        
        result = future.result()
        self.assertEqual(result, 10)
        
        # Check task tracking
        active_tasks = load_balancer.get_active_tasks()
        self.assertEqual(len(active_tasks), 0)  # Task should be completed

    def test_cache_manager(self):
        """Test cache operations."""
        cache_manager = CacheManager(
            cache_dir=self.temp_dir,
            max_memory_size=100,
            max_cache_age=timedelta(minutes=5)
        )
        
        # Test cache put/get
        test_data = {"value": 42}
        cache_manager.put("test_key", test_data)
        
        cached_value = cache_manager.get("test_key")
        self.assertEqual(cached_value, test_data)
        
        # Test cache stats
        stats = cache_manager.get_stats()
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 0)
        
        # Test cache expiration
        with unittest.mock.patch('datetime.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = datetime.utcnow() + timedelta(hours=1)
            cached_value = cache_manager.get("test_key")
            self.assertIsNone(cached_value)

    def test_adaptive_resource_manager(self):
        """Test integrated resource management."""
        def test_calculation(x):
            time.sleep(0.1)  # Simulate work
            return x * 2
            
        # Submit multiple calculations
        futures = []
        for i in range(3):
            future = self.resource_manager.submit_calculation(
                calc_id=f"calc_{i}",
                calc_func=test_calculation,
                args=(i,),
                cache_key=f"result_{i}"
            )
            futures.append(future)
            
        # Wait for results
        results = [future.result() for future in futures]
        self.assertEqual(results, [0, 2, 4])
        
        # Check system status
        status = self.resource_manager.get_system_status()
        self.assertIn('resources', status)
        self.assertIn('active_tasks', status)
        self.assertIn('cache_stats', status)

    def test_memory_management(self):
        """Test memory usage management."""
        cache_manager = CacheManager(
            cache_dir=self.temp_dir,
            max_memory_size=10,  # Very small limit for testing
            max_cache_age=timedelta(minutes=5)
        )
        
        # Add items until memory limit is exceeded
        large_data = "x" * 1000000  # 1MB string
        for i in range(20):
            cache_manager.put(f"key_{i}", large_data)
            
        stats = cache_manager.get_stats()
        self.assertGreater(stats['evictions'], 0)

    def test_concurrent_cache_access(self):
        """Test concurrent cache access."""
        cache_manager = CacheManager(cache_dir=self.temp_dir)
        
        def worker(worker_id):
    """
    Worker.
    
    Args:
        worker_id: Description of worker_id
    
    """

            for i in range(10):
                key = f"worker_{worker_id}_key_{i}"
                cache_manager.put(key, i)
                time.sleep(0.01)
                value = cache_manager.get(key)
                self.assertEqual(value, i)
                
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            thread.start()
            threads.append(thread)
            
        for thread in threads:
            thread.join()
            
        stats = cache_manager.get_stats()
        self.assertGreater(stats['hits'], 0)

    def test_calculation_caching(self):
        """Test calculation result caching."""
        def expensive_calculation(x):
    """
    Expensive calculation.
    
    Args:
        x: Description of x
    
    """

            time.sleep(0.2)  # Simulate expensive computation
            return x * 2
            
        # First calculation - should be computed
        start_time = time.time()
        result1 = self.resource_manager.submit_calculation(
            calc_id="calc_1",
            calc_func=expensive_calculation,
            args=(5,),
            cache_key="test_calc"
        ).result()
        
        time1 = time.time() - start_time
        
        # Second calculation - should be cached
        start_time = time.time()
        result2 = self.resource_manager.submit_calculation(
            calc_id="calc_2",
            calc_func=expensive_calculation,
            args=(5,),
            cache_key="test_calc"
        ).result()
        
        time2 = time.time() - start_time
        
        self.assertEqual(result1, result2)
        self.assertLess(time2, time1)  # Cached result should be faster

    def test_resource_monitoring(self):
        """Test resource monitoring functionality."""
        # Let the monitoring run for a bit
        time.sleep(2)
        
        status = self.resource_manager.get_system_status()
        resources = status['resources']
        
        # Verify resource metrics
        self.assertGreaterEqual(resources['cpu']['current'], 0)
        self.assertGreaterEqual(resources['memory']['current'], 0)
        self.assertGreater(resources['threads']['current'], 0)
        self.assertGreaterEqual(resources['processes']['current'], 0)

    def test_load_balancer_task_types(self):
        """Test different task type handling."""
        load_balancer = LoadBalancer()
        
        def io_task():
    """
    Io task.
    
    """

            time.sleep(0.1)
            return "io_result"
            
        def cpu_task():
    """
    Cpu task.
    
    """

            result = 0
            for i in range(1000000):
                result += i
            return result
            
        # Submit both types of tasks
        io_future = load_balancer.submit_task(
            task_id="io_task",
            task_type="io_bound",
            func=io_task
        )
        
        cpu_future = load_balancer.submit_task(
            task_id="cpu_task",
            task_type="cpu_bound",
            func=cpu_task
        )
        
        self.assertEqual(io_future.result(), "io_result")
        self.assertGreater(cpu_future.result(), 0)

if __name__ == '__main__':
    unittest.main()
