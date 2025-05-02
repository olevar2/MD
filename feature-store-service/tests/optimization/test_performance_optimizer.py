"""
Tests for the performance profiling and optimization service.
"""
import unittest
import tempfile
import shutil
from datetime import datetime, timedelta
import time
import threading
from pathlib import Path
import numpy as np
import pandas as pd

from feature_store_service.optimization.performance_optimizer import (
    PerformanceProfile,
    PerformanceOptimizer,
    PerformanceMonitor
)

def dummy_calculation(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Dummy calculation function for testing."""
    time.sleep(0.1)  # Simulate work
    return pd.Series(np.random.random(len(data)))

class TestPerformanceOptimization(unittest.TestCase):
    """Test suite for performance optimization components."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.optimizer = PerformanceOptimizer(profile_dir=self.temp_dir)
        self.monitor = PerformanceMonitor()
        
        # Create test data
        self.test_data = pd.DataFrame({
            'close': np.random.random(1000),
            'volume': np.random.random(1000)
        })

    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir)

    def test_performance_profiling(self):
        """Test performance profiling of calculations."""
        # Profile a calculation
        result, profile = self.optimizer.profile_calculation(
            indicator_name="TestIndicator",
            calc_func=dummy_calculation,
            data=self.test_data,
            parameters={'period': 14}
        )
        
        self.assertIsInstance(profile, PerformanceProfile)
        self.assertEqual(profile.indicator_name, "TestIndicator")
        self.assertGreater(profile.execution_time, 0)
        self.assertGreater(profile.cpu_time, 0)
        self.assertGreaterEqual(profile.memory_used, 0)
        self.assertEqual(profile.data_points, len(self.test_data))

    def test_performance_analysis(self):
        """Test performance analysis functionality."""
        # Create multiple profiles
        parameters = [
            {'period': 10},
            {'period': 20},
            {'period': 30}
        ]
        
        for params in parameters:
            _, profile = self.optimizer.profile_calculation(
                indicator_name="TestIndicator",
                calc_func=dummy_calculation,
                data=self.test_data,
                parameters=params
            )
        
        # Analyze performance
        analysis = self.optimizer.analyze_performance("TestIndicator")
        
        self.assertIn('execution_time', analysis)
        self.assertIn('memory_usage', analysis)
        self.assertIn('throughput', analysis)
        self.assertIn('parameter_impact', analysis)
        self.assertIn('trend', analysis)
        self.assertIn('bottlenecks', analysis)

    def test_parameter_optimization(self):
        """Test parameter optimization functionality."""
        # Create profiles with different parameters
        test_params = [
            {'period': 5},
            {'period': 10},
            {'period': 15},
            {'period': 20}
        ]
        
        for params in test_params:
            _, _ = self.optimizer.profile_calculation(
                indicator_name="TestIndicator",
                calc_func=dummy_calculation,
                data=self.test_data,
                parameters=params
            )
            time.sleep(0.1)  # Ensure different timestamps
        
        # Get optimized parameters
        optimized = self.optimizer.optimize_parameters(
            "TestIndicator",
            {'period': 30}  # Current parameters
        )
        
        self.assertIn('period', optimized)
        self.assertIsInstance(optimized['period'], (int, float))

    def test_performance_monitoring(self):
        """Test performance monitoring functionality."""
        calc_id = "test_calc_1"
        
        # Start monitoring
        self.monitor.start_calculation(
            calc_id=calc_id,
            indicator_name="TestIndicator",
            parameters={'period': 14}
        )
        
        # Simulate calculation
        time.sleep(0.1)
        
        # End monitoring
        profile = self.monitor.end_calculation(
            calc_id=calc_id,
            data_points=len(self.test_data)
        )
        
        self.assertIsNotNone(profile)
        self.assertEqual(profile.indicator_name, "TestIndicator")
        self.assertGreater(profile.execution_time, 0)

    def test_concurrent_profiling(self):
        """Test concurrent performance profiling."""
        def worker(worker_id):
            _, profile = self.optimizer.profile_calculation(
                indicator_name=f"TestIndicator_{worker_id}",
                calc_func=dummy_calculation,
                data=self.test_data,
                parameters={'period': 14}
            )
            self.assertIsNotNone(profile)
            
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            thread.start()
            threads.append(thread)
            
        for thread in threads:
            thread.join()

    def test_bottleneck_detection(self):
        """Test performance bottleneck detection."""
        # Create profiles with degrading performance
        base_time = 0.1
        for i in range(5):
            time.sleep(0.01)  # Ensure different timestamps
            profile = PerformanceProfile(
                indicator_name="TestIndicator",
                execution_time=base_time * (1 + i * 0.2),  # Increasing time
                cpu_time=base_time * (1 + i * 0.2),
                memory_used=100 * (1 + i * 0.1),  # Increasing memory
                data_points=1000,
                parameters={'period': 14}
            )
            self.optimizer._store_profile(profile)
            
        analysis = self.optimizer.analyze_performance("TestIndicator")
        bottlenecks = analysis['bottlenecks']
        
        self.assertTrue(any(b['type'] == 'degradation' for b in bottlenecks))
        self.assertTrue(any(b['type'] == 'growth' for b in bottlenecks))

    def test_profile_persistence(self):
        """Test profile storage and loading."""
        # Create and store a profile
        _, original_profile = self.optimizer.profile_calculation(
            indicator_name="TestIndicator",
            calc_func=dummy_calculation,
            data=self.test_data,
            parameters={'period': 14}
        )
        
        # Create new optimizer instance (should load existing profiles)
        new_optimizer = PerformanceOptimizer(profile_dir=self.temp_dir)
        
        # Check if profile was loaded
        analysis = new_optimizer.analyze_performance("TestIndicator")
        self.assertGreater(analysis['sample_size'], 0)

    def test_parameter_impact_analysis(self):
        """Test parameter impact analysis."""
        # Create profiles with different parameters
        parameters = [
            {'period': p, 'alpha': a}
            for p in [10, 20, 30]
            for a in [0.1, 0.2, 0.3]
        ]
        
        for params in parameters:
            _, profile = self.optimizer.profile_calculation(
                indicator_name="TestIndicator",
                calc_func=dummy_calculation,
                data=self.test_data,
                parameters=params
            )
            time.sleep(0.01)  # Ensure different timestamps
            
        analysis = self.optimizer.analyze_performance("TestIndicator")
        param_impact = analysis['parameter_impact']
        
        self.assertIn('period', param_impact)
        self.assertIn('alpha', param_impact)
        self.assertIn('correlation', param_impact['period'])
        self.assertIn('optimal_range', param_impact['period'])

    def test_performance_trend_analysis(self):
        """Test performance trend analysis."""
        # Create profiles with improving performance
        base_time = 0.2
        for i in range(5):
            time.sleep(0.01)  # Ensure different timestamps
            profile = PerformanceProfile(
                indicator_name="TestIndicator",
                execution_time=base_time / (1 + i * 0.1),  # Decreasing time
                cpu_time=base_time / (1 + i * 0.1),
                memory_used=100,
                data_points=1000,
                parameters={'period': 14}
            )
            self.optimizer._store_profile(profile)
            
        analysis = self.optimizer.analyze_performance("TestIndicator")
        trend = analysis['trend']
        
        self.assertEqual(trend['direction'], 'improving')
        self.assertGreater(trend['magnitude'], 0)

if __name__ == '__main__':
    unittest.main()
