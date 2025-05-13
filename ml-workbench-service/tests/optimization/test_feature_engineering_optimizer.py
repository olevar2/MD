"""
Tests for the FeatureEngineeringOptimizer class.
"""

import unittest
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import time
from pathlib import Path

# Import the FeatureEngineeringOptimizer class
try:
    """
    try class.
    
    Attributes:
        Add attributes here
    """

    from ml_workbench_service.optimization.feature_engineering_optimizer import FeatureEngineeringOptimizer
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False

@unittest.skipIf(not OPTIMIZER_AVAILABLE, "FeatureEngineeringOptimizer not available")
class TestFeatureEngineeringOptimizer(unittest.TestCase):
    """Test cases for the FeatureEngineeringOptimizer class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.test_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create sample data
        np.random.seed(42)
        self.sample_data = pd.DataFrame(
            np.random.randn(1000, 20),
            columns=[f"feature_{i}" for i in range(20)]
        )
        
        # Initialize optimizer
        self.optimizer = FeatureEngineeringOptimizer(
            cache_dir=self.cache_dir,
            max_cache_size_mb=100,
            n_jobs=2
        )
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_cached_feature_computation(self):
        """Test cached feature computation."""
        # Define a simple feature function
        def compute_mean(data):
    """
    Compute mean.
    
    Args:
        data: Description of data
    
    """

            return data.mean(axis=0)
        
        # First computation (cache miss)
        start_time = time.time()
        result1, metadata1 = self.optimizer.cached_feature_computation(
            data=self.sample_data,
            feature_func=compute_mean
        )
        first_time = time.time() - start_time
        
        self.assertFalse(metadata1["cache_hit"])
        self.assertIsInstance(result1, pd.Series)
        
        # Second computation (cache hit)
        start_time = time.time()
        result2, metadata2 = self.optimizer.cached_feature_computation(
            data=self.sample_data,
            feature_func=compute_mean
        )
        second_time = time.time() - start_time
        
        self.assertTrue(metadata2["cache_hit"])
        self.assertIsInstance(result2, pd.Series)
        pd.testing.assert_series_equal(result1, result2)
        
        # Verify cache hit is faster
        self.assertLess(second_time, first_time)
        
        # Force recomputation
        start_time = time.time()
        result3, metadata3 = self.optimizer.cached_feature_computation(
            data=self.sample_data,
            feature_func=compute_mean,
            force_recompute=True
        )
        third_time = time.time() - start_time
        
        self.assertFalse(metadata3["cache_hit"])
        self.assertIsInstance(result3, pd.Series)
        pd.testing.assert_series_equal(result1, result3)
        
        # Verify forced recomputation is slower than cache hit
        self.assertGreater(third_time, second_time)
    
    def test_parallel_feature_computation(self):
        """Test parallel feature computation."""
        # Define multiple feature functions
        def compute_mean(data):
    """
    Compute mean.
    
    Args:
        data: Description of data
    
    """

            return data.mean(axis=0)
            
        def compute_std(data):
    """
    Compute std.
    
    Args:
        data: Description of data
    
    """

            return data.std(axis=0)
            
        def compute_min(data):
    """
    Compute min.
    
    Args:
        data: Description of data
    
    """

            return data.min(axis=0)
            
        def compute_max(data):
    """
    Compute max.
    
    Args:
        data: Description of data
    
    """

            return data.max(axis=0)
        
        feature_funcs = [compute_mean, compute_std, compute_min, compute_max]
        
        # Compute features in parallel
        results, metadata = self.optimizer.parallel_feature_computation(
            data=self.sample_data,
            feature_funcs=feature_funcs,
            use_cache=True
        )
        
        # Verify results
        self.assertEqual(len(results), 4)
        self.assertIn("compute_mean", results)
        self.assertIn("compute_std", results)
        self.assertIn("compute_min", results)
        self.assertIn("compute_max", results)
        
        # Verify metadata
        self.assertIn("computation_times", metadata)
        self.assertEqual(len(metadata["computation_times"]), 4)
        self.assertIn("cache_hits", metadata)
        self.assertEqual(len(metadata["cache_hits"]), 4)
        self.assertIn("total_time", metadata)
        self.assertIn("success_rate", metadata)
        self.assertEqual(metadata["success_rate"], 1.0)
        
        # Compute again to test cache hits
        results2, metadata2 = self.optimizer.parallel_feature_computation(
            data=self.sample_data,
            feature_funcs=feature_funcs,
            use_cache=True
        )
        
        # Verify cache hits
        self.assertTrue(all(metadata2["cache_hits"].values()))
    
    def test_incremental_feature_computation(self):
        """Test incremental feature computation."""
        # Define feature functions
        def compute_mean(data):
    """
    Compute mean.
    
    Args:
        data: Description of data
    
    """

            return data.mean(axis=0)
            
        def compute_std(data):
    """
    Compute std.
    
    Args:
        data: Description of data
    
    """

            return data.std(axis=0)
        
        feature_funcs = [compute_mean, compute_std]
        
        # Define incremental functions
        def incremental_mean(prev_data, new_data, prev_result):
    """
    Incremental mean.
    
    Args:
        prev_data: Description of prev_data
        new_data: Description of new_data
        prev_result: Description of prev_result
    
    """

            combined_count = len(prev_data) + len(new_data)
            return (prev_result * len(prev_data) + new_data.mean(axis=0) * len(new_data)) / combined_count
            
        def incremental_std(prev_data, new_data, prev_result):
    """
    Incremental std.
    
    Args:
        prev_data: Description of prev_data
        new_data: Description of new_data
        prev_result: Description of prev_result
    
    """

            # This is a simplified approach - in practice, you'd need a more sophisticated algorithm
            # to incrementally update standard deviation
            combined_data = pd.concat([prev_data, new_data], ignore_index=True)
            return combined_data.std(axis=0)
        
        incremental_funcs = [incremental_mean, incremental_std]
        
        # Compute initial features
        initial_features, _ = self.optimizer.parallel_feature_computation(
            data=self.sample_data,
            feature_funcs=feature_funcs
        )
        
        # Create new data
        new_data = pd.DataFrame(
            np.random.randn(200, 20),
            columns=[f"feature_{i}" for i in range(20)]
        )
        
        # Compute features incrementally
        incremental_results, metadata = self.optimizer.incremental_feature_computation(
            previous_data=self.sample_data,
            previous_features=initial_features,
            new_data=new_data,
            feature_funcs=feature_funcs,
            incremental_funcs=incremental_funcs
        )
        
        # Verify results
        self.assertEqual(len(incremental_results), 2)
        self.assertIn("compute_mean", incremental_results)
        self.assertIn("compute_std", incremental_results)
        
        # Verify metadata
        self.assertIn("computation_times", metadata)
        self.assertEqual(len(metadata["computation_times"]), 2)
        self.assertIn("total_time", metadata)
        self.assertIn("success_rate", metadata)
        self.assertEqual(metadata["success_rate"], 1.0)
        
        # Compute features from scratch for comparison
        combined_data = pd.concat([self.sample_data, new_data], ignore_index=True)
        scratch_results, _ = self.optimizer.parallel_feature_computation(
            data=combined_data,
            feature_funcs=feature_funcs
        )
        
        # Verify incremental results are close to scratch results
        pd.testing.assert_series_equal(
            incremental_results["compute_mean"],
            scratch_results["compute_mean"],
            check_exact=False,
            rtol=1e-10
        )
    
    def test_benchmark_feature_pipeline(self):
        """Test benchmarking feature pipeline."""
        # Define feature functions
        def compute_mean(data):
    """
    Compute mean.
    
    Args:
        data: Description of data
    
    """

            return data.mean(axis=0)
            
        def compute_std(data):
    """
    Compute std.
    
    Args:
        data: Description of data
    
    """

            return data.std(axis=0)
            
        def compute_min(data):
    """
    Compute min.
    
    Args:
        data: Description of data
    
    """

            return data.min(axis=0)
            
        def compute_max(data):
    """
    Compute max.
    
    Args:
        data: Description of data
    
    """

            return data.max(axis=0)
        
        feature_funcs = [compute_mean, compute_std, compute_min, compute_max]
        
        # Benchmark with caching and parallel processing
        benchmark_results = self.optimizer.benchmark_feature_pipeline(
            data=self.sample_data,
            feature_funcs=feature_funcs,
            n_runs=2,
            use_cache=True,
            use_parallel=True
        )
        
        # Verify results
        self.assertIn("overall", benchmark_results)
        self.assertIn("per_feature", benchmark_results)
        self.assertIn("timestamp", benchmark_results)
        self.assertIn("n_runs", benchmark_results)
        self.assertIn("use_cache", benchmark_results)
        self.assertIn("use_parallel", benchmark_results)
        
        # Verify overall metrics
        self.assertIn("avg_time", benchmark_results["overall"])
        self.assertIn("min_time", benchmark_results["overall"])
        self.assertIn("max_time", benchmark_results["overall"])
        self.assertIn("std_time", benchmark_results["overall"])
        
        # Verify per-feature metrics
        for func_name in ["compute_mean", "compute_std", "compute_min", "compute_max"]:
            self.assertIn(func_name, benchmark_results["per_feature"])
            self.assertIn("avg_time", benchmark_results["per_feature"][func_name])
            self.assertIn("min_time", benchmark_results["per_feature"][func_name])
            self.assertIn("max_time", benchmark_results["per_feature"][func_name])
            self.assertIn("std_time", benchmark_results["per_feature"][func_name])
            self.assertIn("pct_of_total", benchmark_results["per_feature"][func_name])
        
        # Benchmark without caching and parallel processing
        benchmark_results2 = self.optimizer.benchmark_feature_pipeline(
            data=self.sample_data,
            feature_funcs=feature_funcs,
            n_runs=2,
            use_cache=False,
            use_parallel=False
        )
        
        # Verify caching and parallel processing improve performance
        self.assertGreater(
            benchmark_results2["overall"]["avg_time"],
            benchmark_results["overall"]["avg_time"]
        )

if __name__ == '__main__':
    unittest.main()
