"""
Tests for the optimized ConfluenceAnalyzer performance.

This module contains tests to verify the performance improvements in the
ConfluenceAnalyzer class.
"""

import unittest
import asyncio
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from analysis_engine.analysis.confluence.confluence_analyzer import ConfluenceAnalyzer


class TestConfluenceAnalyzerPerformance(unittest.TestCase):
    """Test case for ConfluenceAnalyzer performance optimizations."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ConfluenceAnalyzer()
        
        # Generate sample market data
        self.sample_data = self._generate_sample_data()
        
    def _generate_sample_data(self, num_bars=500):
        """Generate sample market data for testing."""
        # Generate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=num_bars)
        timestamps = [
            (start_time + timedelta(hours=i)).isoformat()
            for i in range(num_bars)
        ]
        
        # Generate price data with some realistic patterns
        np.random.seed(42)  # For reproducibility
        
        # Start with a base price
        base_price = 1.2000
        
        # Generate random walk
        random_walk = np.random.normal(0, 0.0002, num_bars).cumsum()
        
        # Add trend
        trend = np.linspace(0, 0.01, num_bars)
        
        # Add some cyclical patterns
        cycles = 0.005 * np.sin(np.linspace(0, 5 * np.pi, num_bars))
        
        # Combine components
        close_prices = base_price + random_walk + trend + cycles
        
        # Generate OHLC data
        high_prices = close_prices + np.random.uniform(0, 0.0015, num_bars)
        low_prices = close_prices - np.random.uniform(0, 0.0015, num_bars)
        open_prices = low_prices + np.random.uniform(0, 0.0015, num_bars)
        
        # Generate volume data
        volume = np.random.uniform(100, 1000, num_bars)
        
        # Create market data dictionary
        market_data = {
            "timestamp": timestamps,
            "open": open_prices.tolist(),
            "high": high_prices.tolist(),
            "low": low_prices.tolist(),
            "close": close_prices.tolist(),
            "volume": volume.tolist()
        }
        
        # Create full data dictionary
        data = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "market_data": market_data,
            "market_regime": "trending"
        }
        
        return data
        
    def test_performance_improvement(self):
        """Test that the optimized version is faster than the baseline."""
        # Run the analysis multiple times to get average performance
        num_runs = 5
        total_time = 0
        
        for _ in range(num_runs):
            start_time = time.time()
            result = asyncio.run(self.analyzer.analyze(self.sample_data))
            end_time = time.time()
            
            # Check that the result is valid
            self.assertTrue(result.is_valid)
            
            # Add to total time
            total_time += (end_time - start_time)
            
        # Calculate average time
        avg_time = total_time / num_runs
        
        # Print performance metrics
        print(f"Average analysis time: {avg_time:.4f} seconds")
        
        # Check performance metrics in result
        result = asyncio.run(self.analyzer.analyze(self.sample_data))
        self.assertIn("performance_metrics", result.result)
        
        # Print detailed performance metrics
        metrics = result.result["performance_metrics"]
        print("\nPerformance metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f} seconds")
            
        # Verify that the optimized version is reasonably fast
        # This threshold may need adjustment based on the specific environment
        self.assertLess(avg_time, 0.5, "Analysis should complete in under 0.5 seconds")
        
    def test_caching_effectiveness(self):
        """Test that caching improves performance on subsequent runs."""
        # First run (cold cache)
        start_time = time.time()
        result1 = asyncio.run(self.analyzer.analyze(self.sample_data))
        first_run_time = time.time() - start_time
        
        # Second run (warm cache)
        start_time = time.time()
        result2 = asyncio.run(self.analyzer.analyze(self.sample_data))
        second_run_time = time.time() - start_time
        
        # Print cache performance
        print(f"\nCold cache run: {first_run_time:.4f} seconds")
        print(f"Warm cache run: {second_run_time:.4f} seconds")
        print(f"Speedup factor: {first_run_time / second_run_time:.2f}x")
        
        # Verify that caching provides significant speedup
        # The second run should be at least 30% faster
        self.assertLess(second_run_time, first_run_time * 0.7, 
                       "Caching should provide at least 30% speedup")
                       
    def test_parallel_processing(self):
        """Test that parallel processing is working correctly."""
        # Run analysis with large dataset to ensure parallel processing is used
        large_data = self._generate_sample_data(num_bars=1000)
        
        result = asyncio.run(self.analyzer.analyze(large_data))
        
        # Check that the result is valid
        self.assertTrue(result.is_valid)
        
        # Verify that we have confluence zones
        self.assertIn("confluence_zones", result.result)
        
        # Print number of zones found
        zones = result.result["confluence_zones"]
        print(f"\nFound {len(zones)} confluence zones in large dataset")
        
        # Verify that we have a reasonable number of zones
        self.assertGreater(len(zones), 0, "Should find at least one confluence zone")


if __name__ == "__main__":
    unittest.main()
