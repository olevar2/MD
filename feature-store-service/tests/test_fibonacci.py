"""
Tests for Fibonacci indicators in feature_store_service/indicators/fibonacci.py
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core.fibonacci import (
    FibonacciRetracement,
    FibonacciExtension,
    FibonacciFan,
    FibonacciTimeZones,
    FibonacciCircles,
    FibonacciClusters,
    TrendDirection
)


class TestFibonacciIndicators(unittest.TestCase):
    """Test suite for Fibonacci indicators."""

    def setUp(self):
        """Set up test data."""
        # Create sample OHLCV data with a clear trend for testing
        dates = [datetime.now() + timedelta(days=i) for i in range(100)]
        
        # Create an uptrend followed by a downtrend
        close_prices = np.concatenate([
            np.linspace(100, 200, 50),  # Uptrend
            np.linspace(200, 150, 50)   # Downtrend
        ])
        
        # Add some noise to the data
        noise = np.random.normal(0, 2, 100)
        close_prices = close_prices + noise
        
        # Create high and low prices around close
        high_prices = close_prices + np.random.uniform(1, 5, 100)
        low_prices = close_prices - np.random.uniform(1, 5, 100)
        open_prices = close_prices - np.random.uniform(-3, 3, 100)
        volume = np.random.uniform(1000, 5000, 100)
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        }, index=dates)
        
        # Create manual points for testing
        self.uptrend_points = {
            'start_idx': 0,
            'end_idx': 49,
            'retracement_idx': 60
        }
        
        self.downtrend_points = {
            'start_idx': 49,
            'end_idx': 99,
            'retracement_idx': 75
        }

    def test_fibonacci_retracement_auto_detect(self):
        """Test Fibonacci Retracement with auto-detection."""
        # Initialize with auto-detection
        fib_retracement = FibonacciRetracement(
            levels=[0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0],
            swing_lookback=30,
            auto_detect_swings=True
        )
        
        # Calculate retracement levels
        result = fib_retracement.calculate(self.data)
        
        # Check that retracement columns exist
        for level in [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]:
            level_str = str(level).replace('.', '_')
            self.assertIn(f'fib_retracement_{level_str}', result.columns)
        
        # Check that start and end points are marked
        self.assertIn('fib_retracement_start', result.columns)
        self.assertIn('fib_retracement_end', result.columns)
        
        # At least one row should be marked as start and end
        self.assertTrue(result['fib_retracement_start'].any())
        self.assertTrue(result['fib_retracement_end'].any())

    def test_fibonacci_retracement_manual_points(self):
        """Test Fibonacci Retracement with manual points."""
        # Initialize with manual points
        fib_retracement = FibonacciRetracement(
            levels=[0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0],
            swing_lookback=30,
            auto_detect_swings=False,
            manual_points=self.uptrend_points
        )
        
        # Calculate retracement levels
        result = fib_retracement.calculate(self.data)
        
        # Check that retracement columns exist
        for level in [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]:
            level_str = str(level).replace('.', '_')
            self.assertIn(f'fib_retracement_{level_str}', result.columns)
        
        # Check that start and end points match manual points
        start_idx = self.data.index[self.uptrend_points['start_idx']]
        end_idx = self.data.index[self.uptrend_points['end_idx']]
        
        self.assertTrue(result.loc[start_idx, 'fib_retracement_start'])
        self.assertTrue(result.loc[end_idx, 'fib_retracement_end'])
        
        # Check that 0.0 level matches end point price and 1.0 level matches start point price
        # For uptrend: 0.0 is at the high, 1.0 is at the low
        start_price = self.data.loc[start_idx, 'close']
        end_price = self.data.loc[end_idx, 'close']
        
        # Allow for small floating point differences
        self.assertAlmostEqual(
            result.loc[end_idx, 'fib_retracement_0_0'],
            end_price,
            delta=0.01
        )
        self.assertAlmostEqual(
            result.loc[end_idx, 'fib_retracement_1_0'],
            start_price,
            delta=0.01
        )

    def test_fibonacci_extension_auto_detect(self):
        """Test Fibonacci Extension with auto-detection."""
        # Initialize with auto-detection
        fib_extension = FibonacciExtension(
            levels=[0.0, 0.382, 0.618, 1.0, 1.382, 1.618, 2.0, 2.618],
            swing_lookback=30,
            auto_detect_swings=True
        )
        
        # Calculate extension levels
        result = fib_extension.calculate(self.data)
        
        # Check that extension columns exist
        for level in [0.0, 0.382, 0.618, 1.0, 1.382, 1.618, 2.0, 2.618]:
            level_str = str(level).replace('.', '_')
            self.assertIn(f'fib_extension_{level_str}', result.columns)
        
        # Check that start, end, and retracement points are marked
        self.assertIn('fib_extension_start', result.columns)
        self.assertIn('fib_extension_end', result.columns)
        self.assertIn('fib_extension_retracement', result.columns)
        
        # At least one row should be marked for each point
        self.assertTrue(result['fib_extension_start'].any())
        self.assertTrue(result['fib_extension_end'].any())
        self.assertTrue(result['fib_extension_retracement'].any())

    def test_fibonacci_extension_manual_points(self):
        """Test Fibonacci Extension with manual points."""
        # Initialize with manual points
        fib_extension = FibonacciExtension(
            levels=[0.0, 0.382, 0.618, 1.0, 1.382, 1.618, 2.0, 2.618],
            swing_lookback=30,
            auto_detect_swings=False,
            manual_points=self.downtrend_points
        )
        
        # Calculate extension levels
        result = fib_extension.calculate(self.data)
        
        # Check that extension columns exist
        for level in [0.0, 0.382, 0.618, 1.0, 1.382, 1.618, 2.0, 2.618]:
            level_str = str(level).replace('.', '_')
            self.assertIn(f'fib_extension_{level_str}', result.columns)
        
        # Check that points match manual points
        start_idx = self.data.index[self.downtrend_points['start_idx']]
        end_idx = self.data.index[self.downtrend_points['end_idx']]
        retr_idx = self.data.index[self.downtrend_points['retracement_idx']]
        
        self.assertTrue(result.loc[start_idx, 'fib_extension_start'])
        self.assertTrue(result.loc[end_idx, 'fib_extension_end'])
        self.assertTrue(result.loc[retr_idx, 'fib_extension_retracement'])

    def test_fibonacci_fan(self):
        """Test Fibonacci Fan calculation."""
        # Initialize Fibonacci Fan
        fib_fan = FibonacciFan(
            levels=[0.236, 0.382, 0.5, 0.618, 0.786],
            swing_lookback=30,
            auto_detect_swings=True,
            projection_bars=20
        )
        
        # Calculate fan levels
        result = fib_fan.calculate(self.data)
        
        # Check that fan columns exist
        for level in [0.236, 0.382, 0.5, 0.618, 0.786]:
            level_str = str(level).replace('.', '_')
            self.assertIn(f'fib_fan_{level_str}', result.columns)
        
        # Check that start and end points are marked
        self.assertIn('fib_fan_start', result.columns)
        self.assertIn('fib_fan_end', result.columns)
        
        # At least one row should be marked for each point
        self.assertTrue(result['fib_fan_start'].any())
        self.assertTrue(result['fib_fan_end'].any())

    def test_fibonacci_time_zones(self):
        """Test Fibonacci Time Zones calculation."""
        # Initialize Fibonacci Time Zones
        fib_time_zones = FibonacciTimeZones(
            fib_sequence=[1, 2, 3, 5, 8, 13, 21, 34],
            auto_detect_start=True,
            max_zones=5
        )
        
        # Calculate time zones
        result = fib_time_zones.calculate(self.data)
        
        # Check that time zone columns exist
        self.assertIn('fib_time_zone', result.columns)
        self.assertIn('fib_time_zone_start', result.columns)
        
        # Check that specific time zone columns exist
        for i in range(1, 6):  # max_zones=5
            self.assertIn(f'fib_time_zone_{i}', result.columns)
        
        # At least one row should be marked as start
        self.assertTrue(result['fib_time_zone_start'].any())
        
        # At least one time zone should be marked
        time_zone_columns = [f'fib_time_zone_{i}' for i in range(1, 6)]
        time_zone_marked = False
        for col in time_zone_columns:
            if result[col].any():
                time_zone_marked = True
                break
        self.assertTrue(time_zone_marked)

    def test_fibonacci_circles(self):
        """Test Fibonacci Circles calculation."""
        # Initialize Fibonacci Circles
        fib_circles = FibonacciCircles(
            levels=[0.382, 0.5, 0.618, 1.0, 1.618],
            swing_lookback=30,
            auto_detect_points=True,
            projection_bars=20,
            time_price_ratio=1.0
        )
        
        # Calculate circle levels
        result = fib_circles.calculate(self.data)
        
        # Check that circle columns exist
        for level in [0.382, 0.5, 0.618, 1.0, 1.618]:
            level_str = str(level).replace('.', '_')
            self.assertIn(f'fib_circle_upper_{level_str}', result.columns)
            self.assertIn(f'fib_circle_lower_{level_str}', result.columns)
        
        # Check that center and radius points are marked
        self.assertIn('fib_circle_center', result.columns)
        self.assertIn('fib_circle_radius_ref', result.columns)
        
        # At least one row should be marked for each point
        self.assertTrue(result['fib_circle_center'].any())
        self.assertTrue(result['fib_circle_radius_ref'].any())

    def test_fibonacci_clusters(self):
        """Test Fibonacci Clusters calculation."""
        # First calculate various Fibonacci levels to create input for clusters
        data = self.data.copy()
        
        # Calculate retracement levels
        fib_retracement = FibonacciRetracement(auto_detect_swings=True)
        data = fib_retracement.calculate(data)
        
        # Calculate extension levels
        fib_extension = FibonacciExtension(auto_detect_swings=True)
        data = fib_extension.calculate(data)
        
        # Initialize Fibonacci Clusters
        fib_clusters = FibonacciClusters(
            price_column='close',
            cluster_threshold=3,
            price_tolerance=0.5
        )
        
        # Calculate clusters
        result = fib_clusters.calculate(data)
        
        # Check that cluster columns exist
        self.assertIn('fib_cluster_strength', result.columns)
        self.assertIn('fib_cluster_count', result.columns)
        
        # There should be at least one cluster identified
        self.assertTrue((result['fib_cluster_count'] > 0).any())

    def test_get_info_methods(self):
        """Test that get_info methods return proper information."""
        # Test each indicator's get_info method
        indicators = [
            FibonacciRetracement,
            FibonacciExtension,
            FibonacciFan,
            FibonacciTimeZones,
            FibonacciCircles,
            FibonacciClusters
        ]
        
        for indicator_class in indicators:
            info = indicator_class.get_info()
            
            # Check that required fields exist
            self.assertIn('name', info)
            self.assertIn('description', info)
            self.assertIn('category', info)
            self.assertIn('parameters', info)
            
            # Check that parameters is a list
            self.assertIsInstance(info['parameters'], list)
            
            # Check that each parameter has required fields
            for param in info['parameters']:
                self.assertIn('name', param)
                self.assertIn('description', param)
                self.assertIn('type', param)
                self.assertIn('default', param)


if __name__ == '__main__':
    unittest.main()