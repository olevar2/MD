"""
Unit tests for Elliott Wave utils module.
"""

import unittest
import pandas as pd
import numpy as np
from analysis_engine.analysis.advanced_ta.elliott_wave.utils import (
    detect_zigzag_points, detect_swing_points, calculate_wave_sharpness
)


class TestElliottWaveUtils(unittest.TestCase):
    """Test cases for Elliott Wave utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample price data
        dates = pd.date_range(start='2023-01-01', periods=100)
        self.df = pd.DataFrame({
            'open': np.random.normal(100, 1, 100),
            'high': np.random.normal(101, 1, 100),
            'low': np.random.normal(99, 1, 100),
            'close': np.random.normal(100, 1, 100)
        }, index=dates)
        
        # Ensure high is always >= close and open
        self.df['high'] = self.df[['high', 'open', 'close']].max(axis=1)
        
        # Ensure low is always <= close and open
        self.df['low'] = self.df[['low', 'open', 'close']].min(axis=1)
        
        # Create a clear uptrend for testing
        for i in range(20, 40):
            self.df.iloc[i, self.df.columns.get_loc('close')] = 100 + (i - 20) * 0.5
            self.df.iloc[i, self.df.columns.get_loc('high')] = 101 + (i - 20) * 0.5
            self.df.iloc[i, self.df.columns.get_loc('low')] = 99 + (i - 20) * 0.5
        
        # Create a clear downtrend for testing
        for i in range(60, 80):
            self.df.iloc[i, self.df.columns.get_loc('close')] = 110 - (i - 60) * 0.5
            self.df.iloc[i, self.df.columns.get_loc('high')] = 111 - (i - 60) * 0.5
            self.df.iloc[i, self.df.columns.get_loc('low')] = 109 - (i - 60) * 0.5
    
    def test_detect_zigzag_points(self):
        """Test zigzag points detection."""
        zigzag_points = detect_zigzag_points(
            self.df, 'close', 'high', 'low', 0.03
        )
        
        # Should detect at least some zigzag points
        self.assertGreater(len(zigzag_points), 0)
        
        # Each point should be a tuple of (timestamp, price, is_high)
        for point in zigzag_points:
            self.assertEqual(len(point), 3)
            self.assertIsInstance(point[0], pd.Timestamp)
            self.assertIsInstance(point[1], float)
            self.assertIsInstance(point[2], bool)
    
    def test_detect_swing_points(self):
        """Test swing points detection."""
        swing_points = detect_swing_points(self.df['close'], window=5)
        
        # Should detect at least some swing points
        self.assertGreater(len(swing_points), 0)
        
        # Each point should be a tuple of (index, price, "high"/"low")
        for point in swing_points:
            self.assertEqual(len(point), 3)
            self.assertIsInstance(point[0], int)
            self.assertIsInstance(point[1], float)
            self.assertIn(point[2], ["high", "low"])
    
    def test_calculate_wave_sharpness(self):
        """Test wave sharpness calculation."""
        # Test on uptrend section
        uptrend_sharpness = calculate_wave_sharpness(
            self.df, 20, 39, 'close'
        )
        
        # Test on downtrend section
        downtrend_sharpness = calculate_wave_sharpness(
            self.df, 60, 79, 'close'
        )
        
        # Sharpness should be between 0 and 1
        self.assertGreaterEqual(uptrend_sharpness, 0.0)
        self.assertLessEqual(uptrend_sharpness, 1.0)
        self.assertGreaterEqual(downtrend_sharpness, 0.0)
        self.assertLessEqual(downtrend_sharpness, 1.0)
        
        # Test invalid indices
        invalid_sharpness = calculate_wave_sharpness(
            self.df, 50, 50, 'close'
        )
        self.assertEqual(invalid_sharpness, 0.0)


if __name__ == "__main__":
    unittest.main()