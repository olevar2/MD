"""
Unit tests for harmonic patterns utils module.
"""

import unittest
import pandas as pd
import numpy as np
from feature_store_service.indicators.harmonic_patterns.utils import (
    identify_pivots, calculate_ratio, ratio_matches
)


class TestHarmonicPatternsUtils(unittest.TestCase):
    """Test cases for harmonic patterns utility functions."""
    
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
        
        # Create clear pivot points for testing
        # High pivot at index 25
        self.df.iloc[25, self.df.columns.get_loc('high')] = 110
        for i in range(20, 25):
            self.df.iloc[i, self.df.columns.get_loc('high')] = 105 + (i - 20)
        for i in range(26, 31):
            self.df.iloc[i, self.df.columns.get_loc('high')] = 105 + (30 - i)
            
        # Low pivot at index 50
        self.df.iloc[50, self.df.columns.get_loc('low')] = 90
        for i in range(45, 50):
            self.df.iloc[i, self.df.columns.get_loc('low')] = 95 - (i - 45)
        for i in range(51, 56):
            self.df.iloc[i, self.df.columns.get_loc('low')] = 95 - (55 - i)
    
    def test_identify_pivots(self):
        """Test pivot points identification."""
        result = identify_pivots(self.df, window=5)
        
        # Should have pivot_high and pivot_low columns
        self.assertIn('pivot_high', result.columns)
        self.assertIn('pivot_low', result.columns)
        self.assertIn('pivot_high_value', result.columns)
        self.assertIn('pivot_low_value', result.columns)
        
        # Should detect the high pivot at index 25
        self.assertEqual(result.iloc[25]['pivot_high'], 1)
        self.assertEqual(result.iloc[25]['pivot_high_value'], 110)
        
        # Should detect the low pivot at index 50
        self.assertEqual(result.iloc[50]['pivot_low'], 1)
        self.assertEqual(result.iloc[50]['pivot_low_value'], 90)
    
    def test_calculate_ratio(self):
        """Test ratio calculation."""
        # Test normal case
        self.assertEqual(calculate_ratio(10, 5), 2.0)
        
        # Test with negative values (should use absolute values)
        self.assertEqual(calculate_ratio(-10, 5), 2.0)
        self.assertEqual(calculate_ratio(10, -5), 2.0)
        
        # Test division by zero
        self.assertEqual(calculate_ratio(10, 0), float('inf'))
    
    def test_ratio_matches(self):
        """Test ratio matching with tolerance."""
        # Exact match
        self.assertTrue(ratio_matches(1.618, 1.618, 0.05))
        
        # Within tolerance
        self.assertTrue(ratio_matches(1.60, 1.618, 0.05))
        self.assertTrue(ratio_matches(1.65, 1.618, 0.05))
        
        # Outside tolerance
        self.assertFalse(ratio_matches(1.50, 1.618, 0.05))
        self.assertFalse(ratio_matches(1.75, 1.618, 0.05))


if __name__ == "__main__":
    unittest.main()