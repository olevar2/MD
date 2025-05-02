"""
Test module for Time-Price indicators.

This module tests the implementations of TPO (Time-Price-Opportunity) and Market Profile indicators.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from analysis_engine.analysis.advanced_ta.time_price_indicators import (
    TimeProfileIndicator,
    TPOIndicator,
    ValueArea
)


class TestTPOIndicator(unittest.TestCase):
    """Test cases for TPO Indicator."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample OHLCV data for testing
        dates = [datetime.now() + timedelta(hours=i) for i in range(24)]
        
        # Generate sample price data
        self.test_data = pd.DataFrame({
            'open': np.random.uniform(100, 102, len(dates)),
            'high': np.random.uniform(101, 103, len(dates)),
            'low': np.random.uniform(99, 101, len(dates)),
            'close': np.random.uniform(100, 102, len(dates)),
            'volume': np.random.uniform(1000, 5000, len(dates))
        }, index=dates)
        
    def test_tpo_initialization(self):
        """Test that TPO indicator initializes correctly with value_area_volume parameter."""
        # Test with default parameters
        tpo = TPOIndicator()
        self.assertEqual(tpo.value_area_volume, 0.7)  # Default value
        
        # Test with custom value_area_volume
        custom_value = 0.8
        tpo = TPOIndicator(value_area_volume=custom_value)
        self.assertEqual(tpo.value_area_volume, custom_value)
        
        # Verify the parameter is passed correctly to the parent class
        self.assertIsInstance(tpo, TimeProfileIndicator)
        
    def test_tpo_calculation(self):
        """Test TPO calculation functionality."""
        tpo = TPOIndicator(period="hour", price_precision=2)
        result = tpo.calculate(self.test_data)
        
        # Check that calculation runs without errors and returns a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check that TPO-specific columns are added
        self.assertIn("tpo_poc", result.columns)
        self.assertIn("tpo_value_area_high", result.columns)
        self.assertIn("tpo_value_area_low", result.columns)
        
    def test_incremental_calculation(self):
        """Test TPO incremental calculation functionality."""
        tpo = TPOIndicator(period="hour", price_precision=2)
        
        # Initialize incremental calculation
        state = tpo.initialize_incremental()
        self.assertIsNotNone(state)
        
        # Update with new data
        new_data = {
            "datetime": datetime.now(),
            "open": 101.0,
            "high": 102.0,
            "low": 100.0,
            "close": 101.5,
            "volume": 2000
        }
        
        updated_state = tpo.update_incremental(state, new_data)
        self.assertIsNotNone(updated_state)
        self.assertIn("price_levels", updated_state)


if __name__ == '__main__':
    unittest.main()
