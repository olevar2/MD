"""
Tests for Volume Analysis Indicators.

This module tests the implementation of volume-based analysis indicators,
particularly focusing on the Money Flow Index (MFI) indicator.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from feature_store_service.indicators.volume_analysis import MoneyFlowIndex


class TestMoneyFlowIndex(unittest.TestCase):
    """Test cases for Money Flow Index indicator."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample OHLCV data for testing
        dates = [datetime.now() + timedelta(days=i) for i in range(20)]
        
        # Create price data with a clear pattern for testing
        highs = np.linspace(100, 110, 10).tolist() + np.linspace(110, 100, 10).tolist()
        lows = np.linspace(98, 108, 10).tolist() + np.linspace(108, 98, 10).tolist()
        closes = np.linspace(99, 109, 10).tolist() + np.linspace(109, 99, 10).tolist()
        volumes = [1000] * 5 + [2000] * 5 + [2000] * 5 + [1000] * 5
        
        self.test_data = pd.DataFrame({
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=dates)
        
    def test_mfi_initialization(self):
        """Test MFI initialization with different parameters."""
        # Default parameters
        mfi = MoneyFlowIndex()
        self.assertEqual(mfi.window, 14)
        self.assertEqual(mfi.overbought, 80.0)
        self.assertEqual(mfi.oversold, 20.0)
        
        # Custom parameters
        custom_mfi = MoneyFlowIndex(window=10, overbought=75.0, oversold=25.0)
        self.assertEqual(custom_mfi.window, 10)
        self.assertEqual(custom_mfi.overbought, 75.0)
        self.assertEqual(custom_mfi.oversold, 25.0)
    
    def test_mfi_calculation(self):
        """Test MFI calculation functionality."""
        # Use a shorter window to ensure we have results with our test data
        mfi = MoneyFlowIndex(window=5)
        result = mfi.calculate(self.test_data)
        
        # Check that calculation runs without errors and expected columns are added
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("MFI", result.columns)
        self.assertIn("MFI_overbought", result.columns)
        self.assertIn("MFI_oversold", result.columns)
        
        # First 5 rows should have NaN values due to the window
        self.assertTrue(result["MFI"].iloc[:4].isna().all())
        
        # Check that remaining values are within the expected range (0-100)
        self.assertTrue((result["MFI"].iloc[5:] >= 0).all())
        self.assertTrue((result["MFI"].iloc[5:] <= 100).all())
        
    def test_mfi_signals(self):
        """Test MFI overbought/oversold signals."""
        # Create an MFI with custom thresholds we'll expect to be triggered in our test data
        mfi = MoneyFlowIndex(window=5, overbought=60, oversold=40)
        result = mfi.calculate(self.test_data)
        
        # Our data should trigger both overbought and oversold at some point
        self.assertTrue(result["MFI_overbought"].any())
        self.assertTrue(result["MFI_oversold"].any())
    
    def test_incremental_calculation(self):
        """Test MFI incremental calculation."""
        mfi = MoneyFlowIndex(window=5)
        
        # Create initial state for incremental calculation
        # This would typically come from a prior calculation
        previous_data = pd.Series({
            "high": 105.0,
            "low": 103.0,
            "close": 104.0,
            "volume": 2000,
            "typical_price": 104.0,
            "mfi_positive_sum": 800000,
            "mfi_negative_sum": 200000
        })
        
        # New data point
        new_data = {
            "high": 106.0,
            "low": 104.0,
            "close": 105.0,
            "volume": 2500
        }
        
        # Calculate incrementally
        result = mfi.calculate_incremental(previous_data, new_data)
        
        # Check that result is returned and has expected keys
        self.assertIsInstance(result, dict)
        self.assertIn("MFI", result)
        self.assertIn("MFI_overbought", result)
        self.assertIn("MFI_oversold", result)
        self.assertIn("typical_price", result)
        self.assertIn("mfi_positive_sum", result)
        self.assertIn("mfi_negative_sum", result)
        
        # MFI value should be between 0 and 100
        self.assertGreaterEqual(result["MFI"], 0)
        self.assertLessEqual(result["MFI"], 100)
        
    def test_get_params(self):
        """Test getting indicator parameters."""
        mfi = MoneyFlowIndex(window=12, overbought=75, oversold=25)
        params = mfi.get_params()
        
        self.assertIsInstance(params, dict)
        self.assertEqual(params["window"], 12)
        self.assertEqual(params["overbought"], 75)
        self.assertEqual(params["oversold"], 25)


if __name__ == '__main__':
    unittest.main()
