"""
Unit tests for base indicator functionality.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from feature_store_service.indicators.base_indicator import BaseIndicator

class TestBaseIndicator(unittest.TestCase):
    """Test cases for BaseIndicator class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample OHLCV data
        self.sample_data = pd.DataFrame({
            "open": [100.0, 100.5, 101.5, 102.0, 101.5],
            "high": [101.0, 102.0, 103.0, 102.5, 102.0],
            "low": [99.0, 100.0, 101.0, 101.0, 100.5],
            "close": [100.5, 101.5, 102.5, 101.5, 101.0],
            "volume": [1000, 1200, 1500, 1300, 1100]
        }, index=pd.date_range(start=datetime(2025, 1, 1), periods=5, freq="H"))

    def test_required_columns(self):
        """Test validation of required columns."""
        class TestIndicator(BaseIndicator):
            required_columns = ["close", "volume"]
            
            def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
                return data
        
        # Should work with all required columns
        indicator = TestIndicator()
        result = indicator.calculate(self.sample_data)
        self.assertIsInstance(result, pd.DataFrame)
        
        # Should raise error when missing columns
        missing_data = self.sample_data.drop(columns=["volume"])
        with self.assertRaises(ValueError):
            indicator.calculate(missing_data)

    def test_parameter_validation(self):
        """Test parameter validation."""
        class TestIndicator(BaseIndicator):
            params = {
                "period": {"type": "int", "min": 1, "max": 100, "default": 14},
                "price": {"type": "str", "options": ["close", "open"], "default": "close"}
            }
            
            def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
                return data
        
        # Test valid parameters
        indicator = TestIndicator(period=20, price="close")
        self.assertEqual(indicator.period, 20)
        self.assertEqual(indicator.price, "close")
        
        # Test invalid period
        with self.assertRaises(ValueError):
            TestIndicator(period=0)
        
        # Test invalid price option
        with self.assertRaises(ValueError):
            TestIndicator(price="invalid")

    def test_empty_data_handling(self):
        """Test handling of empty DataFrames."""
        class TestIndicator(BaseIndicator):
            def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
                return data
        
        indicator = TestIndicator()
        empty_data = pd.DataFrame()
        result = indicator.calculate(empty_data)
        self.assertTrue(result.empty)

    def test_nan_handling(self):
        """Test handling of NaN values."""
        class TestIndicator(BaseIndicator):
            def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
                result = data.copy()
                result["indicator"] = data["close"].rolling(window=2).mean()
                return result
        
        # Create data with NaN
        data_with_nan = self.sample_data.copy()
        data_with_nan.loc[data_with_nan.index[2], "close"] = np.nan
        
        indicator = TestIndicator()
        result = indicator.calculate(data_with_nan)
        
        # Verify NaN is propagated correctly
        self.assertTrue(np.isnan(result.loc[result.index[2], "indicator"]))
        self.assertTrue(np.isnan(result.loc[result.index[3], "indicator"]))

if __name__ == "__main__":
    unittest.main()
