"""
Tests for the BaseIndicator class.
"""

import unittest
import pandas as pd
from typing import Dict, Any

from common_lib.indicators.base_indicator import BaseIndicator


class SimpleMovingAverage(BaseIndicator):
    """Simple Moving Average indicator for testing."""
    
    category = "trend"
    name = "SimpleMovingAverage"
    default_params = {"window": 10}
    required_params = {"window": int}
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate SMA for the 'close' column."""
        self.validate_input(data, ["close"])
        result = data.copy()
        window = self.params["window"]
        result[f"SMA_{window}"] = result["close"].rolling(window=window).mean()
        return result


class TestBaseIndicator(unittest.TestCase):
    """Tests for the BaseIndicator class."""
    
    def setUp(self):
        """Set up test data."""
        self.data = pd.DataFrame({
            "close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            "volume": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
        })
    
    def test_initialization(self):
        """Test initialization with parameters."""
        # Test with default parameters
        sma = SimpleMovingAverage()
        self.assertEqual(sma.params["window"], 10)
        
        # Test with custom parameters
        sma = SimpleMovingAverage({"window": 5})
        self.assertEqual(sma.params["window"], 5)
    
    def test_validate_params(self):
        """Test parameter validation."""
        # Test with valid parameters
        sma = SimpleMovingAverage({"window": 5})
        self.assertEqual(sma.params["window"], 5)
        
        # Test with missing required parameter
        with self.assertRaises(ValueError):
            SimpleMovingAverage({"not_window": 5})
        
        # Test with incorrect parameter type
        with self.assertRaises(TypeError):
            SimpleMovingAverage({"window": "5"})
        
        # Test with int for float parameter (should auto-convert)
        class TestIndicator(BaseIndicator):
            required_params = {"value": float}
        
        indicator = TestIndicator({"value": 5})
        self.assertEqual(indicator.params["value"], 5.0)
        self.assertIsInstance(indicator.params["value"], float)
    
    def test_calculate(self):
        """Test the calculate method."""
        sma = SimpleMovingAverage({"window": 3})
        result = sma.calculate(self.data)
        
        # Check that the original data is not modified
        self.assertIn("close", result.columns)
        self.assertIn("volume", result.columns)
        
        # Check that the SMA column is added
        self.assertIn("SMA_3", result.columns)
        
        # Check the SMA values
        # First two values should be NaN (not enough data for window=3)
        self.assertTrue(pd.isna(result["SMA_3"][0]))
        self.assertTrue(pd.isna(result["SMA_3"][1]))
        
        # Check some actual values
        self.assertEqual(result["SMA_3"][2], (1.0 + 2.0 + 3.0) / 3)
        self.assertEqual(result["SMA_3"][3], (2.0 + 3.0 + 4.0) / 3)
    
    def test_validate_input(self):
        """Test input validation."""
        sma = SimpleMovingAverage()
        
        # Test with valid input
        self.assertTrue(sma.validate_input(self.data, ["close"]))
        
        # Test with missing column
        with self.assertRaises(ValueError):
            sma.validate_input(self.data, ["missing_column"])
    
    def test_get_metadata(self):
        """Test the get_metadata method."""
        metadata = SimpleMovingAverage.get_metadata()
        
        self.assertEqual(metadata["name"], "SimpleMovingAverage")
        self.assertEqual(metadata["category"], "trend")
        self.assertIn("description", metadata)
        self.assertEqual(metadata["default_params"], {"window": 10})
        self.assertEqual(metadata["required_params"], {"window": int})
    
    def test_get_params(self):
        """Test the get_params method."""
        sma = SimpleMovingAverage({"window": 5})
        params = sma.get_params()
        
        self.assertEqual(params, {"window": 5})
        
        # Ensure that modifying the returned params doesn't affect the original
        params["window"] = 20
        self.assertEqual(sma.params["window"], 5)
    
    def test_update_parameters(self):
        """Test the update_parameters method."""
        sma = SimpleMovingAverage()
        self.assertEqual(sma.params["window"], 10)
        
        # Update parameter
        sma.update_parameters(window=5)
        self.assertEqual(sma.params["window"], 5)
        
        # Test with invalid parameter
        with self.assertRaises(ValueError):
            sma.update_parameters(invalid_param=5)
    
    def test_string_representation(self):
        """Test string representation."""
        sma = SimpleMovingAverage({"window": 5})
        
        # Test __str__
        self.assertEqual(str(sma), "SimpleMovingAverage(window=5)")
        
        # Test __repr__
        self.assertEqual(repr(sma), "SimpleMovingAverage(window=5)")


if __name__ == "__main__":
    unittest.main()