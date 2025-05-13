"""
Tests for the moving averages indicators.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the common_lib module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from common_lib.indicators.moving_averages import (
    SimpleMovingAverage,
    ExponentialMovingAverage,
    WeightedMovingAverage
)


class TestMovingAverages(unittest.TestCase):
    """Test cases for moving average indicators."""

    def setUp(self):
        """Set up test data."""
        # Create sample data
        self.data = pd.DataFrame({
            'close': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'open': [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1],
            'high': [1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2],
            'low': [0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.9],
            'volume': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        })

    def test_simple_moving_average(self):
        """Test SimpleMovingAverage indicator."""
        # Create indicator with default parameters
        sma = SimpleMovingAverage()

        # Calculate indicator
        result = sma.calculate(self.data)

        # Check that the result has the expected column
        self.assertIn('sma_14', result.columns)

        # Check that the result has the same number of rows as the input
        self.assertEqual(len(result), len(self.data))

        # Check that the first 13 values are NaN (window size - 1)
        self.assertTrue(np.isnan(result['sma_14'].iloc[:13]).all())

        # Create indicator with custom parameters
        sma = SimpleMovingAverage({'window': 3, 'column': 'close'})

        # Calculate indicator
        result = sma.calculate(self.data)

        # Check that the result has the expected column
        self.assertIn('sma_3', result.columns)

        # Check that the first 2 values are NaN (window size - 1)
        self.assertTrue(np.isnan(result['sma_3'].iloc[:2]).all())

        # Check the calculated values
        expected_values = [np.nan, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        for i in range(len(expected_values)):
            if np.isnan(expected_values[i]):
                self.assertTrue(np.isnan(result['sma_3'].iloc[i]))
            else:
                self.assertAlmostEqual(result['sma_3'].iloc[i], expected_values[i])

    def test_exponential_moving_average(self):
        """Test ExponentialMovingAverage indicator."""
        # Create indicator with default parameters
        ema = ExponentialMovingAverage()

        # Calculate indicator
        result = ema.calculate(self.data)

        # Check that the result has the expected column
        self.assertIn('ema_14', result.columns)

        # Check that the result has the same number of rows as the input
        self.assertEqual(len(result), len(self.data))

        # Create indicator with custom parameters
        ema = ExponentialMovingAverage({'window': 3, 'column': 'close'})

        # Calculate indicator
        result = ema.calculate(self.data)

        # Check that the result has the expected column
        self.assertIn('ema_3', result.columns)

        # Note: pandas ewm() doesn't produce NaN for the first value, it uses the first value as-is
        # So we don't check for NaN here

        # Check that the values are increasing (since our test data is strictly increasing)
        for i in range(0, len(result) - 1):
            self.assertLess(result['ema_3'].iloc[i], result['ema_3'].iloc[i + 1])

    def test_weighted_moving_average(self):
        """Test WeightedMovingAverage indicator."""
        # Create indicator with default parameters
        wma = WeightedMovingAverage()

        # Calculate indicator
        result = wma.calculate(self.data)

        # Check that the result has the expected column
        self.assertIn('wma_14', result.columns)

        # Check that the result has the same number of rows as the input
        self.assertEqual(len(result), len(self.data))

        # Create indicator with custom parameters
        wma = WeightedMovingAverage({'window': 3, 'column': 'close'})

        # Calculate indicator
        result = wma.calculate(self.data)

        # Check that the result has the expected column
        self.assertIn('wma_3', result.columns)

        # Check that the first 2 values are NaN (window size - 1)
        self.assertTrue(np.isnan(result['wma_3'].iloc[:2]).all())

        # Check the calculated values (manually calculated)
        # WMA = (1*price_1 + 2*price_2 + 3*price_3) / (1+2+3)
        # For the first valid value: (1*1.0 + 2*2.0 + 3*3.0) / 6 = 2.33
        self.assertAlmostEqual(result['wma_3'].iloc[2], 2.33, places=2)

        # For the second valid value: (1*2.0 + 2*3.0 + 3*4.0) / 6 = 3.33
        self.assertAlmostEqual(result['wma_3'].iloc[3], 3.33, places=2)

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test with invalid window type
        with self.assertRaises(TypeError):
            SimpleMovingAverage({'window': 'invalid', 'column': 'close'})

        # Test with invalid column type
        with self.assertRaises(TypeError):
            SimpleMovingAverage({'window': 14, 'column': 123})

        # Test with missing required parameter
        with self.assertRaises(ValueError):
            SimpleMovingAverage({'window': 14})

        # Test with invalid column name
        sma = SimpleMovingAverage({'window': 14, 'column': 'invalid'})
        with self.assertRaises(ValueError):
            sma.calculate(self.data)


if __name__ == '__main__':
    unittest.main()