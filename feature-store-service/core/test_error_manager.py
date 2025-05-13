"""
Tests for the indicator error management system.
"""
import unittest
from datetime import datetime
import pandas as pd
import numpy as np
from core.error_manager import (
    IndicatorErrorManager,
    CalculationError,
    DataError,
    ParameterError
)

class TestIndicatorErrorManager(unittest.TestCase):
    """Test suite for the IndicatorErrorManager"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.error_manager = IndicatorErrorManager()
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2025-01-01', periods=5),
            'close': [100.0, 101.0, 102.0, 101.5, 102.5]
        })

    def test_register_error(self):
        """Test error registration."""
        error = CalculationError(
            "Division by zero",
            details={'line': 42, 'input_data': self.test_data}
        )
        self.error_manager.register_error(error, "RSI")
        
        self.assertIn("RSI", self.error_manager.error_registry)
        self.assertEqual(len(self.error_manager.error_registry["RSI"]), 1)
        self.assertEqual(
            self.error_manager.error_registry["RSI"][0].error_type,
            "calculation_error"
        )

    def test_handle_calculation_error(self):
        """Test handling of calculation errors."""
        error = CalculationError(
            "Invalid period",
            details={
                'parameters': {'period': 14},
                'input_data': self.test_data
            }
        )
        result = self.error_manager.handle_error(error, "Moving Average")
        
        # The handler should attempt recovery
        self.assertIsNone(result)  # In this case, recovery isn't implemented
        self.assertIn("Moving Average", self.error_manager.error_registry)

    def test_handle_data_error(self):
        """Test handling of data errors."""
        # Create data with some NaN values
        data_with_nans = self.test_data.copy()
        data_with_nans.loc[2, 'close'] = np.nan
        
        error = DataError(
            "Missing values in data",
            details={
                'input_data': data_with_nans,
                'invalid_rows': [2]
            }
        )
        result = self.error_manager.handle_error(error, "RSI")
        
        # Should return cleaned data without the invalid row
        self.assertIsNotNone(result)
        self.assertEqual(len(result), len(data_with_nans) - 1)

    def test_handle_parameter_error(self):
        """Test handling of parameter errors."""
        error = ParameterError(
            "Invalid parameter values",
            details={
                'parameters': {
                    'period': -14,
                    'price_source': 'invalid'
                }
            }
        )
        result = self.error_manager.handle_error(error, "MACD")
        
        # Should return corrected parameters
        self.assertIsNotNone(result)
        self.assertGreater(result['period'], 0)
        self.assertEqual(result['price_source'], 'close')

    def test_get_error_summary_specific_indicator(self):
        """Test getting error summary for a specific indicator."""
        # Register multiple errors
        self.error_manager.register_error(
            CalculationError("Error 1"),
            "RSI"
        )
        self.error_manager.register_error(
            DataError("Error 2"),
            "RSI"
        )
        self.error_manager.register_error(
            ParameterError("Error 3"),
            "MACD"
        )
        
        summary = self.error_manager.get_error_summary("RSI")
        
        self.assertEqual(summary['total_errors'], 2)
        self.assertEqual(len(summary['affected_indicators']), 1)
        self.assertEqual(summary['affected_indicators'][0], "RSI")

    def test_get_error_summary_all_indicators(self):
        """Test getting error summary for all indicators."""
        # Register multiple errors
        self.error_manager.register_error(
            CalculationError("Error 1"),
            "RSI"
        )
        self.error_manager.register_error(
            DataError("Error 2"),
            "MACD"
        )
        
        summary = self.error_manager.get_error_summary()
        
        self.assertEqual(summary['total_errors'], 2)
        self.assertEqual(len(summary['affected_indicators']), 2)
        self.assertIn("RSI", summary['affected_indicators'])
        self.assertIn("MACD", summary['affected_indicators'])

    def test_error_recovery_strategies(self):
        """Test various error recovery strategies."""
        # Test calculation error recovery
        calc_error = CalculationError(
            "Period too large",
            details={
                'parameters': {'period': 50},
                'input_data': self.test_data
            }
        )
        calc_result = self.error_manager.handle_error(calc_error, "SMA")
        # Recovery not implemented in this case
        self.assertIsNone(calc_result)
        
        # Test data error recovery with NaN values
        data_with_nans = self.test_data.copy()
        data_with_nans.loc[2:3, 'close'] = np.nan
        data_error = DataError(
            "Missing values",
            details={'input_data': data_with_nans}
        )
        data_result = self.error_manager.handle_error(data_error, "RSI")
        # Should return interpolated data
        self.assertIsNotNone(data_result)
        self.assertFalse(data_result.isna().any().any())
        
        # Test parameter error recovery
        param_error = ParameterError(
            "Invalid parameters",
            details={
                'parameters': {
                    'period': -5,
                    'price_source': 'invalid',
                    'time_period': 'invalid'
                }
            }
        )
        param_result = self.error_manager.handle_error(param_error, "MACD")
        # Should return corrected parameters
        self.assertIsNotNone(param_result)
        self.assertGreater(param_result['period'], 0)
        self.assertEqual(param_result['price_source'], 'close')
        self.assertEqual(param_result['time_period'], '1d')

if __name__ == '__main__':
    unittest.main()
