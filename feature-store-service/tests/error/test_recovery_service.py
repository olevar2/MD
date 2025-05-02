"""
Tests for the error recovery service.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from feature_store_service.error.error_manager import (
    CalculationError,
    DataError,
    ParameterError
)
from feature_store_service.error.recovery_service import (
    ErrorRecoveryService,
    DataRecoveryStrategy,
    CalculationRecoveryStrategy,
    ParameterRecoveryStrategy
)

class TestErrorRecoveryService(unittest.TestCase):
    """Test suite for the ErrorRecoveryService."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.recovery_service = ErrorRecoveryService()
        
        # Create sample data for testing
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2025-01-01', periods=5),
            'open': [100.0, 101.0, np.nan, 101.5, 102.5],
            'high': [102.0, 103.0, 104.0, 103.5, 104.5],
            'low': [99.0, 100.0, 101.0, 100.5, 101.5],
            'close': [101.0, 102.0, np.nan, 102.5, 103.5],
            'volume': [1000, 1100, 1200, 1150, 1250]
        })

    def test_data_recovery_missing_values(self):
        """Test recovery from missing values in data."""
        error = DataError(
            message="Missing values detected",
            details={'input_data': self.test_data}
        )
        
        result = self.recovery_service.recover(error)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.isna().any().any())  # No NaN values

    def test_data_recovery_outliers(self):
        """Test recovery from outliers in data."""
        # Create data with outliers
        data_with_outliers = self.test_data.copy()
        data_with_outliers.loc[2, 'close'] = 1000.0  # Obvious outlier
        
        error = DataError(
            message="Outliers detected",
            details={
                'input_data': data_with_outliers,
                'has_outliers': True
            }
        )
        
        result = self.recovery_service.recover(error)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertNotEqual(result.loc[2, 'close'], 1000.0)  # Outlier should be fixed

    def test_calculation_recovery_period_adjustment(self):
        """Test recovery from calculation errors through period adjustment."""
        error = CalculationError(
            message="Period too large for available data",
            details={
                'parameters': {
                    'period': 50,
                    'price_source': 'close'
                }
            }
        )
        
        result = self.recovery_service.recover(error)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertLess(result['period'], 50)  # Period should be reduced

    def test_calculation_recovery_method_adjustment(self):
        """Test recovery from calculation errors through method adjustment."""
        error = CalculationError(
            message="EMA calculation failed",
            details={
                'parameters': {
                    'period': 14,
                    'method': 'ema'
                }
            }
        )
        
        result = self.recovery_service.recover(error)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertEqual(result['method'], 'sma')  # Should switch to SMA

    def test_parameter_recovery_invalid_values(self):
        """Test recovery from invalid parameter values."""
        error = ParameterError(
            message="Invalid parameter values",
            details={
                'parameters': {
                    'period': -14,
                    'price_source': 'invalid'
                }
            }
        )
        
        result = self.recovery_service.recover(error)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertGreater(result['period'], 0)
        self.assertEqual(result['price_source'], 'close')

    def test_parameter_recovery_missing_values(self):
        """Test recovery from missing parameter values."""
        error = ParameterError(
            message="Missing parameters",
            details={
                'parameters': {
                    'period': 14
                }
            }
        )
        
        result = self.recovery_service.recover(error)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn('price_source', result)  # Should add default price_source

    def test_recovery_statistics(self):
        """Test recovery statistics tracking."""
        # Create and recover from multiple errors
        errors = [
            DataError("Missing values", {'input_data': self.test_data}),
            CalculationError("Invalid period", {'parameters': {'period': -14}}),
            ParameterError("Invalid price source", {'parameters': {'price_source': 'invalid'}})
        ]
        
        for error in errors:
            self.recovery_service.recover(error)
            
        stats = self.recovery_service.get_statistics()
        
        self.assertIn('data', stats)
        self.assertIn('calculation', stats)
        self.assertIn('parameter', stats)
        
        for strategy_stats in stats.values():
            self.assertIn('success_count', strategy_stats)
            self.assertIn('failure_count', strategy_stats)
            self.assertIn('success_rate', strategy_stats)

    def test_data_recovery_strategy_directly(self):
        """Test DataRecoveryStrategy implementation directly."""
        strategy = DataRecoveryStrategy()
        
        # Test with missing values
        error = DataError(
            message="Missing values",
            details={'input_data': self.test_data}
        )
        
        self.assertTrue(strategy.can_handle(error))
        result = strategy.recover(error)
        self.assertIsNotNone(result)
        self.assertFalse(result.isna().any().any())

    def test_calculation_recovery_strategy_directly(self):
        """Test CalculationRecoveryStrategy implementation directly."""
        strategy = CalculationRecoveryStrategy()
        
        # Test with calculation error
        error = CalculationError(
            message="Period too large",
            details={'parameters': {'period': 50}}
        )
        
        self.assertTrue(strategy.can_handle(error))
        result = strategy.recover(error)
        self.assertIsNotNone(result)
        self.assertLess(result['period'], 50)

    def test_parameter_recovery_strategy_directly(self):
        """Test ParameterRecoveryStrategy implementation directly."""
        strategy = ParameterRecoveryStrategy()
        
        # Test with parameter error
        error = ParameterError(
            message="Invalid parameters",
            details={'parameters': {'period': -14}}
        )
        
        self.assertTrue(strategy.can_handle(error))
        result = strategy.recover(error)
        self.assertIsNotNone(result)
        self.assertGreater(result['period'], 0)

    def test_unrecoverable_error(self):
        """Test handling of unrecoverable errors."""
        # Create an error with insufficient information
        error = DataError(
            message="Data error",
            details={}  # No input data provided
        )
        
        result = self.recovery_service.recover(error)
        self.assertIsNone(result)  # Should return None for unrecoverable error

if __name__ == '__main__':
    unittest.main()
