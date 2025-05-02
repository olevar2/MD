"""
Tests for the indicator data validation service.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from feature_store_service.validation.data_validator import (
    DataValidationService,
    IndicatorValidationType,
    ValidationSeverity
)

class TestDataValidationService(unittest.TestCase):
    """Test suite for the DataValidationService"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.validator = DataValidationService()
        
        # Create valid OHLCV test data
        dates = pd.date_range(start='2025-01-01', periods=5, freq='1min')
        self.valid_ohlcv = pd.DataFrame({
            'timestamp': dates,
            'open': [100.0, 101.0, 102.0, 101.5, 102.5],
            'high': [102.0, 103.0, 104.0, 103.5, 104.5],
            'low': [99.0, 100.0, 101.0, 100.5, 101.5],
            'close': [101.0, 102.0, 103.0, 102.5, 103.5],
            'volume': [1000, 1100, 1200, 1150, 1250]
        })

    def test_validate_ohlcv_valid_data(self):
        """Test OHLCV validation with valid data."""
        result = self.validator.validate('ohlcv', self.valid_ohlcv)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.severity, ValidationSeverity.INFO)

    def test_validate_ohlcv_invalid_high_low(self):
        """Test OHLCV validation with invalid high/low relationship."""
        invalid_data = self.valid_ohlcv.copy()
        # Make high less than low
        invalid_data.loc[0, 'high'] = 98.0
        
        result = self.validator.validate('ohlcv', invalid_data)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.severity, ValidationSeverity.ERROR)
        self.assertIn('High < Low', result.message)

    def test_validate_ohlcv_missing_columns(self):
        """Test OHLCV validation with missing columns."""
        invalid_data = self.valid_ohlcv.drop(columns=['volume'])
        
        result = self.validator.validate('ohlcv', invalid_data)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.severity, ValidationSeverity.ERROR)
        self.assertIn('Missing required columns', result.message)

    def test_validate_indicator_input_valid(self):
        """Test indicator input validation with valid data and parameters."""
        params = {
            'period': 14,
            'price_source': 'close'
        }
        
        result = self.validator.validate(
            'indicator_input',
            self.valid_ohlcv,
            params=params
        )
        self.assertTrue(result.is_valid)

    def test_validate_indicator_input_invalid_params(self):
        """Test indicator input validation with invalid parameters."""
        params = {
            'period': -1,  # Invalid period
            'price_source': 'close'
        }
        
        result = self.validator.validate(
            'indicator_input',
            self.valid_ohlcv,
            params=params
        )
        self.assertFalse(result.is_valid)
        self.assertIn('period must be positive', str(result.details))

    def test_validate_time_series_valid(self):
        """Test time series validation with valid data."""
        result = self.validator.validate('time_series', self.valid_ohlcv)
        self.assertTrue(result.is_valid)

    def test_validate_time_series_gaps(self):
        """Test time series validation with gaps."""
        data = self.valid_ohlcv.copy()
        # Create a gap by removing a row
        data = pd.concat([data.iloc[:2], data.iloc[3:]])
        
        result = self.validator.validate('time_series', data)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.severity, ValidationSeverity.WARNING)
        self.assertIn('Found gaps in time series', result.message)

    def test_validate_calculation_valid(self):
        """Test calculation validation with valid results."""
        # Create a simple moving average result
        result_data = pd.DataFrame({
            'timestamp': self.valid_ohlcv['timestamp'],
            'sma': [100.0, 101.0, 102.0, 101.5, 102.5]
        })
        
        expected_properties = {
            'columns': ['timestamp', 'sma'],
            'length': 5
        }
        
        result = self.validator.validate(
            'calculation',
            result_data,
            expected_properties=expected_properties
        )
        self.assertTrue(result.is_valid)

    def test_validate_calculation_with_nans(self):
        """Test calculation validation with NaN values."""
        result_data = pd.DataFrame({
            'timestamp': self.valid_ohlcv['timestamp'],
            'sma': [np.nan, 101.0, 102.0, 101.5, 102.5]
        })
        
        result = self.validator.validate('calculation', result_data)
        self.assertFalse(result.is_valid)
        self.assertIn('nan_values', str(result.details))

    def test_validate_calculation_wrong_length(self):
        """Test calculation validation with incorrect length."""
        result_data = pd.DataFrame({
            'timestamp': self.valid_ohlcv['timestamp'][:-1],
            'sma': [100.0, 101.0, 102.0, 101.5]
        })
        
        expected_properties = {
            'columns': ['timestamp', 'sma'],
            'length': 5
        }
        
        result = self.validator.validate(
            'calculation',
            result_data,
            expected_properties=expected_properties
        )
        self.assertFalse(result.is_valid)
        self.assertIn('wrong_length', str(result.details))

    def test_invalid_validation_type(self):
        """Test validation with invalid validation type."""
        result = self.validator.validate('invalid_type', self.valid_ohlcv)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.severity, ValidationSeverity.ERROR)
        self.assertIn('Unknown validation type', result.message)

if __name__ == '__main__':
    unittest.main()
