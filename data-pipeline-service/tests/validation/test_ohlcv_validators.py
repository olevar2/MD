"""
Test ohlcv validators module.

This module provides functionality for...
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from data_pipeline_service.validation.ohlcv_validators import (
    OHLCVSchemaValidator, 
    OHLCVLogicalValidator,
    OHLCVOutlierDetector
)

class TestOHLCVSchemaValidator(unittest.TestCase):
    """Test suite for the OHLCV Schema Validator"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.validator = OHLCVSchemaValidator()
        
        # Valid OHLCV data
        self.valid_data = pd.DataFrame({
            'timestamp': [datetime(2025, 4, 1, 12, 0), datetime(2025, 4, 1, 12, 5)],
            'open': [1.1234, 1.1235],
            'high': [1.1240, 1.1238],
            'low': [1.1230, 1.1233],
            'close': [1.1235, 1.1237],
            'volume': [1000, 1200]
        })
        
    def test_valid_ohlcv_data(self):
        """Test validation with valid OHLCV data"""
        result = self.validator.validate(self.valid_data)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
    
    def test_missing_columns(self):
        """Test validation with missing columns"""
        invalid_data = self.valid_data.drop(columns=['high'])
        result = self.validator.validate(invalid_data)
        self.assertFalse(result.is_valid)
        self.assertTrue(any('missing required column' in str(err) for err in result.errors))
    
    def test_invalid_data_types(self):
        """Test validation with invalid data types"""
        invalid_data = self.valid_data.copy()
        invalid_data['volume'] = invalid_data['volume'].astype(str)
        result = self.validator.validate(invalid_data)
        self.assertFalse(result.is_valid)
        self.assertTrue(any('invalid data type' in str(err) for err in result.errors))


class TestOHLCVLogicalValidator(unittest.TestCase):
    """Test suite for the OHLCV Logical Validator"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.validator = OHLCVLogicalValidator()
        
        # Valid OHLCV data
        self.valid_data = pd.DataFrame({
            'timestamp': [datetime(2025, 4, 1, 12, 0), datetime(2025, 4, 1, 12, 5)],
            'open': [1.1234, 1.1235],
            'high': [1.1240, 1.1238],
            'low': [1.1230, 1.1233],
            'close': [1.1235, 1.1237],
            'volume': [1000, 1200]
        })
    
    def test_valid_ohlcv_logic(self):
        """Test logical validation with valid OHLCV data"""
        result = self.validator.validate(self.valid_data)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
    
    def test_high_lower_than_low(self):
        """Test validation when high is lower than low"""
        invalid_data = self.valid_data.copy()
        invalid_data.loc[0, 'high'] = 1.1220  # High less than low
        result = self.validator.validate(invalid_data)
        self.assertFalse(result.is_valid)
        self.assertTrue(any('high value must be greater than low value' in str(err) for err in result.errors))
    
    def test_open_outside_range(self):
        """Test validation when open is outside high-low range"""
        invalid_data = self.valid_data.copy()
        invalid_data.loc[0, 'open'] = 1.1250  # Open greater than high
        result = self.validator.validate(invalid_data)
        self.assertFalse(result.is_valid)
        self.assertTrue(any('open value must be within high-low range' in str(err) for err in result.errors))


class TestOHLCVOutlierDetector(unittest.TestCase):
    """Test suite for the OHLCV Outlier Detector"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.detector = OHLCVOutlierDetector(z_threshold=3.0)
        
        # Generate a larger dataset for testing outliers
        np.random.seed(42)
        timestamps = pd.date_range('2025-04-01', periods=100, freq='5min')
        base_price = 1.1200
        
        # Create normal price movements
        price_changes = np.random.normal(0, 0.0005, 100)
        prices = base_price + np.cumsum(price_changes)
        
        self.normal_data = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': prices + np.random.uniform(0.0001, 0.0010, 100),
            'low': prices - np.random.uniform(0.0001, 0.0010, 100),
            'close': prices + np.random.normal(0, 0.0002, 100),
            'volume': np.random.randint(800, 1500, 100)
        })
        
        # Create a copy with outliers
        self.outlier_data = self.normal_data.copy()
        # Add price spike outlier
        self.outlier_data.loc[50, 'high'] = base_price + 0.0500
        # Add volume outlier
        self.outlier_data.loc[75, 'volume'] = 50000
    
    def test_no_outliers_detected(self):
        """Test outlier detection with normal data"""
        result = self.detector.validate(self.normal_data)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.warnings), 0)
    
    def test_price_outlier_detected(self):
        """Test outlier detection with price spike"""
        result = self.detector.validate(self.outlier_data)
        self.assertFalse(result.is_valid)
        self.assertTrue(any('price outlier detected' in str(warning) for warning in result.warnings))
    
    def test_custom_threshold(self):
        """Test outlier detection with custom threshold"""
        strict_detector = OHLCVOutlierDetector(z_threshold=2.0)
        lenient_detector = OHLCVOutlierDetector(z_threshold=5.0)
        
        # The strict detector should find more outliers
        strict_result = strict_detector.validate(self.normal_data)
        lenient_result = lenient_detector.validate(self.outlier_data)
        
        self.assertTrue(len(strict_result.warnings) >= len(lenient_result.warnings))


if __name__ == '__main__':
    unittest.main()
