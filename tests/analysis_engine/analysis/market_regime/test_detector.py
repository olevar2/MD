"""
Unit tests for the Market Regime Detector.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from analysis_engine.analysis.market_regime.detector import RegimeDetector
from analysis_engine.analysis.market_regime.models import FeatureSet


class TestRegimeDetector(unittest.TestCase):
    """Test cases for the RegimeDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = RegimeDetector()
        self.sample_data = self._create_sample_data()
        self.trending_data = self._create_trending_data()
        self.ranging_data = self._create_ranging_data()
        self.volatile_data = self._create_volatile_data()
    
    def _create_sample_data(self):
        """Create a sample price dataset for testing."""
        # Create a date range
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create price data with a slight uptrend
        close = np.linspace(100, 110, 100) + np.random.normal(0, 1, 100)
        high = close + np.random.uniform(0, 2, 100)
        low = close - np.random.uniform(0, 2, 100)
        open_price = close.copy()
        np.random.shuffle(open_price)
        volume = np.random.uniform(1000, 5000, 100)
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)
        
        return df
    
    def _create_trending_data(self):
        """Create a sample dataset with a strong trend."""
        # Create a date range
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create price data with a strong uptrend
        close = np.linspace(100, 150, 100) + np.random.normal(0, 2, 100)
        high = close + np.random.uniform(0, 3, 100)
        low = close - np.random.uniform(0, 3, 100)
        open_price = close - np.random.uniform(-2, 2, 100)
        volume = np.random.uniform(1000, 5000, 100) * (1 + np.linspace(0, 0.5, 100))
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)
        
        return df
    
    def _create_ranging_data(self):
        """Create a sample dataset with a ranging market."""
        # Create a date range
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create price data with a range
        base = 100 + np.sin(np.linspace(0, 4*np.pi, 100)) * 10
        close = base + np.random.normal(0, 1, 100)
        high = close + np.random.uniform(0, 2, 100)
        low = close - np.random.uniform(0, 2, 100)
        open_price = close - np.random.uniform(-2, 2, 100)
        volume = np.random.uniform(1000, 3000, 100)
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)
        
        return df
    
    def _create_volatile_data(self):
        """Create a sample dataset with high volatility."""
        # Create a date range
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create price data with high volatility
        close = 100 + np.cumsum(np.random.normal(0, 3, 100))
        high = close + np.random.uniform(1, 5, 100)
        low = close - np.random.uniform(1, 5, 100)
        open_price = close - np.random.uniform(-3, 3, 100)
        volume = np.random.uniform(3000, 10000, 100)
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)
        
        return df
    
    def test_feature_extraction(self):
        """Test that feature extraction returns the expected features."""
        features = self.detector.extract_features(self.sample_data)
        
        # Check that we have a FeatureSet object
        self.assertIsInstance(features, FeatureSet)
        
        # Check that all required features are present
        self.assertIsNotNone(features.volatility)
        self.assertIsNotNone(features.trend_strength)
        self.assertIsNotNone(features.momentum)
        self.assertIsNotNone(features.mean_reversion)
        self.assertIsNotNone(features.range_width)
        
        # Check that additional features are present
        self.assertIsNotNone(features.additional_features)
        self.assertIn('price_velocity', features.additional_features)
        self.assertIn('volume_trend', features.additional_features)
        self.assertIn('swing_strength', features.additional_features)
    
    def test_trending_market_features(self):
        """Test feature extraction on trending market data."""
        features = self.detector.extract_features(self.trending_data)
        
        # In a trending market, we expect:
        # - Higher trend_strength
        # - Consistent momentum direction
        # - Lower mean_reversion
        self.assertGreater(features.trend_strength, 0.3)
        self.assertNotEqual(features.momentum, 0)
        
        # Convert to dict for easier testing
        feature_dict = features.to_dict()
        
        # Check additional features
        self.assertIn('price_velocity', feature_dict)
        # In a trending market, price velocity should align with the trend
        if features.momentum > 0:
            self.assertGreater(feature_dict['price_velocity'], 0)
        elif features.momentum < 0:
            self.assertLess(feature_dict['price_velocity'], 0)
    
    def test_ranging_market_features(self):
        """Test feature extraction on ranging market data."""
        features = self.detector.extract_features(self.ranging_data)
        
        # In a ranging market, we expect:
        # - Lower trend_strength
        # - Higher mean_reversion
        # - Defined range_width
        self.assertLess(features.trend_strength, 0.5)
        self.assertGreater(features.range_width, 0)
    
    def test_volatile_market_features(self):
        """Test feature extraction on volatile market data."""
        features = self.detector.extract_features(self.volatile_data)
        
        # In a volatile market, we expect:
        # - Higher volatility
        # - Potentially higher swing_strength
        self.assertGreater(features.volatility, 1.0)
        self.assertGreater(features.additional_features['swing_strength'], 0.01)
    
    def test_atr_calculation(self):
        """Test the ATR calculation."""
        atr = self.detector._calculate_atr(self.sample_data)
        
        # ATR should be a Series
        self.assertIsInstance(atr, pd.Series)
        
        # ATR should be positive
        self.assertTrue((atr.dropna() >= 0).all())
        
        # ATR should have the same length as the input data
        self.assertEqual(len(atr), len(self.sample_data))
    
    def test_adx_calculation(self):
        """Test the ADX calculation."""
        adx = self.detector._calculate_adx(self.sample_data)
        
        # ADX should be a Series
        self.assertIsInstance(adx, pd.Series)
        
        # ADX should be between 0 and 100
        valid_adx = adx.dropna()
        self.assertTrue((valid_adx >= 0).all() and (valid_adx <= 100).all())
    
    def test_rsi_calculation(self):
        """Test the RSI calculation."""
        rsi = self.detector._calculate_rsi(self.sample_data)
        
        # RSI should be a Series
        self.assertIsInstance(rsi, pd.Series)
        
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        self.assertTrue((valid_rsi >= 0).all() and (valid_rsi <= 100).all())
    
    def test_missing_columns(self):
        """Test that an error is raised when required columns are missing."""
        # Create data with missing columns
        bad_data = self.sample_data[['open', 'close']].copy()
        
        # This should raise a ValueError
        with self.assertRaises(ValueError):
            self.detector.extract_features(bad_data)
    
    def test_feature_consistency(self):
        """Test that features are consistent for similar data."""
        # Create two similar datasets
        data1 = self.sample_data.copy()
        data2 = data1 + 0.001  # Small difference
        
        features1 = self.detector.extract_features(data1)
        features2 = self.detector.extract_features(data2)
        
        # Features should be similar for similar data
        self.assertAlmostEqual(features1.volatility, features2.volatility, places=2)
        self.assertAlmostEqual(features1.trend_strength, features2.trend_strength, places=2)
        self.assertAlmostEqual(features1.momentum, features2.momentum, places=2)


if __name__ == '__main__':
    unittest.main()