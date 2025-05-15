"""
Unit tests for market regime detection.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from market_analysis_service.core.market_regime import MarketRegimeDetector
from market_analysis_service.models.market_analysis_models import MarketRegimeType

class TestMarketRegime(unittest.TestCase):
    """
    Unit tests for market regime detection.
    """
    
    def setUp(self):
        """
        Set up test data.
        """
        # Create a market regime detector
        self.market_regime_detector = MarketRegimeDetector()
        
        # Create test data
        self.create_test_data()
        
    def create_test_data(self):
        """
        Create test data for market regime detection.
        """
        # Create a DataFrame with OHLCV data
        dates = [datetime.now() - timedelta(days=i) for i in range(100)]
        dates.reverse()
        
        # Create price data with different regimes
        close = []
        high = []
        low = []
        
        for i in range(100):
            if i < 20:
                # Uptrend
                c = 100 + i * 0.5
            elif i < 40:
                # Range
                c = 110 + 5 * np.sin((i - 20) / 5)
            elif i < 60:
                # Downtrend
                c = 110 - (i - 40) * 0.5
            elif i < 80:
                # Volatile
                c = 90 + 10 * np.sin((i - 60) / 2) + np.random.normal(0, 5)
            else:
                # Uptrend
                c = 90 + (i - 80) * 0.5
                
            # Add some noise
            noise = np.random.normal(0, 1)
            c += noise
            
            # Create high and low
            h = c + abs(np.random.normal(0, 2))
            l = c - abs(np.random.normal(0, 2))
            
            close.append(c)
            high.append(h)
            low.append(l)
            
        # Create DataFrame
        self.test_data = pd.DataFrame({
            "timestamp": dates,
            "open": close,  # Use close as open for simplicity
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, 100)
        })
        
    def test_get_available_regimes(self):
        """
        Test getting available regimes.
        """
        regimes = self.market_regime_detector._get_available_regimes()
        
        # Check that we have regimes
        self.assertTrue(len(regimes) > 0)
        
        # Check that each regime has the required fields
        for regime in regimes:
            self.assertIn("id", regime)
            self.assertIn("name", regime)
            self.assertIn("description", regime)
            
    def test_detect_market_regime(self):
        """
        Test detecting market regimes.
        """
        # Detect market regimes
        regimes = self.market_regime_detector.detect_market_regime(
            data=self.test_data,
            window_size=10
        )
        
        # Check that we found at least one regime
        self.assertTrue(len(regimes) > 0)
        
        # Check that the regimes have the required fields
        for regime in regimes:
            self.assertIn("regime_type", regime)
            self.assertIn("start_index", regime)
            self.assertIn("end_index", regime)
            self.assertIn("start_date", regime)
            self.assertIn("end_date", regime)
            self.assertIn("confidence", regime)
            
            # Check that the regime type is valid
            self.assertIn(regime["regime_type"], [r.value for r in MarketRegimeType])
            
            # Check that the confidence is between 0 and 1
            self.assertGreaterEqual(regime["confidence"], 0)
            self.assertLessEqual(regime["confidence"], 1)
            
    def test_detect_trend(self):
        """
        Test detecting trend regimes.
        """
        # Detect trend regimes
        regimes = self.market_regime_detector._detect_trend(
            data=self.test_data,
            window_size=10,
            parameters={}
        )
        
        # Check that we found at least one regime
        self.assertTrue(len(regimes) > 0)
        
        # Check that the regimes have the required fields
        for regime in regimes:
            self.assertIn("regime_type", regime)
            self.assertIn("start_index", regime)
            self.assertIn("end_index", regime)
            self.assertIn("confidence", regime)
            
            # Check that the regime type is either trending up or trending down
            self.assertIn(regime["regime_type"], [MarketRegimeType.TRENDING_UP.value, MarketRegimeType.TRENDING_DOWN.value])
            
    def test_detect_volatility(self):
        """
        Test detecting volatile regimes.
        """
        # Detect volatile regimes
        regimes = self.market_regime_detector._detect_volatility(
            data=self.test_data,
            window_size=10,
            parameters={}
        )
        
        # Check that the regimes have the required fields
        for regime in regimes:
            self.assertIn("regime_type", regime)
            self.assertIn("start_index", regime)
            self.assertIn("end_index", regime)
            self.assertIn("confidence", regime)
            
            # Check that the regime type is volatile
            self.assertEqual(regime["regime_type"], MarketRegimeType.VOLATILE.value)
            
    def test_detect_range(self):
        """
        Test detecting ranging regimes.
        """
        # Detect ranging regimes
        regimes = self.market_regime_detector._detect_range(
            data=self.test_data,
            window_size=10,
            parameters={}
        )
        
        # Check that the regimes have the required fields
        for regime in regimes:
            self.assertIn("regime_type", regime)
            self.assertIn("start_index", regime)
            self.assertIn("end_index", regime)
            self.assertIn("confidence", regime)
            
            # Check that the regime type is ranging
            self.assertEqual(regime["regime_type"], MarketRegimeType.RANGING.value)
            
if __name__ == "__main__":
    unittest.main()