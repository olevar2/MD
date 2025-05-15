"""
Unit tests for volatility analysis.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from market_analysis_service.core.volatility_analysis import VolatilityAnalyzer

class TestVolatilityAnalysis(unittest.TestCase):
    """
    Unit tests for volatility analysis.
    """
    
    def setUp(self):
        """
        Set up test data.
        """
        # Create a volatility analyzer
        self.volatility_analyzer = VolatilityAnalyzer()
        
        # Create test data
        self.create_test_data()
        
    def create_test_data(self):
        """
        Create test data for volatility analysis.
        """
        # Create a DataFrame with OHLCV data
        dates = [datetime.now() - timedelta(days=i) for i in range(100)]
        dates.reverse()
        
        # Create price data with different volatility regimes
        close = []
        high = []
        low = []
        
        for i in range(100):
            if i < 20:
                # Low volatility
                c = 100 + i * 0.1 + np.random.normal(0, 0.5)
            elif i < 40:
                # Medium volatility
                c = 102 + i * 0.1 + np.random.normal(0, 1.0)
            elif i < 60:
                # High volatility
                c = 106 + i * 0.1 + np.random.normal(0, 2.0)
            elif i < 80:
                # Medium volatility
                c = 112 + i * 0.1 + np.random.normal(0, 1.0)
            else:
                # Low volatility
                c = 116 + i * 0.1 + np.random.normal(0, 0.5)
                
            # Create high and low
            h = c + abs(np.random.normal(0, 0.5))
            l = c - abs(np.random.normal(0, 0.5))
            
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
        
    def test_analyze_volatility(self):
        """
        Test analyzing volatility.
        """
        # Analyze volatility
        volatility_results = self.volatility_analyzer.analyze_volatility(
            data=self.test_data,
            window_sizes=[5, 10, 20]
        )
        
        # Check that we have volatility data
        self.assertIn("volatility", volatility_results)
        
        # Check that we have regimes
        self.assertIn("regimes", volatility_results)
        
        # Check that we have forecasts
        self.assertIn("forecasts", volatility_results)
        
        # Check that we have term structure
        self.assertIn("term_structure", volatility_results)
        
        # Check that the volatility data has the requested window sizes
        for window in ["5", "10", "20"]:
            self.assertIn(window, volatility_results["volatility"])
            
            # Check that the window data has the required fields
            window_data = volatility_results["volatility"][window]
            self.assertIn("current", window_data)
            self.assertIn("average", window_data)
            self.assertIn("percentile", window_data)
            self.assertIn("rolling", window_data)
            
            # Check that the current volatility is positive
            self.assertGreaterEqual(window_data["current"], 0)
            
            # Check that the average volatility is positive
            self.assertGreaterEqual(window_data["average"], 0)
            
            # Check that the percentile is between 0 and 100
            self.assertGreaterEqual(window_data["percentile"], 0)
            self.assertLessEqual(window_data["percentile"], 100)
            
        # Check that the regimes data has the required fields
        regimes_data = volatility_results["regimes"]
        self.assertIn("current_regime", regimes_data)
        self.assertIn("regime_thresholds", regimes_data)
        self.assertIn("regime_history", regimes_data)
        
        # Check that the current regime is valid
        self.assertIn(regimes_data["current_regime"], ["low", "medium", "high", "unknown"])
        
        # Check that the term structure has the required fields
        for term in volatility_results["term_structure"]:
            self.assertIn("window", term)
            self.assertIn("volatility", term)
            
            # Check that the window is positive
            self.assertGreaterEqual(term["window"], 0)
            
            # Check that the volatility is positive
            self.assertGreaterEqual(term["volatility"], 0)
            
    def test_calculate_percentile(self):
        """
        Test calculating percentile.
        """
        # Create test series
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Calculate percentile for different values
        p0 = self.volatility_analyzer._calculate_percentile(0, series)
        p5 = self.volatility_analyzer._calculate_percentile(5, series)
        p10 = self.volatility_analyzer._calculate_percentile(10, series)
        p15 = self.volatility_analyzer._calculate_percentile(15, series)
        
        # Check that the percentiles are correct
        self.assertEqual(p0, 0)
        self.assertEqual(p5, 40)
        self.assertEqual(p10, 90)
        self.assertEqual(p15, 100)
        
    def test_calculate_volatility_regimes(self):
        """
        Test calculating volatility regimes.
        """
        # Calculate volatility regimes
        regimes = self.volatility_analyzer._calculate_volatility_regimes(
            returns=self.test_data["close"].pct_change().dropna(),
            parameters={}
        )
        
        # Check that we have the required fields
        self.assertIn("current_regime", regimes)
        self.assertIn("regime_thresholds", regimes)
        self.assertIn("regime_history", regimes)
        
        # Check that the current regime is valid
        self.assertIn(regimes["current_regime"], ["low", "medium", "high", "unknown"])
        
    def test_calculate_volatility_forecasts(self):
        """
        Test calculating volatility forecasts.
        """
        # Calculate volatility forecasts
        forecasts = self.volatility_analyzer._calculate_volatility_forecasts(
            returns=self.test_data["close"].pct_change().dropna(),
            parameters={}
        )
        
        # Check that we have the required fields
        self.assertIn("forecast", forecasts)
        self.assertIn("confidence_interval", forecasts)
        
        # Check that the forecast is positive or None
        if forecasts["forecast"] is not None:
            self.assertGreaterEqual(forecasts["forecast"], 0)
            
        # Check that the confidence interval is valid
        if forecasts["confidence_interval"] is not None:
            self.assertIn("lower", forecasts["confidence_interval"])
            self.assertIn("upper", forecasts["confidence_interval"])
            self.assertLessEqual(forecasts["confidence_interval"]["lower"], forecasts["confidence_interval"]["upper"])
            
    def test_calculate_volatility_term_structure(self):
        """
        Test calculating volatility term structure.
        """
        # Create test volatility data
        volatility = {
            "5": {"current": 0.1},
            "10": {"current": 0.15},
            "20": {"current": 0.2}
        }
        
        # Calculate volatility term structure
        term_structure = self.volatility_analyzer._calculate_volatility_term_structure(volatility)
        
        # Check that we have the correct number of terms
        self.assertEqual(len(term_structure), 3)
        
        # Check that the terms have the required fields
        for term in term_structure:
            self.assertIn("window", term)
            self.assertIn("volatility", term)
            
        # Check that the terms are sorted by window size
        self.assertEqual(term_structure[0]["window"], 5)
        self.assertEqual(term_structure[1]["window"], 10)
        self.assertEqual(term_structure[2]["window"], 20)
        
if __name__ == "__main__":
    unittest.main()
