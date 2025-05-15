"""
Unit tests for correlation analysis.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from market_analysis_service.core.correlation_analysis import CorrelationAnalyzer

class TestCorrelationAnalysis(unittest.TestCase):
    """
    Unit tests for correlation analysis.
    """
    
    def setUp(self):
        """
        Set up test data.
        """
        # Create a correlation analyzer
        self.correlation_analyzer = CorrelationAnalyzer()
        
        # Create test data
        self.create_test_data()
        
    def create_test_data(self):
        """
        Create test data for correlation analysis.
        """
        # Create a DataFrame with OHLCV data for multiple symbols
        dates = [datetime.now() - timedelta(days=i) for i in range(100)]
        dates.reverse()
        
        # Create price data with different correlations
        self.test_data = {}
        
        # Symbol 1: Base symbol
        close1 = []
        
        for i in range(100):
            c = 100 + i * 0.5 + np.random.normal(0, 2)
            close1.append(c)
            
        self.test_data["SYMBOL1"] = pd.DataFrame({
            "timestamp": dates,
            "open": close1,  # Use close as open for simplicity
            "high": [c + abs(np.random.normal(0, 1)) for c in close1],
            "low": [c - abs(np.random.normal(0, 1)) for c in close1],
            "close": close1,
            "volume": np.random.randint(1000, 10000, 100)
        })
        
        # Symbol 2: Highly correlated with Symbol 1
        close2 = [c * 1.2 + np.random.normal(0, 1) for c in close1]
        
        self.test_data["SYMBOL2"] = pd.DataFrame({
            "timestamp": dates,
            "open": close2,
            "high": [c + abs(np.random.normal(0, 1)) for c in close2],
            "low": [c - abs(np.random.normal(0, 1)) for c in close2],
            "close": close2,
            "volume": np.random.randint(1000, 10000, 100)
        })
        
        # Symbol 3: Negatively correlated with Symbol 1
        close3 = [150 - c + np.random.normal(0, 2) for c in close1]
        
        self.test_data["SYMBOL3"] = pd.DataFrame({
            "timestamp": dates,
            "open": close3,
            "high": [c + abs(np.random.normal(0, 1)) for c in close3],
            "low": [c - abs(np.random.normal(0, 1)) for c in close3],
            "close": close3,
            "volume": np.random.randint(1000, 10000, 100)
        })
        
        # Symbol 4: Uncorrelated with Symbol 1
        close4 = []
        
        for i in range(100):
            c = 100 + 10 * np.sin(i / 10) + np.random.normal(0, 2)
            close4.append(c)
            
        self.test_data["SYMBOL4"] = pd.DataFrame({
            "timestamp": dates,
            "open": close4,
            "high": [c + abs(np.random.normal(0, 1)) for c in close4],
            "low": [c - abs(np.random.normal(0, 1)) for c in close4],
            "close": close4,
            "volume": np.random.randint(1000, 10000, 100)
        })
        
    def test_analyze_correlations(self):
        """
        Test analyzing correlations.
        """
        # Analyze correlations
        correlation_results = self.correlation_analyzer.analyze_correlations(
            data=self.test_data,
            window_size=20,
            method="pearson"
        )
        
        # Check that we have a correlation matrix
        self.assertIn("correlation_matrix", correlation_results)
        
        # Check that we have correlation pairs
        self.assertIn("correlation_pairs", correlation_results)
        
        # Check that the correlation matrix has all symbols
        for symbol in self.test_data.keys():
            self.assertIn(symbol, correlation_results["correlation_matrix"])
            
        # Check that the correlation pairs have the required fields
        for pair in correlation_results["correlation_pairs"]:
            self.assertIn("symbol1", pair)
            self.assertIn("symbol2", pair)
            self.assertIn("correlation", pair)
            self.assertIn("p_value", pair)
            
            # Check that the correlation is between -1 and 1
            self.assertGreaterEqual(pair["correlation"], -1)
            self.assertLessEqual(pair["correlation"], 1)
            
            # Check that the p-value is between 0 and 1
            self.assertGreaterEqual(pair["p_value"], 0)
            self.assertLessEqual(pair["p_value"], 1)
            
    def test_correlation_breakdown(self):
        """
        Test analyzing correlation breakdown.
        """
        # Analyze correlation breakdown
        breakdown_results = self.correlation_analyzer.analyze_correlation_breakdown(
            data=self.test_data,
            window_size=20,
            method="pearson",
            breakdown_threshold=0.3
        )
        
        # Check that we have breakdown pairs
        self.assertIn("breakdown_pairs", breakdown_results)
        
        # Check that the breakdown pairs have the required fields
        for pair in breakdown_results["breakdown_pairs"]:
            self.assertIn("symbol1", pair)
            self.assertIn("symbol2", pair)
            self.assertIn("current_correlation", pair)
            self.assertIn("correlation_change", pair)
            self.assertIn("breakdown_risk", pair)
            
            # Check that the correlation is between -1 and 1
            self.assertGreaterEqual(pair["current_correlation"], -1)
            self.assertLessEqual(pair["current_correlation"], 1)
            
            # Check that the correlation change is positive
            self.assertGreaterEqual(pair["correlation_change"], 0)
            
            # Check that the breakdown risk is positive
            self.assertGreaterEqual(pair["breakdown_risk"], 0)
            
    def test_calculate_p_value(self):
        """
        Test calculating p-value.
        """
        # Create test series
        x = pd.Series(np.random.normal(0, 1, 100))
        y = pd.Series(np.random.normal(0, 1, 100))
        
        # Calculate p-value
        p_value = self.correlation_analyzer._calculate_p_value(x, y, "pearson")
        
        # Check that the p-value is between 0 and 1
        self.assertGreaterEqual(p_value, 0)
        self.assertLessEqual(p_value, 1)
        
if __name__ == "__main__":
    unittest.main()
