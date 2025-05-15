"""
Unit tests for support and resistance detection.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from market_analysis_service.core.support_resistance import SupportResistanceDetector
from market_analysis_service.models.market_analysis_models import SupportResistanceMethod

class TestSupportResistance(unittest.TestCase):
    """
    Unit tests for support and resistance detection.
    """
    
    def setUp(self):
        """
        Set up test data.
        """
        # Create a support and resistance detector
        self.support_resistance_detector = SupportResistanceDetector()
        
        # Create test data
        self.create_test_data()
        
    def create_test_data(self):
        """
        Create test data for support and resistance detection.
        """
        # Create a DataFrame with OHLCV data
        dates = [datetime.now() - timedelta(days=i) for i in range(100)]
        dates.reverse()
        
        # Create price data with clear support and resistance levels
        close = []
        high = []
        low = []
        
        for i in range(100):
            if i < 20:
                # Range between support at 100 and resistance at 110
                c = 100 + 5 * np.sin(i / 5) + 5
            elif i < 40:
                # Range between support at 110 and resistance at 120
                c = 110 + 5 * np.sin((i - 20) / 5) + 5
            elif i < 60:
                # Range between support at 120 and resistance at 130
                c = 120 + 5 * np.sin((i - 40) / 5) + 5
            elif i < 80:
                # Range between support at 110 and resistance at 120
                c = 110 + 5 * np.sin((i - 60) / 5) + 5
            else:
                # Range between support at 100 and resistance at 110
                c = 100 + 5 * np.sin((i - 80) / 5) + 5
                
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
        
    def test_get_available_methods(self):
        """
        Test getting available methods.
        """
        methods = self.support_resistance_detector._get_available_methods()
        
        # Check that we have methods
        self.assertTrue(len(methods) > 0)
        
        # Check that each method has the required fields
        for method in methods:
            self.assertIn("id", method)
            self.assertIn("name", method)
            self.assertIn("description", method)
            
    def test_identify_support_resistance(self):
        """
        Test identifying support and resistance levels.
        """
        # Identify support and resistance levels
        levels = self.support_resistance_detector.identify_support_resistance(
            data=self.test_data,
            methods=[SupportResistanceMethod.PRICE_SWINGS, SupportResistanceMethod.MOVING_AVERAGE],
            levels_count=10
        )
        
        # Check that we found at least one level
        self.assertTrue(len(levels) > 0)
        
        # Check that the levels have the required fields
        for level in levels:
            self.assertIn("price", level)
            self.assertIn("type", level)
            self.assertIn("strength", level)
            self.assertIn("method", level)
            
            # Check that the type is either support or resistance
            self.assertIn(level["type"], ["support", "resistance"])
            
            # Check that the method is one of the requested methods
            self.assertIn(level["method"], [SupportResistanceMethod.PRICE_SWINGS.value, SupportResistanceMethod.MOVING_AVERAGE.value])
            
    def test_identify_levels_price_swings(self):
        """
        Test identifying levels using price swings method.
        """
        # Identify levels using price swings method
        levels = self.support_resistance_detector._identify_levels_price_swings(
            data=self.test_data,
            parameters={}
        )
        
        # Check that we found at least one level
        self.assertTrue(len(levels) > 0)
        
        # Check that the levels have the required fields
        for level in levels:
            self.assertIn("price", level)
            self.assertIn("type", level)
            self.assertIn("strength", level)
            self.assertIn("method", level)
            self.assertIn("touches", level)
            
            # Check that the method is price swings
            self.assertEqual(level["method"], SupportResistanceMethod.PRICE_SWINGS.value)
            
    def test_identify_levels_moving_average(self):
        """
        Test identifying levels using moving average method.
        """
        # Identify levels using moving average method
        levels = self.support_resistance_detector._identify_levels_moving_average(
            data=self.test_data,
            parameters={}
        )
        
        # Check that we found at least one level
        self.assertTrue(len(levels) > 0)
        
        # Check that the levels have the required fields
        for level in levels:
            self.assertIn("price", level)
            self.assertIn("type", level)
            self.assertIn("strength", level)
            self.assertIn("method", level)
            
            # Check that the method is moving average
            self.assertEqual(level["method"], SupportResistanceMethod.MOVING_AVERAGE.value)
            
    def test_remove_duplicate_levels(self):
        """
        Test removing duplicate levels.
        """
        # Create some duplicate levels
        levels = [
            {"price": 100.0, "type": "support", "strength": 80.0, "method": "price_swings"},
            {"price": 100.1, "type": "support", "strength": 90.0, "method": "price_swings"},  # Duplicate of the first level
            {"price": 110.0, "type": "resistance", "strength": 70.0, "method": "price_swings"},
            {"price": 120.0, "type": "resistance", "strength": 60.0, "method": "price_swings"}
        ]
        
        # Remove duplicate levels
        unique_levels = self.support_resistance_detector._remove_duplicate_levels(levels)
        
        # Check that we removed the duplicate level
        self.assertEqual(len(unique_levels), 3)
        
        # Check that we kept the stronger level
        self.assertEqual(unique_levels[0]["price"], 100.1)
        self.assertEqual(unique_levels[0]["strength"], 90.0)
        
if __name__ == "__main__":
    unittest.main()