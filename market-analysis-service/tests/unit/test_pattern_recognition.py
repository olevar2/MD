"""
Unit tests for pattern recognition.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from market_analysis_service.core.pattern_recognition import PatternRecognizer
from market_analysis_service.models.market_analysis_models import PatternType

class TestPatternRecognition(unittest.TestCase):
    """
    Unit tests for pattern recognition.
    """
    
    def setUp(self):
        """
        Set up test data.
        """
        # Create a pattern recognizer
        self.pattern_recognizer = PatternRecognizer()
        
        # Create test data
        self.create_test_data()
        
    def create_test_data(self):
        """
        Create test data for pattern recognition.
        """
        # Create a DataFrame with OHLCV data
        dates = [datetime.now() - timedelta(days=i) for i in range(100)]
        dates.reverse()
        
        # Create a head and shoulders pattern
        close = []
        high = []
        low = []
        
        for i in range(100):
            if i < 20:
                # Initial uptrend
                c = 100 + i
            elif i < 30:
                # Left shoulder
                c = 120 - (i - 20) * 0.5
            elif i < 40:
                # Rise to head
                c = 115 + (i - 30) * 1.5
            elif i < 50:
                # Fall from head
                c = 130 - (i - 40) * 1.5
            elif i < 60:
                # Rise to right shoulder
                c = 115 + (i - 50) * 0.5
            elif i < 70:
                # Fall from right shoulder
                c = 120 - (i - 60) * 1.0
            else:
                # Final downtrend
                c = 110 - (i - 70) * 0.5
                
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
        
    def test_get_available_patterns(self):
        """
        Test getting available patterns.
        """
        patterns = self.pattern_recognizer._get_available_patterns()
        
        # Check that we have patterns
        self.assertTrue(len(patterns) > 0)
        
        # Check that each pattern has the required fields
        for pattern in patterns:
            self.assertIn("id", pattern)
            self.assertIn("name", pattern)
            self.assertIn("description", pattern)
            self.assertIn("min_bars", pattern)
            
    def test_recognize_patterns(self):
        """
        Test recognizing patterns.
        """
        # Recognize patterns
        patterns = self.pattern_recognizer.recognize_patterns(
            data=self.test_data,
            pattern_types=[PatternType.HEAD_AND_SHOULDERS],
            min_confidence=0.5
        )
        
        # Check that we found at least one pattern
        self.assertTrue(len(patterns) > 0)
        
        # Check that the pattern has the required fields
        for pattern in patterns:
            self.assertIn("pattern_type", pattern)
            self.assertIn("start_index", pattern)
            self.assertIn("end_index", pattern)
            self.assertIn("confidence", pattern)
            
            # Check that the pattern type is head and shoulders
            self.assertEqual(pattern["pattern_type"], PatternType.HEAD_AND_SHOULDERS.value)
            
            # Check that the confidence is at least the minimum
            self.assertGreaterEqual(pattern["confidence"], 0.5)
            
    def test_find_local_maxima(self):
        """
        Test finding local maxima.
        """
        # Create a simple array with known maxima
        data = np.array([1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 1, 2, 3, 2, 1])
        
        # Find local maxima
        maxima = self.pattern_recognizer._find_local_maxima(data, window=1)
        
        # Check that we found the correct maxima
        self.assertEqual(len(maxima), 2)
        self.assertEqual(maxima[0], 2)
        self.assertEqual(maxima[1], 7)
        
    def test_find_local_minima(self):
        """
        Test finding local minima.
        """
        # Create a simple array with known minima
        data = np.array([3, 2, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 2, 3])
        
        # Find local minima
        minima = self.pattern_recognizer._find_local_minima(data, window=1)
        
        # Check that we found the correct minima
        self.assertEqual(len(minima), 2)
        self.assertEqual(minima[0], 2)
        self.assertEqual(minima[1], 7)
        
if __name__ == "__main__":
    unittest.main()