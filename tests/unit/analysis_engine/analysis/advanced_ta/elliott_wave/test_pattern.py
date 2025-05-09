"""
Unit tests for Elliott Wave pattern module.
"""

import unittest
from datetime import datetime
from analysis_engine.analysis.advanced_ta.base import ConfidenceLevel, MarketDirection
from analysis_engine.analysis.advanced_ta.elliott_wave.models import (
    WaveType, WavePosition, WaveDegree
)
from analysis_engine.analysis.advanced_ta.elliott_wave.pattern import ElliottWavePattern


class TestElliottWavePattern(unittest.TestCase):
    """Test cases for ElliottWavePattern class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.now = datetime.now()
        self.pattern = ElliottWavePattern(
            pattern_name="Test Impulse Wave",
            pattern_type=WaveType.IMPULSE,
            wave_degree=WaveDegree.INTERMEDIATE,
            confidence=ConfidenceLevel.MEDIUM,
            direction=MarketDirection.BULLISH,
            start_time=self.now,
            end_time=self.now,
            start_price=100.0,
            end_price=110.0,
            waves={
                WavePosition.ONE: (self.now, 102.0),
                WavePosition.TWO: (self.now, 101.0),
                WavePosition.THREE: (self.now, 106.0),
                WavePosition.FOUR: (self.now, 105.0),
                WavePosition.FIVE: (self.now, 110.0)
            },
            fibonacci_levels={
                "wave3_1.618": 105.0,
                "wave5_0.618": 108.0
            },
            completion_percentage=100.0
        )
    
    def test_init(self):
        """Test initialization of ElliottWavePattern."""
        self.assertEqual(self.pattern.pattern_name, "Test Impulse Wave")
        self.assertEqual(self.pattern.pattern_type, WaveType.IMPULSE)
        self.assertEqual(self.pattern.wave_degree, WaveDegree.INTERMEDIATE)
        self.assertEqual(self.pattern.confidence, ConfidenceLevel.MEDIUM)
        self.assertEqual(self.pattern.direction, MarketDirection.BULLISH)
        self.assertEqual(self.pattern.start_time, self.now)
        self.assertEqual(self.pattern.end_time, self.now)
        self.assertEqual(self.pattern.start_price, 100.0)
        self.assertEqual(self.pattern.end_price, 110.0)
        self.assertEqual(len(self.pattern.waves), 5)
        self.assertEqual(len(self.pattern.fibonacci_levels), 2)
        self.assertEqual(self.pattern.completion_percentage, 100.0)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        pattern_dict = self.pattern.to_dict()
        
        self.assertEqual(pattern_dict["pattern_name"], "Test Impulse Wave")
        self.assertEqual(pattern_dict["pattern_type"], "impulse")
        self.assertEqual(pattern_dict["wave_type"], "impulse")
        self.assertEqual(pattern_dict["wave_degree"], "Intermediate")
        self.assertEqual(pattern_dict["confidence"], "medium")
        self.assertEqual(pattern_dict["direction"], "bullish")
        self.assertEqual(pattern_dict["completion_percentage"], 100.0)
        
        # Check waves dictionary
        self.assertIn("waves", pattern_dict)
        self.assertIn("1", pattern_dict["waves"])
        self.assertIn("2", pattern_dict["waves"])
        self.assertIn("3", pattern_dict["waves"])
        self.assertIn("4", pattern_dict["waves"])
        self.assertIn("5", pattern_dict["waves"])
        
        # Check fibonacci levels
        self.assertIn("fibonacci_levels", pattern_dict)
        self.assertIn("wave3_1.618", pattern_dict["fibonacci_levels"])
        self.assertIn("wave5_0.618", pattern_dict["fibonacci_levels"])


if __name__ == "__main__":
    unittest.main()