"""
Test module for advanced pattern recognition.

This module tests the advanced pattern recognition capabilities.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from feature_store_service.indicators.advanced_patterns import (
    AdvancedPatternFacade,
    RenkoPatternRecognizer,
    IchimokuPatternRecognizer
)


class TestAdvancedPatterns(unittest.TestCase):
    """Test case for advanced pattern recognition."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample OHLCV data
        dates = [datetime.now() - timedelta(days=i) for i in range(200)]
        dates.reverse()
        
        # Create a simple uptrend followed by a downtrend
        close_prices = []
        for i in range(200):
            if i < 100:
                # Uptrend
                close_prices.append(100 + i * 0.5 + np.random.normal(0, 1))
            else:
                # Downtrend
                close_prices.append(150 - (i - 100) * 0.3 + np.random.normal(0, 1))
        
        # Create high and low prices
        high_prices = [price + np.random.uniform(0.1, 0.5) for price in close_prices]
        low_prices = [price - np.random.uniform(0.1, 0.5) for price in close_prices]
        open_prices = [prev_close + np.random.normal(0, 0.2) for prev_close in [close_prices[0]] + close_prices[:-1]]
        volume = [1000 + np.random.randint(0, 500) for _ in range(200)]
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        }, index=dates)
    
    def test_advanced_pattern_facade(self):
        """Test the AdvancedPatternFacade."""
        # Initialize the facade
        facade = AdvancedPatternFacade(
            pattern_types=None,  # Use all patterns
            lookback_period=50,
            sensitivity=0.75
        )
        
        # Calculate patterns
        result = facade.calculate(self.data)
        
        # Check that the result contains the expected columns
        self.assertIn('has_advanced_pattern', result.columns)
        
        # Check that at least some patterns were detected
        self.assertTrue(result['has_advanced_pattern'].sum() > 0)
        
        # Test the find_patterns method
        patterns = facade.find_patterns(self.data)
        
        # Check that patterns were found
        self.assertTrue(len(patterns) > 0)
        
        # Check that the patterns dictionary contains expected keys
        for pattern_type in facade.get_supported_patterns():
            self.assertIn(pattern_type, patterns)
    
    def test_renko_pattern_recognizer(self):
        """Test the RenkoPatternRecognizer."""
        # Initialize the recognizer
        recognizer = RenkoPatternRecognizer(
            brick_size=None,  # Auto-calculate
            brick_method='atr',
            atr_period=14,
            min_trend_length=3,
            min_consolidation_length=4,
            pattern_types=None,  # Use all patterns
            lookback_period=50,
            sensitivity=0.75
        )
        
        # Calculate patterns
        result = recognizer.calculate(self.data)
        
        # Check that the result contains Renko-specific columns
        self.assertIn('renko_brick_direction', result.columns)
        self.assertIn('renko_brick_open', result.columns)
        self.assertIn('renko_brick_close', result.columns)
        
        # Check for pattern columns
        pattern_cols = [col for col in result.columns if col.startswith('pattern_renko_')]
        self.assertTrue(len(pattern_cols) > 0)
        
        # Test the find_patterns method
        patterns = recognizer.find_patterns(self.data)
        
        # Check that the patterns dictionary contains expected keys
        for pattern_type in ['renko_reversal', 'renko_breakout', 'renko_double_top', 'renko_double_bottom']:
            self.assertIn(pattern_type, patterns)
    
    def test_ichimoku_pattern_recognizer(self):
        """Test the IchimokuPatternRecognizer."""
        # Initialize the recognizer
        recognizer = IchimokuPatternRecognizer(
            tenkan_period=9,
            kijun_period=26,
            senkou_b_period=52,
            displacement=26,
            pattern_types=None,  # Use all patterns
            lookback_period=100,
            sensitivity=0.75
        )
        
        # Calculate patterns
        result = recognizer.calculate(self.data)
        
        # Check that the result contains Ichimoku-specific columns
        self.assertIn('ichimoku_tenkan', result.columns)
        self.assertIn('ichimoku_kijun', result.columns)
        self.assertIn('ichimoku_senkou_a', result.columns)
        self.assertIn('ichimoku_senkou_b', result.columns)
        
        # Check for pattern columns
        pattern_cols = [col for col in result.columns if col.startswith('pattern_ichimoku_')]
        self.assertTrue(len(pattern_cols) > 0)
        
        # Test the find_patterns method
        patterns = recognizer.find_patterns(self.data)
        
        # Check that the patterns dictionary contains expected keys
        for pattern_type in ['ichimoku_tk_cross', 'ichimoku_kumo_breakout', 'ichimoku_kumo_twist', 'ichimoku_chikou_cross']:
            self.assertIn(pattern_type, patterns)


if __name__ == '__main__':
    unittest.main()