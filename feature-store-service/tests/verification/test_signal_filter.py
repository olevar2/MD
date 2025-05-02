"""
Tests for the Signal Filtering System.
"""
import unittest
from datetime import datetime
import pandas as pd
import numpy as np
from feature_store_service.verification.signal_filter import (
    SignalFilter,
    SignalType,
    SignalConfidence,
    FilteredSignal
)

class TestSignalFilter(unittest.TestCase):
    """Test suite for the SignalFilter"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.filter = SignalFilter()
        
        # Create test data
        self.test_price_data = pd.Series([
            100.0, 101.0, 102.0, 150.0, 103.0  # 150.0 is an obvious outlier
        ])
        
        self.test_volume_data = pd.Series([
            1000, 1100, 1200, 5000, 1250  # 5000 is an obvious outlier
        ])
        
        self.test_pattern_data = {
            'type': 'head_and_shoulders',
            'confidence': 0.8,
            'completion': 0.95,
            'quality': 0.85,
            'points': {
                'left_shoulder': 100.0,
                'head': 110.0,
                'right_shoulder': 102.0,
                'neckline': 98.0
            }
        }

    def test_price_signal_filtering(self):
        """Test price signal filtering."""
        context = {'volatility': 0.1}
        
        # Test single price filtering
        result = self.filter.filter_signal(
            SignalType.PRICE,
            150.0,
            context={'recent_prices': [100.0, 101.0, 102.0]}
        )
        self.assertEqual(result.confidence, SignalConfidence.LOW)
        self.assertNotEqual(result.filtered_value, 150.0)
        
        # Test price series filtering
        result = self.filter.filter_signal(
            SignalType.PRICE,
            self.test_price_data,
            context=context
        )
        self.assertIsInstance(result.filtered_value, pd.Series)
        self.assertNotEqual(result.filtered_value[3], 150.0)
        self.assertEqual(result.signal_type, SignalType.PRICE)

    def test_volume_signal_filtering(self):
        """Test volume signal filtering."""
        # Test volume series filtering
        result = self.filter.filter_signal(
            SignalType.VOLUME,
            self.test_volume_data
        )
        self.assertIsInstance(result.filtered_value, pd.Series)
        self.assertNotEqual(result.filtered_value[3], 5000)
        self.assertEqual(result.signal_type, SignalType.VOLUME)

    def test_pattern_signal_filtering(self):
        """Test pattern signal filtering."""
        result = self.filter.filter_signal(
            SignalType.PATTERN,
            self.test_pattern_data
        )
        self.assertEqual(result.signal_type, SignalType.PATTERN)
        self.assertEqual(result.confidence, SignalConfidence.HIGH)
        self.assertIn('pattern_type', result.metadata)
        self.assertIn('completion', result.metadata)
        self.assertIn('quality', result.metadata)

    def test_indicator_signal_filtering(self):
        """Test indicator signal filtering."""
        indicator_data = pd.Series([
            0.7, 0.72, 0.95, 0.73, 0.74  # 0.95 is an outlier
        ])
        
        result = self.filter.filter_signal(
            SignalType.INDICATOR,
            indicator_data,
            context={'indicator_type': 'RSI'}
        )
        self.assertIsInstance(result.filtered_value, pd.Series)
        self.assertNotEqual(result.filtered_value[2], 0.95)
        self.assertEqual(result.signal_type, SignalType.INDICATOR)

    def test_market_signal_filtering(self):
        """Test market signal filtering."""
        market_data = {
            'volatility': 0.5,
            'trend': 0.8,
            'volume': 1000000
        }
        
        result = self.filter.filter_signal(
            SignalType.MARKET,
            market_data,
            context={'market_type': 'forex'}
        )
        self.assertIsInstance(result.filtered_value, dict)
        self.assertEqual(result.signal_type, SignalType.MARKET)

    def test_composite_signal_filtering(self):
        """Test composite signal filtering."""
        composite_data = {
            'PRICE': self.test_price_data,
            'VOLUME': self.test_volume_data,
            'MARKET': {
                'volatility': 0.5,
                'trend': 0.8
            }
        }
        
        result = self.filter.filter_signal(
            SignalType.COMPOSITE,
            composite_data
        )
        self.assertIsInstance(result.filtered_value, dict)
        self.assertEqual(result.signal_type, SignalType.COMPOSITE)
        self.assertIn('PRICE', result.filtered_value)
        self.assertIn('VOLUME', result.filtered_value)
        self.assertIn('MARKET', result.filtered_value)

    def test_invalid_signal_handling(self):
        """Test handling of invalid signals."""
        # Test with invalid signal type
        result = self.filter.filter_signal(
            'INVALID_TYPE',
            100.0
        )
        self.assertEqual(result.confidence, SignalConfidence.INVALID)
        self.assertIsNone(result.filtered_value)

        # Test with invalid data
        result = self.filter.filter_signal(
            SignalType.PRICE,
            None
        )
        self.assertEqual(result.confidence, SignalConfidence.INVALID)

    def test_confidence_assessment(self):
        """Test signal confidence assessment."""
        # Test high confidence case
        result = self.filter.filter_signal(
            SignalType.PRICE,
            pd.Series([100.0, 101.0, 102.0, 103.0]),
            context={'volatility': 0.1}
        )
        self.assertEqual(result.confidence, SignalConfidence.HIGH)

        # Test low confidence case
        result = self.filter.filter_signal(
            SignalType.PRICE,
            pd.Series([100.0, 150.0, 102.0, 160.0]),
            context={'volatility': 0.5}
        )
        self.assertEqual(result.confidence, SignalConfidence.LOW)

    def test_filter_summary(self):
        """Test filter summary generation."""
        # Generate some filtered signals
        self.filter.filter_signal(SignalType.PRICE, self.test_price_data)
        self.filter.filter_signal(SignalType.VOLUME, self.test_volume_data)
        self.filter.filter_signal(SignalType.PATTERN, self.test_pattern_data)
        
        summary = self.filter.get_filter_summary()
        self.assertIn('total_signals', summary)
        self.assertIn('by_type', summary)
        self.assertIn('by_confidence', summary)
        self.assertEqual(summary['total_signals'], 3)
        self.assertIn(SignalType.PRICE.value, summary['by_type'])
        self.assertIn(SignalType.VOLUME.value, summary['by_type'])
        self.assertIn(SignalType.PATTERN.value, summary['by_type'])

if __name__ == '__main__':
    unittest.main()
