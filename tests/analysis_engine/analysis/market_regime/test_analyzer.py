"""
Unit tests for the Market Regime Analyzer.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from analysis_engine.analysis.market_regime.analyzer import MarketRegimeAnalyzer
from analysis_engine.analysis.market_regime.models import (
    RegimeType, DirectionType, VolatilityLevel, RegimeClassification
)


class TestMarketRegimeAnalyzer(unittest.TestCase):
    """Test cases for the MarketRegimeAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = MarketRegimeAnalyzer()
        self.sample_data = self._create_sample_data()
    
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
    
    def test_analyze(self):
        """Test the analyze method."""
        # Analyze the sample data
        result = self.analyzer.analyze(self.sample_data)
        
        # Check that we get a RegimeClassification object
        self.assertIsInstance(result, RegimeClassification)
        
        # Check that all required attributes are present
        self.assertIsNotNone(result.regime)
        self.assertIsNotNone(result.confidence)
        self.assertIsNotNone(result.direction)
        self.assertIsNotNone(result.volatility)
        self.assertIsNotNone(result.timestamp)
        self.assertIsNotNone(result.features)
    
    def test_analyze_cached(self):
        """Test the analyze_cached method."""
        # Call the cached method
        result = self.analyzer.analyze_cached(
            instrument='EUR/USD',
            timeframe='H1',
            price_data_key='test_key',
            timestamp='2023-01-01T00:00:00'
        )
        
        # Check that we get a dictionary
        self.assertIsInstance(result, dict)
        
        # Check that all required keys are present
        self.assertIn('regime', result)
        self.assertIn('confidence', result)
        self.assertIn('direction', result)
        self.assertIn('volatility', result)
        self.assertIn('timestamp', result)
        self.assertIn('features', result)
        
        # Call again with the same parameters to test caching
        result2 = self.analyzer.analyze_cached(
            instrument='EUR/USD',
            timeframe='H1',
            price_data_key='test_key',
            timestamp='2023-01-01T00:00:00'
        )
        
        # Results should be identical due to caching
        self.assertEqual(result, result2)
    
    def test_regime_change_notification(self):
        """Test that subscribers are notified of regime changes."""
        # Create a mock subscriber
        mock_subscriber = MagicMock()
        
        # Subscribe to regime changes
        self.analyzer.subscribe_to_regime_changes(mock_subscriber)
        
        # Analyze data to establish initial regime
        self.analyzer.analyze(self.sample_data)
        
        # Mock subscriber should not be called yet
        mock_subscriber.assert_not_called()
        
        # Create data for a different regime
        volatile_data = self.sample_data.copy()
        volatile_data['high'] = volatile_data['close'] + 5
        volatile_data['low'] = volatile_data['close'] - 5
        
        # Analyze the volatile data
        self.analyzer.analyze(volatile_data)
        
        # Mock subscriber should be called
        mock_subscriber.assert_called_once()
        
        # Unsubscribe
        self.analyzer.unsubscribe_from_regime_changes(mock_subscriber)
        
        # Reset the mock
        mock_subscriber.reset_mock()
        
        # Analyze more data
        self.analyzer.analyze(self.sample_data)
        
        # Mock subscriber should not be called after unsubscribing
        mock_subscriber.assert_not_called()
    
    def test_get_historical_regimes(self):
        """Test the get_historical_regimes method."""
        # Get historical regimes
        results = self.analyzer.get_historical_regimes(
            self.sample_data,
            window_size=20
        )
        
        # Check that we get a list of RegimeClassification objects
        self.assertIsInstance(results, list)
        self.assertTrue(all(isinstance(r, RegimeClassification) for r in results))
        
        # Check that we get the expected number of results
        expected_count = len(self.sample_data) - 20 + 1
        self.assertEqual(len(results), expected_count)
    
    def test_analyzer_coordination(self):
        """Test that the analyzer correctly coordinates detector and classifier."""
        # Create mock detector and classifier
        mock_detector = MagicMock()
        mock_classifier = MagicMock()
        
        # Set up return values
        from analysis_engine.analysis.market_regime.models import FeatureSet
        mock_features = FeatureSet(
            volatility=0.8,
            trend_strength=0.7,
            momentum=0.5,
            mean_reversion=-0.1,
            range_width=0.02
        )
        mock_detector.extract_features.return_value = mock_features
        
        mock_classification = RegimeClassification(
            regime=RegimeType.TRENDING_BULLISH,
            confidence=0.85,
            direction=DirectionType.BULLISH,
            volatility=VolatilityLevel.MEDIUM,
            timestamp=datetime.now(),
            features=mock_features.to_dict()
        )
        mock_classifier.classify.return_value = mock_classification
        
        # Replace the analyzer's components with mocks
        self.analyzer.detector = mock_detector
        self.analyzer.classifier = mock_classifier
        
        # Analyze data
        result = self.analyzer.analyze(self.sample_data)
        
        # Check that detector and classifier were called
        mock_detector.extract_features.assert_called_once_with(self.sample_data)
        mock_classifier.classify.assert_called_once()
        
        # Check that the result is the mock classification
        self.assertEqual(result, mock_classification)
    
    def test_error_handling(self):
        """Test error handling in the _check_regime_change method."""
        # Create a mock subscriber that raises an exception
        def failing_subscriber(new_classification, old_classification):
            raise Exception("Test exception")
        
        # Subscribe the failing subscriber
        self.analyzer.subscribe_to_regime_changes(failing_subscriber)
        
        # Analyze data - this should not raise an exception
        try:
            self.analyzer.analyze(self.sample_data)
            # If we get here, no exception was raised
            success = True
        except:
            success = False
        
        self.assertTrue(success)


if __name__ == '__main__':
    unittest.main()