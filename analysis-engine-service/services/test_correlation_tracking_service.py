"""
Test correlation tracking service module.

This module provides functionality for...
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from analysis_engine.multi_asset.correlation_tracking_service import CorrelationTrackingService
from analysis_engine.multi_asset.asset_registry import AssetRegistry


class TestCorrelationTrackingService(unittest.TestCase):
    """Test cases for the CorrelationTrackingService"""

    def setUp(self):
        """Set up test fixtures"""
        self.asset_registry = MagicMock(spec=AssetRegistry)
        self.service = CorrelationTrackingService(asset_registry=self.asset_registry)
        
        # Add some test data to the correlation history
        test_pairs = [('EUR/USD', 'GBP/USD'), ('EUR/USD', 'USD/JPY'), ('GBP/USD', 'AUD/USD')]
        now = datetime.now()
        
        for pair in test_pairs:
            self.service.correlation_history[pair] = []
            # Add 20 data points with varying correlation values
            for i in range(20):
                # Create some variation in the correlation values
                if i < 10:
                    corr = 0.7 + (np.random.random() - 0.5) * 0.2  # Around 0.7
                else:
                    corr = 0.5 + (np.random.random() - 0.5) * 0.2  # Around 0.5
                
                timestamp = now - timedelta(days=20-i)
                self.service.correlation_history[pair].append((timestamp, corr))

    def test_get_correlation_stability(self):
        """Test the get_correlation_stability method"""
        # Test with a pair that has history
        stability = self.service.get_correlation_stability('EUR/USD', 'GBP/USD', lookback_days=30)
        
        # Stability should be a float
        self.assertIsInstance(stability, float)
        
        # Stability should be positive (standard deviation)
        self.assertGreaterEqual(stability, 0.0)
        
        # Test with a pair that has no history
        stability = self.service.get_correlation_stability('USD/CAD', 'NZD/USD', lookback_days=30)
        
        # Should return None for pairs with no history
        self.assertIsNone(stability)
        
        # Test with a shorter lookback period
        stability_short = self.service.get_correlation_stability('EUR/USD', 'GBP/USD', lookback_days=5)
        
        # Should return a value even with a shorter lookback
        self.assertIsInstance(stability_short, float)

    @patch('analysis_engine.multi_asset.correlation_tracking_service.CorrelationTrackingService._get_price_data')
    async def test_calculate_correlations(self, mock_get_price_data):
        """Test the calculate_correlations method"""
        # Mock price data
        mock_data = {
            'EUR/USD': pd.DataFrame({
                'close': np.random.random(100)
            }),
            'GBP/USD': pd.DataFrame({
                'close': np.random.random(100)
            }),
            'USD/JPY': pd.DataFrame({
                'close': np.random.random(100)
            })
        }
        
        # Configure the mock to return the test data
        mock_get_price_data.side_effect = lambda symbol, days: mock_data.get(symbol)
        
        # Call the method
        result = await self.service.calculate_correlations(['EUR/USD', 'GBP/USD', 'USD/JPY'])
        
        # Check the result structure
        self.assertIsInstance(result, dict)
        self.assertIn('EUR/USD', result)
        self.assertIn('GBP/USD', result['EUR/USD'])
        
        # Correlation should be between -1 and 1
        self.assertGreaterEqual(result['EUR/USD']['GBP/USD'], -1.0)
        self.assertLessEqual(result['EUR/USD']['GBP/USD'], 1.0)

    def test_correlation_history_update(self):
        """Test that correlation history is properly updated"""
        # Create a mock correlation matrix
        correlation_matrix = {
            'EUR/USD': {'GBP/USD': 0.8, 'USD/JPY': -0.6},
            'GBP/USD': {'EUR/USD': 0.8, 'USD/JPY': -0.5},
            'USD/JPY': {'EUR/USD': -0.6, 'GBP/USD': -0.5}
        }
        
        # Create a mock cache key
        key = frozenset(['EUR/USD', 'GBP/USD', 'USD/JPY'])
        
        # Update the cache and history
        now = datetime.now()
        self.service.correlation_cache[key] = (now, correlation_matrix)
        
        # Call get_correlation_matrix to trigger history update
        result = self.service.get_correlation_matrix(['EUR/USD', 'GBP/USD', 'USD/JPY'], use_cached=True)
        
        # Check that history was updated
        pair = tuple(sorted(['EUR/USD', 'GBP/USD']))
        self.assertIn(pair, self.service.correlation_history)
        
        # The last entry should have the timestamp close to now
        last_entry = self.service.correlation_history[pair][-1]
        self.assertAlmostEqual(last_entry[0].timestamp(), now.timestamp(), delta=1)  # Within 1 second
        self.assertEqual(last_entry[1], 0.8)  # The correlation value


if __name__ == '__main__':
    unittest.main()
