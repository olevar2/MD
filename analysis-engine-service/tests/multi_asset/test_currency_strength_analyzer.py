import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from analysis_engine.multi_asset.currency_strength_analyzer import CurrencyStrengthAnalyzer
from analysis_engine.multi_asset.correlation_tracking_service import CorrelationTrackingService

class TestCurrencyStrengthAnalyzer(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.correlation_service = MagicMock(spec=CorrelationTrackingService)
        self.analyzer = CurrencyStrengthAnalyzer(
            base_currencies=["EUR", "GBP", "USD", "JPY"],
            quote_currencies=["USD", "EUR", "JPY", "GBP"],
            lookback_periods=20,
            correlation_service=self.correlation_service
        )

        # Create test price data
        self.price_data = self._create_test_price_data()

        # Add some test data to the strength history
        now = datetime.now()
        for currency in ["EUR", "GBP", "USD", "JPY"]:
            self.analyzer.strength_history[currency] = []
            for i in range(10):
                # Create some variation in the strength values
                strength = 0.3 + (np.random.random() - 0.5) * 0.6  # Between -0.3 and 0.9
                timestamp = now - timedelta(days=10-i)
                self.analyzer.strength_history[currency].append({
                    "timestamp": timestamp,
                    "strength": strength
                })

    def _create_test_price_data(self):
        """Create test price data for multiple currency pairs"""
        # Create a date range
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

        # Create price data for different pairs
        price_data = {}

        # EUR/USD - uptrend
        price_data["EUR/USD"] = pd.DataFrame({
            'open': np.linspace(1.05, 1.15, 100) + np.random.normal(0, 0.005, 100),
            'high': np.linspace(1.06, 1.16, 100) + np.random.normal(0, 0.005, 100),
            'low': np.linspace(1.04, 1.14, 100) + np.random.normal(0, 0.005, 100),
            'close': np.linspace(1.05, 1.15, 100) + np.random.normal(0, 0.003, 100)
        }, index=dates)

        # GBP/USD - similar to EUR/USD (positive correlation)
        price_data["GBP/USD"] = pd.DataFrame({
            'open': np.linspace(1.25, 1.35, 100) + np.random.normal(0, 0.005, 100),
            'high': np.linspace(1.26, 1.36, 100) + np.random.normal(0, 0.005, 100),
            'low': np.linspace(1.24, 1.34, 100) + np.random.normal(0, 0.005, 100),
            'close': np.linspace(1.25, 1.35, 100) + np.random.normal(0, 0.003, 100)
        }, index=dates)

        # USD/JPY - downtrend (negative correlation with EUR/USD)
        price_data["USD/JPY"] = pd.DataFrame({
            'open': np.linspace(150, 140, 100) + np.random.normal(0, 0.5, 100),
            'high': np.linspace(151, 141, 100) + np.random.normal(0, 0.5, 100),
            'low': np.linspace(149, 139, 100) + np.random.normal(0, 0.5, 100),
            'close': np.linspace(150, 140, 100) + np.random.normal(0, 0.3, 100)
        }, index=dates)

        # EUR/JPY - sideways
        price_data["EUR/JPY"] = pd.DataFrame({
            'open': np.linspace(160, 160, 100) + np.random.normal(0, 1.0, 100),
            'high': np.linspace(161, 161, 100) + np.random.normal(0, 1.0, 100),
            'low': np.linspace(159, 159, 100) + np.random.normal(0, 1.0, 100),
            'close': np.linspace(160, 160, 100) + np.random.normal(0, 0.7, 100)
        }, index=dates)

        return price_data

    def test_initialization(self):
        """Test service initialization"""
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(len(self.analyzer.base_currencies), 4)
        self.assertEqual(len(self.analyzer.quote_currencies), 4)
        self.assertEqual(self.analyzer.lookback_periods, 20)

    def test_update_strength_from_pair_performance(self):
        """Test updating strength based on pair performance."""
        # Simulate EURUSD going up (EUR strong, USD weak)
        self.analyzer.update_strength_from_pair_performance('EURUSD', 0.01) # 1% increase
        self.assertGreater(self.analyzer.strength_scores['EUR'], 0)
        self.assertLess(self.analyzer.strength_scores['USD'], 0)
        self.assertEqual(self.analyzer.strength_scores['JPY'], 0) # Unaffected

        # Simulate GBPJPY going down (GBP weak, JPY strong)
        self.analyzer.update_strength_from_pair_performance('GBPJPY', -0.005) # 0.5% decrease
        self.assertLess(self.analyzer.strength_scores['GBP'], 0)
        self.assertGreater(self.analyzer.strength_scores['JPY'], 0)

    def test_calculate_relative_strength(self):
        """Test calculating relative strength scores."""
        # Simulate some updates
        self.analyzer.update_strength_from_pair_performance('EURUSD', 0.01)
        self.analyzer.update_strength_from_pair_performance('USDJPY', 0.005)
        self.analyzer.update_strength_from_pair_performance('EURGBP', 0.002)

        relative_strengths = self.analyzer.calculate_relative_strength()

        # Check basic properties
        self.assertEqual(len(relative_strengths), 4)
        self.assertTrue('EUR' in relative_strengths)
        self.assertTrue('USD' in relative_strengths)
        self.assertTrue('JPY' in relative_strengths)
        self.assertTrue('GBP' in relative_strengths)

        # EUR should likely be strongest, GBP weakest based on inputs
        self.assertGreater(relative_strengths['EUR'], relative_strengths['USD'])
        self.assertGreater(relative_strengths['EUR'], relative_strengths['JPY'])
        self.assertGreater(relative_strengths['EUR'], relative_strengths['GBP'])
        self.assertLess(relative_strengths['GBP'], relative_strengths['USD'])
        self.assertLess(relative_strengths['GBP'], relative_strengths['JPY'])


    def test_get_strongest_weakest(self):
        """Test identifying strongest and weakest currencies."""
        self.analyzer.strength_scores = {'USD': -0.5, 'EUR': 1.0, 'JPY': 0.2, 'GBP': -0.8}
        strongest, weakest = self.analyzer.get_strongest_weakest()
        self.assertEqual(strongest, 'EUR')
        self.assertEqual(weakest, 'GBP')

    def test_compute_divergence_signals(self):
        """Test computing divergence and convergence signals"""
        # Calculate currency strength first
        with patch.object(self.analyzer, 'calculate_currency_strength') as mock_calc:
            # Mock the currency strength calculation
            mock_calc.return_value = {
                'EUR': 0.8,
                'GBP': 0.5,
                'USD': -0.3,
                'JPY': -0.6
            }

            # Call the method
            results = self.analyzer.compute_divergence_signals(self.price_data, lookback_periods=20)

            # Check the structure of the results
            self.assertIsInstance(results, dict)
            self.assertIn('currency_divergence', results)
            self.assertIn('pair_divergence', results)
            self.assertIn('cross_pair_divergence', results)
            self.assertIn('basket_divergence', results)
            self.assertIn('timestamp', results)

            # Check that the method called calculate_currency_strength
            mock_calc.assert_called_once_with(self.price_data)

    def test_find_currency_divergence(self):
        """Test finding currencies with significant divergence"""
        # Set up test data in strength history
        now = datetime.now()
        self.analyzer.strength_history = {
            'EUR': [
                {'timestamp': now - timedelta(days=5), 'strength': 0.2},
                {'timestamp': now - timedelta(days=4), 'strength': 0.3},
                {'timestamp': now - timedelta(days=3), 'strength': 0.4},
                {'timestamp': now - timedelta(days=2), 'strength': 0.5},
                {'timestamp': now - timedelta(days=1), 'strength': 0.6},
                {'timestamp': now, 'strength': 0.9}  # Significant increase
            ],
            'USD': [
                {'timestamp': now - timedelta(days=5), 'strength': 0.1},
                {'timestamp': now - timedelta(days=4), 'strength': 0.1},
                {'timestamp': now - timedelta(days=3), 'strength': 0.0},
                {'timestamp': now - timedelta(days=2), 'strength': 0.0},
                {'timestamp': now - timedelta(days=1), 'strength': -0.1},
                {'timestamp': now, 'strength': -0.5}  # Significant decrease
            ],
            'JPY': [
                {'timestamp': now - timedelta(days=5), 'strength': -0.2},
                {'timestamp': now - timedelta(days=4), 'strength': -0.2},
                {'timestamp': now - timedelta(days=3), 'strength': -0.1},
                {'timestamp': now - timedelta(days=2), 'strength': -0.1},
                {'timestamp': now - timedelta(days=1), 'strength': 0.0},
                {'timestamp': now, 'strength': 0.1}  # Small change
            ],
            'GBP': [
                {'timestamp': now - timedelta(days=5), 'strength': 0.3},
                {'timestamp': now - timedelta(days=4), 'strength': 0.3},
                {'timestamp': now - timedelta(days=3), 'strength': 0.2},
                {'timestamp': now - timedelta(days=2), 'strength': 0.2},
                {'timestamp': now - timedelta(days=1), 'strength': 0.2},
                {'timestamp': now, 'strength': 0.1}  # Small change
            ]
        }

        # Call the method
        divergences = self.analyzer.find_currency_divergence(threshold=0.3)

        # Check the results
        self.assertEqual(len(divergences), 2)  # EUR and USD should have significant divergence

        # Check that EUR and USD are in the results
        currencies = [d['currency'] for d in divergences]
        self.assertIn('EUR', currencies)
        self.assertIn('USD', currencies)

        # Check that EUR is strengthening and USD is weakening
        for div in divergences:
            if div['currency'] == 'EUR':
                self.assertEqual(div['direction'], 'strengthening')
            elif div['currency'] == 'USD':
                self.assertEqual(div['direction'], 'weakening')

if __name__ == '__main__':
    unittest.main()
