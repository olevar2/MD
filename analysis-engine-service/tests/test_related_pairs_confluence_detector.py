"""
Test related pairs confluence detector module.

This module provides functionality for...
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from analysis_engine.multi_asset.related_pairs_confluence_detector import RelatedPairsConfluenceDetector
from analysis_engine.multi_asset.correlation_tracking_service import CorrelationTrackingService
from analysis_engine.multi_asset.currency_strength_analyzer import CurrencyStrengthAnalyzer

class TestRelatedPairsConfluenceDetector(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        # Keep the original config for backward compatibility
        related_pairs_config = {
            'EURUSD': ['USDCHF', 'GBPUSD', 'EURJPY', 'USDJPY']
        }

        # Add new services for enhanced functionality
        self.correlation_service = MagicMock(spec=CorrelationTrackingService)
        self.currency_strength_analyzer = MagicMock(spec=CurrencyStrengthAnalyzer)

        self.detector = RelatedPairsConfluenceDetector(
            related_pairs_config=related_pairs_config,
            correlation_service=self.correlation_service,
            currency_strength_analyzer=self.currency_strength_analyzer,
            lookback_periods=20
        )

        # Create test price data
        self.price_data = self._create_test_price_data()

        # Mock the _extract_currencies_from_pairs method
        self.detector._extract_currencies_from_pairs = MagicMock(return_value={
            'EUR/USD': ('EUR', 'USD'),
            'GBP/USD': ('GBP', 'USD'),
            'USD/JPY': ('USD', 'JPY'),
            'EUR/JPY': ('EUR', 'JPY')
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
        self.assertIsNotNone(self.detector)
        self.assertEqual(list(self.detector.related_pairs_config.keys()), ['EURUSD'])
        self.assertEqual(self.detector.lookback_periods, 20)
        self.assertIs(self.detector.correlation_service, self.correlation_service)
        self.assertIs(self.detector.currency_strength_analyzer, self.currency_strength_analyzer)

    def test_find_confluence_no_signals(self):
        """Test finding confluence when no related signals are provided."""
        primary_signal = {'pair': 'EURUSD', 'direction': 'BUY', 'confidence': 0.7}
        related_signals = {}
        confluence_score, supporting_signals = self.detector.find_confluence(primary_signal, related_signals)
        self.assertEqual(confluence_score, 0) # No related signals, no confluence
        self.assertEqual(len(supporting_signals), 0)

    def test_find_confluence_supporting_signals(self):
        """Test finding confluence with supporting related signals."""
        primary_signal = {'pair': 'EURUSD', 'direction': 'BUY', 'confidence': 0.7}
        # USDCHF down supports EURUSD up
        # GBPUSD up supports EURUSD up (positive correlation assumed)
        # EURJPY up supports EURUSD up
        related_signals = {
            'USDCHF': {'pair': 'USDCHF', 'direction': 'SELL', 'confidence': 0.6},
            'GBPUSD': {'pair': 'GBPUSD', 'direction': 'BUY', 'confidence': 0.8},
            'EURJPY': {'pair': 'EURJPY', 'direction': 'BUY', 'confidence': 0.5},
            'USDJPY': {'pair': 'USDJPY', 'direction': 'BUY', 'confidence': 0.4}, # Doesn't directly support/oppose EURUSD BUY strongly
        }
        # Note: Actual implementation needs correlation logic (positive/negative)
        # Assuming simple logic for this test:
        # - USDCHF SELL supports EURUSD BUY
        # - GBPUSD BUY supports EURUSD BUY
        # - EURJPY BUY supports EURUSD BUY
        confluence_score, supporting_signals = self.detector.find_confluence(primary_signal, related_signals)

        # Expecting positive confluence score based on supporting signals
        self.assertGreater(confluence_score, 0)
        # Expecting 3 supporting signals based on simple logic above
        self.assertEqual(len(supporting_signals), 3)
        self.assertTrue('USDCHF' in supporting_signals)
        self.assertTrue('GBPUSD' in supporting_signals)
        self.assertTrue('EURJPY' in supporting_signals)

    def test_find_confluence_opposing_signals(self):
        """Test finding confluence with opposing related signals."""
        primary_signal = {'pair': 'EURUSD', 'direction': 'BUY', 'confidence': 0.7}
        # USDCHF BUY opposes EURUSD BUY
        # GBPUSD SELL opposes EURUSD BUY
        related_signals = {
            'USDCHF': {'pair': 'USDCHF', 'direction': 'BUY', 'confidence': 0.6},
            'GBPUSD': {'pair': 'GBPUSD', 'direction': 'SELL', 'confidence': 0.8},
        }
        confluence_score, supporting_signals = self.detector.find_confluence(primary_signal, related_signals)

        # Expecting negative confluence score (divergence)
        self.assertLess(confluence_score, 0)
        self.assertEqual(len(supporting_signals), 0) # No supporting signals

    # Tests for the enhanced detect_confluence method
    def test_detect_confluence_with_no_data(self):
        """Test detect_confluence with no price data"""
        result = self.detector.detect_confluence(
            symbol="EUR/USD",
            price_data={},
            signal_type="trend",
            signal_direction="bullish"
        )

        self.assertIn("error", result)

    def test_detect_confluence_with_no_related_pairs(self):
        """Test detect_confluence with no related pairs"""
        result = self.detector.detect_confluence(
            symbol="EUR/USD",
            price_data=self.price_data,
            signal_type="trend",
            signal_direction="bullish",
            related_pairs={}
        )

        self.assertEqual(result["related_pairs_count"], 0)
        self.assertEqual(result["confluence_score"], 0.0)

    @patch('analysis_engine.multi_asset.related_pairs_confluence_detector.RelatedPairsConfluenceDetector._detect_signal')
    def test_detect_confluence_with_related_pairs(self, mock_detect_signal):
        """Test detect_confluence with related pairs"""
        # Mock the _detect_signal method to return predefined signals
        mock_detect_signal.side_effect = lambda df, signal_type, lookback: {
            "type": signal_type,
            "direction": "bullish" if df.equals(self.price_data["EUR/USD"]) or df.equals(self.price_data["GBP/USD"]) else "bearish",
            "strength": 0.8 if df.equals(self.price_data["EUR/USD"]) or df.equals(self.price_data["GBP/USD"]) else 0.6
        }

        # Mock currency strength data
        self.currency_strength_analyzer.calculate_currency_strength.return_value = {
            "EUR": 0.7,
            "USD": -0.3,
            "GBP": 0.5,
            "JPY": 0.1
        }

        # Call the method with related pairs
        result = self.detector.detect_confluence(
            symbol="EUR/USD",
            price_data=self.price_data,
            signal_type="trend",
            signal_direction="bullish",
            related_pairs={
                "GBP/USD": 0.85,  # Highly correlated, should confirm
                "USD/JPY": -0.7,   # Negatively correlated, should confirm (opposite direction)
                "EUR/JPY": 0.3     # Weakly correlated
            },
            use_currency_strength=True,
            min_confirmation_strength=0.5
        )

        # Check the result structure
        self.assertEqual(result["symbol"], "EUR/USD")
        self.assertEqual(result["signal_type"], "trend")
        self.assertEqual(result["signal_direction"], "bullish")
        self.assertEqual(result["related_pairs_count"], 3)

        # Should have confirmations (GBP/USD and USD/JPY)
        self.assertGreaterEqual(result["confirmation_count"], 1)

        # Confluence score should be positive
        self.assertGreater(result["confluence_score"], 0.5)

    def test_extract_currencies_from_pairs(self):
        """Test the _extract_currencies_from_pairs method"""
        # Restore the original method for this test
        self.detector._extract_currencies_from_pairs = RelatedPairsConfluenceDetector._extract_currencies_from_pairs.__get__(self.detector, RelatedPairsConfluenceDetector)

        # Test with various formats
        pairs = ["EUR/USD", "GBP_JPY", "AUDUSD", "NZDJPY"]
        result = self.detector._extract_currencies_from_pairs(pairs)

        # Check the results
        self.assertEqual(len(result), 4)
        self.assertEqual(result["EUR/USD"], ("EUR", "USD"))
        self.assertEqual(result["GBP_JPY"], ("GBP", "JPY"))
        self.assertEqual(result["AUDUSD"], ("AUD", "USD"))
        self.assertEqual(result["NZDJPY"], ("NZD", "JPY"))

if __name__ == '__main__':
    unittest.main()
