"""
Test market regime detector module.

This module provides functionality for...
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from analysis_engine.services.market_regime_detector import MarketRegimeDetector, MarketRegime


class TestMarketRegimeDetector(unittest.TestCase):
    """Test cases for the MarketRegimeDetector"""

    def setUp(self):
        """Set up test fixtures"""
        self.detector = MarketRegimeDetector()
        
        # Create test price data
        self.price_data = self._create_test_price_data()

    def _create_test_price_data(self):
        """Create test price data for different market regimes"""
        # Create a date range
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        
        # Create price data for different regimes
        
        # Trending market (first 50 days)
        trend_data = pd.DataFrame({
            'open': np.linspace(100, 150, 50) + np.random.normal(0, 1, 50),
            'high': np.linspace(102, 152, 50) + np.random.normal(0, 1, 50),
            'low': np.linspace(98, 148, 50) + np.random.normal(0, 1, 50),
            'close': np.linspace(100, 150, 50) + np.random.normal(0, 0.5, 50),
            'volume': np.random.randint(1000, 5000, 50)
        }, index=dates[:50])
        
        # Ranging market (next 50 days)
        range_data = pd.DataFrame({
            'open': np.linspace(150, 150, 50) + np.random.normal(0, 3, 50),
            'high': np.linspace(155, 155, 50) + np.random.normal(0, 3, 50),
            'low': np.linspace(145, 145, 50) + np.random.normal(0, 3, 50),
            'close': np.linspace(150, 150, 50) + np.random.normal(0, 2, 50),
            'volume': np.random.randint(800, 3000, 50)
        }, index=dates[50:100])
        
        # Breakout market (next 50 days)
        breakout_data = pd.DataFrame({
            'open': np.concatenate([np.linspace(150, 150, 30), np.linspace(150, 180, 20)]) + np.random.normal(0, 2, 50),
            'high': np.concatenate([np.linspace(155, 155, 30), np.linspace(155, 185, 20)]) + np.random.normal(0, 2, 50),
            'low': np.concatenate([np.linspace(145, 145, 30), np.linspace(145, 175, 20)]) + np.random.normal(0, 2, 50),
            'close': np.concatenate([np.linspace(150, 150, 30), np.linspace(150, 180, 20)]) + np.random.normal(0, 1, 50),
            'volume': np.concatenate([np.random.randint(800, 3000, 30), np.random.randint(3000, 8000, 20)])
        }, index=dates[100:150])
        
        # Volatile market (last 50 days)
        volatile_data = pd.DataFrame({
            'open': np.linspace(180, 160, 50) + np.random.normal(0, 8, 50),
            'high': np.linspace(185, 165, 50) + np.random.normal(0, 8, 50),
            'low': np.linspace(175, 155, 50) + np.random.normal(0, 8, 50),
            'close': np.linspace(180, 160, 50) + np.random.normal(0, 6, 50),
            'volume': np.random.randint(2000, 10000, 50)
        }, index=dates[150:])
        
        # Combine all data
        return pd.concat([trend_data, range_data, breakout_data, volatile_data])

    def test_initialization(self):
        """Test detector initialization"""
        self.assertIsNotNone(self.detector)
        self.assertEqual(len(self.detector.history), 0)
        self.assertEqual(len(self.detector.transition_history), 0)

    def test_detect_regime(self):
        """Test detecting market regime"""
        # Test with trending data
        trend_result = self.detector.detect_regime(self.price_data.iloc[:50])
        self.assertEqual(trend_result["regime"], MarketRegime.TRENDING)
        
        # Test with ranging data
        range_result = self.detector.detect_regime(self.price_data.iloc[50:100])
        self.assertEqual(range_result["regime"], MarketRegime.RANGING)
        
        # Test with breakout data
        breakout_result = self.detector.detect_regime(self.price_data.iloc[100:150])
        self.assertEqual(breakout_result["regime"], MarketRegime.BREAKOUT)
        
        # Test with volatile data
        volatile_result = self.detector.detect_regime(self.price_data.iloc[150:])
        self.assertEqual(volatile_result["regime"], MarketRegime.VOLATILE)

    def test_history_tracking(self):
        """Test that history is properly tracked"""
        # Detect regimes for different segments
        self.detector.detect_regime(self.price_data.iloc[:50])
        self.detector.detect_regime(self.price_data.iloc[50:100])
        self.detector.detect_regime(self.price_data.iloc[100:150])
        
        # Check that history has been updated
        self.assertEqual(len(self.detector.history), 3)
        
        # Check that history entries have the correct structure
        for entry in self.detector.history:
            self.assertIn("timestamp", entry)
            self.assertIn("regime", entry)
            self.assertIn("confidence", entry)

    def test_transition_history_tracking(self):
        """Test that transition history is properly tracked"""
        # Detect regimes for different segments to trigger transitions
        self.detector.detect_regime(self.price_data.iloc[:50])  # TRENDING
        self.detector.detect_regime(self.price_data.iloc[50:100])  # RANGING
        self.detector.detect_regime(self.price_data.iloc[100:150])  # BREAKOUT
        
        # Check that transition history has been updated
        self.assertEqual(len(self.detector.transition_history), 2)
        
        # Check that transition entries have the correct structure
        for entry in self.detector.transition_history:
            self.assertIn("timestamp", entry)
            self.assertIn("from_regime", entry)
            self.assertIn("to_regime", entry)
            self.assertIn("from_confidence", entry)
            self.assertIn("to_confidence", entry)
            self.assertIn("transition_metrics", entry)
        
        # Check that transitions are correct
        self.assertEqual(self.detector.transition_history[0]["from_regime"], MarketRegime.TRENDING)
        self.assertEqual(self.detector.transition_history[0]["to_regime"], MarketRegime.RANGING)
        self.assertEqual(self.detector.transition_history[1]["from_regime"], MarketRegime.RANGING)
        self.assertEqual(self.detector.transition_history[1]["to_regime"], MarketRegime.BREAKOUT)

    def test_get_transition_history(self):
        """Test getting transition history"""
        # Detect regimes for different segments to trigger transitions
        self.detector.detect_regime(self.price_data.iloc[:50])  # TRENDING
        self.detector.detect_regime(self.price_data.iloc[50:100])  # RANGING
        self.detector.detect_regime(self.price_data.iloc[100:150])  # BREAKOUT
        self.detector.detect_regime(self.price_data.iloc[150:])  # VOLATILE
        
        # Get transition history
        transitions = self.detector.get_transition_history(limit=2)
        
        # Check that we get the correct number of transitions
        self.assertEqual(len(transitions), 2)
        
        # Check that we get the most recent transitions
        self.assertEqual(transitions[0]["from_regime"], MarketRegime.RANGING)
        self.assertEqual(transitions[0]["to_regime"], MarketRegime.BREAKOUT)
        self.assertEqual(transitions[1]["from_regime"], MarketRegime.BREAKOUT)
        self.assertEqual(transitions[1]["to_regime"], MarketRegime.VOLATILE)

    def test_get_transition_frequency(self):
        """Test getting transition frequency statistics"""
        # Detect regimes for different segments to trigger transitions
        self.detector.detect_regime(self.price_data.iloc[:50])  # TRENDING
        self.detector.detect_regime(self.price_data.iloc[50:100])  # RANGING
        self.detector.detect_regime(self.price_data.iloc[100:150])  # BREAKOUT
        self.detector.detect_regime(self.price_data.iloc[150:])  # VOLATILE
        
        # Get transition frequency
        stats = self.detector.get_transition_frequency(lookback_days=30)
        
        # Check the structure of the statistics
        self.assertIn("total_transitions", stats)
        self.assertIn("transitions_per_day", stats)
        self.assertIn("most_common_transition", stats)
        self.assertIn("transition_counts", stats)
        
        # Check that we have the correct number of transitions
        self.assertEqual(stats["total_transitions"], 3)
        
        # Check that transitions_per_day is calculated correctly
        self.assertAlmostEqual(stats["transitions_per_day"], 3 / 30, places=5)


if __name__ == '__main__':
    unittest.main()
