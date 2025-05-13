"""
Test regime transition predictor module.

This module provides functionality for...
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

from analysis_engine.services.regime_transition_predictor import RegimeTransitionPredictor
from analysis_engine.services.market_regime_detector import MarketRegimeDetector, MarketRegime
from analysis_engine.multi_asset.correlation_tracking_service import CorrelationTrackingService

class TestRegimeTransitionPredictor(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures, if any."""
        self.predictor = RegimeTransitionPredictor(
            regimes=[MarketRegime.TRENDING, MarketRegime.RANGING, MarketRegime.VOLATILE],
            lookback_period=10
        )
        # Mock indicator calculation function
        self.predictor._calculate_transition_indicators = lambda data: self._mock_indicators(data)

    def _mock_indicators(self, data):
    """
     mock indicators.
    
    Args:
        data: Description of data
    
    """

        # Dummy indicators for testing
        return pd.DataFrame({
            'volatility_change': data['close'].pct_change().rolling(5).std().diff(),
            'trend_strength_change': data['close'].rolling(5).mean().diff().diff()
        }, index=data.index)

    def test_initialization(self):
        """Test service initialization."""
        self.assertIsNotNone(self.predictor)
        self.assertEqual(len(self.predictor.regimes), 3)
        self.assertEqual(self.predictor.lookback_period, 10)
        # Check transition matrix initialization
        self.assertEqual(self.predictor.transition_matrix.shape, (3, 3))
        self.assertTrue((self.predictor.transition_matrix == 0).all().all())

    def test_update_regime_history(self):
        """Test updating the regime history."""
        history = [MarketRegime.TRENDING, MarketRegime.TRENDING, MarketRegime.RANGING]
        for regime in history:
            self.predictor.update_regime_history(regime)

        self.assertEqual(len(self.predictor.regime_history), 3)
        self.assertEqual(self.predictor.regime_history[-1], MarketRegime.RANGING)

        # Test history limit
        for _ in range(15):
            self.predictor.update_regime_history(MarketRegime.VOLATILE)
        self.assertEqual(len(self.predictor.regime_history), self.predictor.lookback_period) # Should be capped

    def test_calculate_transition_probabilities(self):
        """Test calculating transition probabilities."""
        # Add history: T -> T -> R -> V -> V -> T -> R
        history = [
            MarketRegime.TRENDING, MarketRegime.TRENDING, MarketRegime.RANGING,
            MarketRegime.VOLATILE, MarketRegime.VOLATILE, MarketRegime.TRENDING,
            MarketRegime.RANGING
        ]
        for regime in history:
            self.predictor.update_regime_history(regime)

        self.predictor.calculate_transition_probabilities()
        matrix = self.predictor.get_transition_matrix()

        # Verify some expected transitions based on history
        # T -> T occurred once
        # T -> R occurred once
        # R -> V occurred once
        # V -> V occurred once
        # V -> T occurred once
        # T -> R occurred once (second time)
        # Total transitions = 6

        # Example check (T -> R probability) - occurred 2 out of 3 times starting from T
        trending_idx = self.predictor.regime_map[MarketRegime.TRENDING]
        ranging_idx = self.predictor.regime_map[MarketRegime.RANGING]
        volatile_idx = self.predictor.regime_map[MarketRegime.VOLATILE]

        # Note: Probabilities are P(to | from)
        self.assertAlmostEqual(matrix.loc[MarketRegime.TRENDING, MarketRegime.RANGING], 2/3)
        self.assertAlmostEqual(matrix.loc[MarketRegime.TRENDING, MarketRegime.TRENDING], 1/3)
        self.assertAlmostEqual(matrix.loc[MarketRegime.RANGING, MarketRegime.VOLATILE], 1/1) # Only one transition from R
        self.assertAlmostEqual(matrix.loc[MarketRegime.VOLATILE, MarketRegime.VOLATILE], 1/2)
        self.assertAlmostEqual(matrix.loc[MarketRegime.VOLATILE, MarketRegime.TRENDING], 1/2)


    def test_predict_next_regime(self):
        """Test predicting the next regime based on indicators."""
        # Create mock market data
        data = pd.DataFrame({'close': np.random.randn(20).cumsum() + 100},
                            index=pd.to_datetime(pd.date_range(start='2023-01-01', periods=20)))

        # Assume current regime is TRENDING
        self.predictor.current_regime = MarketRegime.TRENDING
        # Set some dummy probabilities
        self.predictor.transition_matrix.loc[MarketRegime.TRENDING, MarketRegime.RANGING] = 0.6
        self.predictor.transition_matrix.loc[MarketRegime.TRENDING, MarketRegime.VOLATILE] = 0.3
        self.predictor.transition_matrix.loc[MarketRegime.TRENDING, MarketRegime.TRENDING] = 0.1

        # Note: The actual prediction logic using indicators is complex and depends
        # on the specific implementation (_calculate_early_warning_score).
        # This test focuses on the structure.
        prediction = self.predictor.predict_next_regime(data)

        self.assertIsNotNone(prediction)
        self.assertTrue('predicted_regime' in prediction)
        self.assertTrue('probability' in prediction)
        self.assertTrue('warning_score' in prediction)
        # Predicted regime should be one of the defined regimes
        self.assertIn(prediction['predicted_regime'], self.predictor.regimes)

    # Tests for inter-market correlation integration
    @patch('analysis_engine.services.regime_transition_predictor.RegimeTransitionPredictor._analyze_correlated_markets')
    async def test_predict_with_inter_market_correlations(self, mock_analyze_markets):
        """Test predicting with inter-market correlations"""
        # Create a new predictor with mocked dependencies
        regime_detector = MagicMock(spec=MarketRegimeDetector)
        correlation_service = MagicMock(spec=CorrelationTrackingService)

        predictor = RegimeTransitionPredictor(
            regime_detector=regime_detector,
            correlation_service=correlation_service
        )

        # Mock the early warning indicators
        with patch.object(predictor, '_calculate_early_warning_indicators') as mock_calc:
            mock_calc.return_value = {
                "volatility_expansion": 0.8,
                "momentum_change": 0.6,
                "volume_spike": 0.4,
                "range_expansion": 0.7
            }

            # Mock the regime detector
            regime_detector.detect_regime.return_value = {
                "regime": MarketRegime.RANGING,
                "confidence": 0.8,
                "metrics": {
                    "volatility_ratio": 0.5,
                    "adx": 15,
                    "ma_alignment": 0.2
                }
            }

            # Mock the correlated markets analysis
            mock_analyze_markets.return_value = {
                "correlated_markets_dominant_regime": 0.7,
                "correlated_markets_dominant_regime_type": MarketRegime.BREAKOUT.value,
                "inter_market_transition_signal": 0.6,
                "leading_markets_signal": 0.5
            }

            # Mock the update_correlated_markets method
            with patch.object(predictor, 'update_correlated_markets', new_callable=AsyncMock) as mock_update:
                mock_update.return_value = {
                    "GBP/USD": 0.85,
                    "USD/JPY": -0.7,
                    "EUR/JPY": 0.6
                }

                # Create test price data
                price_data = pd.DataFrame({
                    'close': np.random.randn(20).cumsum() + 100,
                    'open': np.random.randn(20).cumsum() + 100,
                    'high': np.random.randn(20).cumsum() + 102,
                    'low': np.random.randn(20).cumsum() + 98,
                    'volume': np.random.randint(1000, 5000, 20)
                }, index=pd.to_datetime(pd.date_range(start='2023-01-01', periods=20)))

                # Call the method with inter-market correlations
                result = await predictor.predict_regime_transition(
                    symbol="EUR/USD",
                    price_data=price_data,
                    timeframe="1h",
                    use_inter_market_correlations=True,
                    correlated_markets_data={
                        "GBP/USD": price_data,
                        "USD/JPY": price_data,
                        "EUR/JPY": price_data
                    }
                )

                # Check that inter-market signals are included
                self.assertIn("inter_market_signals", result)
                self.assertTrue(result["used_inter_market_correlations"])

                # Check that the methods were called correctly
                mock_update.assert_called_once_with("EUR/USD")
                mock_analyze_markets.assert_called_once()

    async def test_update_correlated_markets(self):
        """Test updating correlated markets"""
        # Create a new predictor with mocked dependencies
        regime_detector = MagicMock(spec=MarketRegimeDetector)
        correlation_service = MagicMock(spec=CorrelationTrackingService)

        predictor = RegimeTransitionPredictor(
            regime_detector=regime_detector,
            correlation_service=correlation_service
        )

        # Mock the correlation service
        correlation_service.get_highest_correlations = AsyncMock(return_value=[
            {"symbol": "GBP/USD", "correlation": 0.85},
            {"symbol": "USD/JPY", "correlation": -0.7},
            {"symbol": "EUR/JPY", "correlation": 0.6}
        ])

        # Call the method
        result = await predictor.update_correlated_markets("EUR/USD", min_correlation=0.6)

        # Check the result
        self.assertEqual(len(result), 3)
        self.assertEqual(result["GBP/USD"], 0.85)
        self.assertEqual(result["USD/JPY"], -0.7)
        self.assertEqual(result["EUR/JPY"], 0.6)

        # Check that the correlated markets were stored
        self.assertEqual(predictor.correlated_markets["EUR/USD"], result)

        # Check that the correlation service was called correctly
        correlation_service.get_highest_correlations.assert_called_once_with(
            symbol="EUR/USD",
            min_threshold=0.6
        )

if __name__ == '__main__':
    unittest.main()
