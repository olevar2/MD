"""
Integration tests for the strategy enhancement services.

This test suite validates that all the strategy enhancement services
work together correctly to improve trading strategy performance.
"""
import unittest
import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any

from analysis_engine.services.timeframe_optimization_service import TimeframeOptimizationService, SignalOutcome
from analysis_engine.multi_asset.currency_strength_analyzer import CurrencyStrengthAnalyzer
from analysis_engine.multi_asset.related_pairs_confluence_detector import RelatedPairsConfluenceDetector
from analysis_engine.analysis.sequence_pattern_recognizer import SequencePatternRecognizer, PatternType
from analysis_engine.services.regime_transition_predictor import RegimeTransitionPredictor
from analysis_engine.services.market_regime_detector import MarketRegime


class TestStrategyEnhancementIntegration(unittest.TestCase):
    """Integration test for strategy enhancement services."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create timeframe optimization service
        self.timeframes = ["1m", "5m", "15m", "1h", "4h"]
        self.primary_timeframe = "1h"
        self.timeframe_optimizer = TimeframeOptimizationService(
            timeframes=self.timeframes,
            primary_timeframe=self.primary_timeframe
        )
        
        # Create currency strength analyzer
        self.currency_strength_analyzer = CurrencyStrengthAnalyzer()
        
        # Create related pairs confluence detector
        self.related_pairs_detector = RelatedPairsConfluenceDetector(
            currency_strength_analyzer=self.currency_strength_analyzer
        )
        
        # Create sequence pattern recognizer
        self.pattern_recognizer = SequencePatternRecognizer()
        
        # Create regime transition predictor
        self.regime_transition_predictor = RegimeTransitionPredictor()
        
        # Create test price data
        self.price_data = self._create_test_price_data()
        
        # Add some performance history to the timeframe optimizer
        self._add_test_performance_data()
    
    def _create_test_price_data(self) -> Dict[str, pd.DataFrame]:
        """Create test price data for multiple timeframes and pairs."""
        price_data = {}
        
        # Create data for EUR/USD
        for tf in self.timeframes:
            # Create base data
            length = 100
            df = pd.DataFrame({
                'open': np.random.normal(1.10, 0.01, length),
                'high': np.random.normal(1.11, 0.01, length),
                'low': np.random.normal(1.09, 0.01, length),
                'close': np.random.normal(1.10, 0.01, length),
                'volume': np.random.randint(1000, 5000, length)
            })
            
            # Add trend for 1h and 4h timeframes
            if tf in ["1h", "4h"]:
                df['close'] = np.linspace(1.10, 1.15, length)  # Bullish trend
            
            # Add technical indicators
            df['short_ma'] = df['close'].rolling(window=10).mean()
            df['medium_ma'] = df['close'].rolling(window=20).mean()
            df['ATR'] = (df['high'] - df['low']).rolling(window=14).mean()
            
            price_data[f"EUR/USD_{tf}"] = df
        
        # Create data for GBP/USD (correlated with EUR/USD)
        for tf in self.timeframes:
            # Create base data
            length = 100
            df = pd.DataFrame({
                'open': np.random.normal(1.30, 0.01, length),
                'high': np.random.normal(1.31, 0.01, length),
                'low': np.random.normal(1.29, 0.01, length),
                'close': np.random.normal(1.30, 0.01, length),
                'volume': np.random.randint(1000, 5000, length)
            })
            
            # Add trend for 1h and 4h timeframes (correlated with EUR/USD)
            if tf in ["1h", "4h"]:
                df['close'] = np.linspace(1.30, 1.35, length)  # Bullish trend
            
            # Add technical indicators
            df['short_ma'] = df['close'].rolling(window=10).mean()
            df['medium_ma'] = df['close'].rolling(window=20).mean()
            df['ATR'] = (df['high'] - df['low']).rolling(window=14).mean()
            
            price_data[f"GBP/USD_{tf}"] = df
        
        # Create data for USD/JPY (negatively correlated with EUR/USD)
        for tf in self.timeframes:
            # Create base data
            length = 100
            df = pd.DataFrame({
                'open': np.random.normal(110.0, 0.5, length),
                'high': np.random.normal(111.0, 0.5, length),
                'low': np.random.normal(109.0, 0.5, length),
                'close': np.random.normal(110.0, 0.5, length),
                'volume': np.random.randint(1000, 5000, length)
            })
            
            # Add trend for 1h and 4h timeframes (negatively correlated with EUR/USD)
            if tf in ["1h", "4h"]:
                df['close'] = np.linspace(110.0, 105.0, length)  # Bearish trend
            
            # Add technical indicators
            df['short_ma'] = df['close'].rolling(window=10).mean()
            df['medium_ma'] = df['close'].rolling(window=20).mean()
            df['ATR'] = (df['high'] - df['low']).rolling(window=14).mean()
            
            price_data[f"USD/JPY_{tf}"] = df
        
        return price_data
    
    def _add_test_performance_data(self):
        """Add test performance data to the timeframe optimizer."""
        # Add winning signals for 1h timeframe
        for i in range(20):
            self.timeframe_optimizer.record_timeframe_performance(
                timeframe="1h",
                outcome=SignalOutcome.WIN,
                symbol="EUR/USD",
                pips_result=10.0,
                confidence=0.8,
                timestamp=datetime.now() - timedelta(days=i)
            )
        
        # Add some losing signals for 1h timeframe
        for i in range(10):
            self.timeframe_optimizer.record_timeframe_performance(
                timeframe="1h",
                outcome=SignalOutcome.LOSS,
                symbol="EUR/USD",
                pips_result=-5.0,
                confidence=0.7,
                timestamp=datetime.now() - timedelta(days=i)
            )
        
        # Add winning signals for 4h timeframe (higher win rate)
        for i in range(25):
            self.timeframe_optimizer.record_timeframe_performance(
                timeframe="4h",
                outcome=SignalOutcome.WIN,
                symbol="EUR/USD",
                pips_result=15.0,
                confidence=0.8,
                timestamp=datetime.now() - timedelta(days=i)
            )
        
        # Add some losing signals for 4h timeframe
        for i in range(5):
            self.timeframe_optimizer.record_timeframe_performance(
                timeframe="4h",
                outcome=SignalOutcome.LOSS,
                symbol="EUR/USD",
                pips_result=-7.0,
                confidence=0.7,
                timestamp=datetime.now() - timedelta(days=i)
            )
    
    def _prepare_price_data_for_symbol(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Prepare price data for a specific symbol."""
        symbol_data = {}
        for key, df in self.price_data.items():
            if key.startswith(f"{symbol}_"):
                timeframe = key.split('_')[1]
                symbol_data[timeframe] = df
        return symbol_data
    
    def test_end_to_end_strategy_enhancement(self):
        """Test end-to-end integration of all strategy enhancement services."""
        # Prepare price data for EUR/USD
        symbol = "EUR/USD"
        price_data = self._prepare_price_data_for_symbol(symbol)
        
        # Step 1: Optimize timeframe weights
        weights = self.timeframe_optimizer.optimize_timeframe_weights()
        
        # Check that weights are calculated
        self.assertGreater(len(weights), 0)
        
        # Step 2: Calculate currency strength
        strength_values = self.currency_strength_analyzer.calculate_currency_strength(
            {f"{symbol}": df for symbol, df in self.price_data.items()}
        )
        
        # Check that strength values are calculated
        self.assertGreater(len(strength_values), 0)
        
        # Step 3: Detect related pairs confluence
        related_pairs = {
            "GBP/USD": 0.8,  # Positively correlated
            "USD/JPY": -0.7  # Negatively correlated
        }
        
        confluence_result = self.related_pairs_detector.detect_confluence(
            symbol=symbol,
            price_data={key.split('_')[0]: df for key, df in self.price_data.items()},
            signal_type="trend",
            signal_direction="bullish",
            related_pairs=related_pairs
        )
        
        # Check that confluence is detected
        self.assertGreater(confluence_result.get("confluence_score", 0), 0)
        
        # Step 4: Detect sequence patterns
        pattern_results = self.pattern_recognizer.detect_patterns(price_data)
        
        # Check that patterns are analyzed
        self.assertIn("timeframes_analyzed", pattern_results)
        
        # Step 5: Predict regime transitions
        prediction = self.regime_transition_predictor.predict_regime_transition(
            symbol=symbol,
            price_data=price_data["1h"],
            current_regime=MarketRegime.TRENDING,
            timeframe="1h"
        )
        
        # Check that prediction is made
        self.assertIn("transition_probability", prediction)
        
        # Step 6: Create enhanced trading signal
        signal = self._create_enhanced_trading_signal(
            symbol=symbol,
            price_data=price_data,
            weights=weights,
            strength_values=strength_values,
            confluence_result=confluence_result,
            pattern_results=pattern_results,
            regime_prediction=prediction
        )
        
        # Check that signal is created with enhanced properties
        self.assertIn("timeframe_weights", signal)
        self.assertIn("currency_strength", signal)
        self.assertIn("related_pairs_confluence", signal)
        self.assertIn("pattern_detection", signal)
        self.assertIn("regime_transition", signal)
        
        # Check that confidence is calculated
        self.assertIn("confidence", signal)
        self.assertGreaterEqual(signal["confidence"], 0.0)
        self.assertLessEqual(signal["confidence"], 1.0)
    
    def _create_enhanced_trading_signal(
        self,
        symbol: str,
        price_data: Dict[str, pd.DataFrame],
        weights: Dict[str, float],
        strength_values: Dict[str, float],
        confluence_result: Dict[str, Any],
        pattern_results: Dict[str, Any],
        regime_prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create an enhanced trading signal using all services."""
        # Get current price
        current_price = price_data["1h"]["close"].iloc[-1]
        
        # Calculate base signal properties
        direction = "bullish" if price_data["1h"]["short_ma"].iloc[-1] > price_data["1h"]["medium_ma"].iloc[-1] else "bearish"
        
        # Calculate stop loss and take profit
        atr = price_data["1h"]["ATR"].iloc[-1]
        stop_loss = current_price - (atr * 2.0) if direction == "bullish" else current_price + (atr * 2.0)
        take_profit = current_price + (atr * 4.0) if direction == "bullish" else current_price - (atr * 4.0)
        
        # Create base signal
        signal = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "direction": direction,
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "confidence": 0.7  # Base confidence
        }
        
        # Enhance with timeframe weights
        signal["timeframe_weights"] = weights
        
        # Enhance with currency strength
        base_currency, quote_currency = symbol.split('/')
        signal["currency_strength"] = {
            "base": strength_values.get(base_currency, 0),
            "quote": strength_values.get(quote_currency, 0),
            "difference": strength_values.get(base_currency, 0) - strength_values.get(quote_currency, 0)
        }
        
        # Adjust confidence based on currency strength
        strength_diff = signal["currency_strength"]["difference"]
        if (direction == "bullish" and strength_diff > 0) or (direction == "bearish" and strength_diff < 0):
            signal["confidence"] += 0.1
        else:
            signal["confidence"] -= 0.1
        
        # Enhance with related pairs confluence
        signal["related_pairs_confluence"] = {
            "score": confluence_result.get("confluence_score", 0),
            "confirmations": confluence_result.get("confirmation_count", 0),
            "contradictions": confluence_result.get("contradiction_count", 0)
        }
        
        # Adjust confidence based on confluence
        signal["confidence"] += confluence_result.get("confluence_score", 0) * 0.2
        
        # Enhance with pattern detection
        if "sequence_patterns" in pattern_results and pattern_results["sequence_patterns"]:
            top_pattern = pattern_results["sequence_patterns"][0]
            signal["pattern_detection"] = {
                "type": top_pattern.get("type"),
                "confidence": top_pattern.get("confidence", 0),
                "timeframes": top_pattern.get("timeframes", [])
            }
            
            # Adjust confidence based on pattern confidence
            signal["confidence"] += top_pattern.get("confidence", 0) * 0.1
        else:
            signal["pattern_detection"] = {"found": False}
        
        # Enhance with regime transition prediction
        signal["regime_transition"] = {
            "current_regime": regime_prediction.get("current_regime"),
            "predicted_regime": regime_prediction.get("most_likely_next_regime"),
            "probability": regime_prediction.get("transition_probability", 0)
        }
        
        # Adjust confidence based on regime compatibility
        current_regime = regime_prediction.get("current_regime")
        next_regime = regime_prediction.get("most_likely_next_regime")
        
        if direction == "bullish" and next_regime in ["TRENDING_UP", "BREAKOUT"]:
            signal["confidence"] += 0.1
        elif direction == "bearish" and next_regime in ["TRENDING_DOWN", "BREAKOUT"]:
            signal["confidence"] += 0.1
        else:
            signal["confidence"] -= 0.1
        
        # Ensure confidence is within valid range
        signal["confidence"] = max(0.0, min(1.0, signal["confidence"]))
        
        return signal


if __name__ == "__main__":
    unittest.main()
