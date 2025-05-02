"""
Comprehensive Indicator Test Suite

This module provides a robust testing framework for all indicators
implemented in the forex trading platform.
"""

import logging
import time
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
import json
import os
import sys
import pytest
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path to ensure imports work
sys.path.append(str(Path(__file__).parent.parent.parent))

from feature_store_service.indicators import (
    moving_averages, 
    oscillators, 
    volatility, 
    volume,
    indicator_registry,
    statistical_regression_indicators,
    multi_timeframe,
    fractal_indicators,
    gann_tools,
    ml_integration
)

logger = logging.getLogger(__name__)

class IndicatorTestCase(unittest.TestCase):
    """Base class for all indicator tests"""
    
    def setUp(self):
        """Set up test fixture"""
        # Create standard test data
        self.dates = pd.date_range(start='2020-01-01', periods=500, freq='1h')
        
        # Generate price data with known patterns
        np.random.seed(42)  # For reproducibility
        
        # Base price with trend
        base_price = np.linspace(100, 150, len(self.dates))
        
        # Add cyclical component
        cycles = 10 * np.sin(np.linspace(0, 15, len(self.dates)))
        
        # Add random noise
        noise = np.random.normal(0, 3, len(self.dates))
        
        # Combine components
        prices = base_price + cycles + noise
        
        # Create OHLC data
        self.data = pd.DataFrame({
            'timestamp': self.dates,
            'open': prices,
            'high': prices + np.random.uniform(0, 2, len(prices)),
            'low': prices - np.random.uniform(0, 2, len(prices)),
            'close': prices + np.random.normal(0, 1, len(prices)),
            'volume': np.random.randint(1000, 10000, len(prices))
        })
        
        # Set timestamp as index
        self.data.set_index('timestamp', inplace=True)
        
        # Generate specific patterns for testing
        self._generate_test_patterns()
    
    def _generate_test_patterns(self):
        """Generate additional test patterns"""
        # Trending data (uptrend)
        dates_trend = pd.date_range(start='2020-01-01', periods=100, freq='1h')
        trend = np.linspace(100, 200, len(dates_trend))
        noise = np.random.normal(0, 2, len(dates_trend))
        
        self.trending_data = pd.DataFrame({
            'timestamp': dates_trend,
            'open': trend + noise,
            'high': trend + noise + 1,
            'low': trend + noise - 1,
            'close': trend + noise + 0.5,
            'volume': np.random.randint(1000, 10000, len(dates_trend))
        }).set_index('timestamp')
        
        # Ranging data
        dates_range = pd.date_range(start='2020-01-01', periods=100, freq='1h')
        center = 150
        range_price = center + 20 * np.sin(np.linspace(0, 6*np.pi, len(dates_range)))
        noise = np.random.normal(0, 1, len(dates_range))
        
        self.ranging_data = pd.DataFrame({
            'timestamp': dates_range,
            'open': range_price + noise,
            'high': range_price + noise + 2,
            'low': range_price + noise - 2,
            'close': range_price + noise + 0.5,
            'volume': np.random.randint(1000, 10000, len(dates_range))
        }).set_index('timestamp')
        
        # Volatile data
        dates_volatile = pd.date_range(start='2020-01-01', periods=100, freq='1h')
        base = np.linspace(100, 150, len(dates_volatile))
        volatility = np.random.normal(0, 10, len(dates_volatile))
        
        self.volatile_data = pd.DataFrame({
            'timestamp': dates_volatile,
            'open': base + volatility,
            'high': base + volatility + np.random.uniform(0, 5, len(dates_volatile)),
            'low': base + volatility - np.random.uniform(0, 5, len(dates_volatile)),
            'close': base + volatility + np.random.normal(0, 2, len(dates_volatile)),
            'volume': np.random.randint(1000, 30000, len(dates_volatile))
        }).set_index('timestamp')
    
    def assert_indicator_output(self, indicator_fn: Callable, expected_behavior: str, 
                             params: Dict = None, data: pd.DataFrame = None):
        """
        Assert that an indicator function behaves as expected
        
        Args:
            indicator_fn: The indicator function to test
            expected_behavior: String description of expected behavior to check
            params: Parameters for the indicator function
            data: Data to use for testing (uses self.data if None)
        """
        test_data = data if data is not None else self.data
        params = params or {}
        
        # Call the indicator function
        result = indicator_fn(test_data, **params)
        
        # Check that result is not None or empty
        self.assertIsNotNone(result, "Indicator returned None")
        
        # Ensure result is a DataFrame or Series
        self.assertTrue(isinstance(result, (pd.DataFrame, pd.Series)), 
                        f"Expected DataFrame or Series, got {type(result)}")
        
        # Check if result has expected columns/values based on behavior
        if expected_behavior == "above_zero":
            # Check if any values are above zero
            if isinstance(result, pd.DataFrame):
                for col in result.columns:
                    self.assertTrue((result[col] > 0).any(), 
                                    f"Expected some values above zero for {col}")
            else:
                self.assertTrue((result > 0).any(), 
                                "Expected some values above zero")
                                
        elif expected_behavior == "between_0_and_100":
            # Check if values are between 0 and 100
            if isinstance(result, pd.DataFrame):
                for col in result.columns:
                    self.assertTrue(((result[col] >= 0) & (result[col] <= 100)).all(), 
                                    f"Expected values between 0 and 100 for {col}")
            else:
                self.assertTrue(((result >= 0) & (result <= 100)).all(), 
                                "Expected values between 0 and 100")
                                
        elif expected_behavior == "follows_trend":
            # Check if indicator generally follows the trend in price
            if isinstance(result, pd.DataFrame):
                # Just check first column for simplicity
                col = result.columns[0]
                correlation = np.corrcoef(test_data['close'].values[len(test_data)-len(result):], 
                                         result[col].values)[0, 1]
            else:
                correlation = np.corrcoef(test_data['close'].values[len(test_data)-len(result):], 
                                         result.values)[0, 1]
                                         
            self.assertGreater(abs(correlation), 0.5, 
                             f"Expected stronger correlation with trend: {correlation}")
        
        return result


class MovingAverageTests(IndicatorTestCase):
    """Tests for moving average indicators"""
    
    def test_simple_moving_average(self):
        """Test simple moving average calculation"""
        periods = [5, 10, 20, 50, 200]
        
        for period in periods:
            with self.subTest(f"SMA-{period}"):
                result = self.assert_indicator_output(
                    moving_averages.simple_moving_average,
                    "follows_trend",
                    {"period": period}
                )
                
                # Check length
                expected_length = len(self.data) - period + 1
                self.assertEqual(len(result), expected_length)
                
                # Check manually for a few points
                for i in range(min(5, len(result))):
                    idx = period - 1 + i
                    expected_value = self.data['close'][idx-period+1:idx+1].mean()
                    self.assertAlmostEqual(result.iloc[i], expected_value)
    
    def test_exponential_moving_average(self):
        """Test exponential moving average calculation"""
        periods = [5, 10, 20, 50, 200]
        
        for period in periods:
            with self.subTest(f"EMA-{period}"):
                result = self.assert_indicator_output(
                    moving_averages.exponential_moving_average,
                    "follows_trend",
                    {"period": period}
                )
                
                # Check length
                self.assertEqual(len(result), len(self.data))
                
                # Check smoothing effect
                std_price = self.data['close'].std()
                std_ema = result.std()
                self.assertLess(std_ema, std_price, 
                              f"EMA should be smoother than price (lower std dev)")
    
    def test_weighted_moving_average(self):
        """Test weighted moving average calculation"""
        periods = [5, 10, 20]
        
        for period in periods:
            with self.subTest(f"WMA-{period}"):
                result = self.assert_indicator_output(
                    moving_averages.weighted_moving_average,
                    "follows_trend",
                    {"period": period}
                )
                
                # Check length
                expected_length = len(self.data) - period + 1
                self.assertEqual(len(result), expected_length)
    
    def test_hull_moving_average(self):
        """Test Hull moving average calculation"""
        periods = [5, 10, 20]
        
        for period in periods:
            with self.subTest(f"Hull-{period}"):
                result = self.assert_indicator_output(
                    moving_averages.hull_moving_average,
                    "follows_trend",
                    {"period": period}
                )
                
                # Check that Hull responds faster to trend changes
                # by comparing with SMA of same period
                sma = moving_averages.simple_moving_average(self.data, period=period)
                
                # Get common index range
                common_idx = result.index.intersection(sma.index)
                
                # Calculate lag correlation
                lag_corr_hull = pd.Series(self.data['close']).shift(-1).loc[common_idx].corr(
                    result.loc[common_idx])
                lag_corr_sma = pd.Series(self.data['close']).shift(-1).loc[common_idx].corr(
                    sma.loc[common_idx])
                
                self.assertGreaterEqual(lag_corr_hull, lag_corr_sma, 
                                      "Hull MA should respond faster to price changes than SMA")


class OscillatorTests(IndicatorTestCase):
    """Tests for oscillator indicators"""
    
    def test_rsi(self):
        """Test Relative Strength Index calculation"""
        periods = [9, 14, 25]
        
        for period in periods:
            with self.subTest(f"RSI-{period}"):
                result = self.assert_indicator_output(
                    oscillators.relative_strength_index,
                    "between_0_and_100",
                    {"period": period}
                )
                
                # Check if RSI goes above 70 in uptrends and below 30 in downtrends
                uptrend_data = pd.DataFrame({
                    'close': np.linspace(100, 200, 100)
                })
                uptrend_rsi = oscillators.relative_strength_index(uptrend_data, period=period)
                self.assertTrue((uptrend_rsi > 70).any(), "RSI should exceed 70 in strong uptrend")
                
                downtrend_data = pd.DataFrame({
                    'close': np.linspace(200, 100, 100)
                })
                downtrend_rsi = oscillators.relative_strength_index(downtrend_data, period=period)
                self.assertTrue((downtrend_rsi < 30).any(), "RSI should go below 30 in strong downtrend")
    
    def test_stochastic_oscillator(self):
        """Test Stochastic Oscillator calculation"""
        k_periods = [5, 14]
        d_periods = [3, 5]
        
        for k_period in k_periods:
            for d_period in d_periods:
                with self.subTest(f"Stochastic-K{k_period}-D{d_period}"):
                    result = self.assert_indicator_output(
                        oscillators.stochastic_oscillator,
                        "between_0_and_100",
                        {"k_period": k_period, "d_period": d_period}
                    )
                    
                    # Check if result has K and D lines
                    self.assertIn("%K", result.columns)
                    self.assertIn("%D", result.columns)
                    
                    # Check if K is more volatile than D
                    k_std = result["%K"].std()
                    d_std = result["%D"].std()
                    self.assertGreater(k_std, d_std, "%K should be more volatile than %D")
    
    def test_macd(self):
        """Test MACD calculation"""
        result = self.assert_indicator_output(
            oscillators.macd,
            "above_zero",
            {"fast_period": 12, "slow_period": 26, "signal_period": 9}
        )
        
        # Check if result has all expected components
        self.assertIn("MACD", result.columns)
        self.assertIn("Signal", result.columns)
        self.assertIn("Histogram", result.columns)
        
        # Check if MACD Histogram sums to near zero (mean-reverting)
        hist_sum = result["Histogram"].sum()
        self.assertAlmostEqual(hist_sum / len(result), 0, delta=1.0)
    
    def test_commodity_channel_index(self):
        """Test CCI calculation"""
        periods = [14, 20]
        
        for period in periods:
            with self.subTest(f"CCI-{period}"):
                result = self.assert_indicator_output(
                    oscillators.commodity_channel_index,
                    "above_zero",
                    {"period": period}
                )
                
                # Check if CCI mean is close to 0
                mean_cci = result.mean()
                self.assertAlmostEqual(mean_cci, 0, delta=20)
                
                # Check if CCI mostly stays between -100 and +100
                within_range = ((result >= -100) & (result <= 100)).mean()
                self.assertGreater(within_range, 0.7, 
                                 "CCI should stay between -100 and +100 most of the time")


class VolatilityTests(IndicatorTestCase):
    """Tests for volatility indicators"""
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        periods = [20]
        deviations = [2.0, 2.5]
        
        for period in periods:
            for dev in deviations:
                with self.subTest(f"BB-{period}-{dev}"):
                    result = self.assert_indicator_output(
                        volatility.bollinger_bands,
                        "follows_trend",
                        {"period": period, "deviations": dev}
                    )
                    
                    # Check if result has all expected bands
                    self.assertIn("middle", result.columns)
                    self.assertIn("upper", result.columns)
                    self.assertIn("lower", result.columns)
                    
                    # Check if middle band is equal to SMA
                    sma = moving_averages.simple_moving_average(self.data, period=period)
                    pd.testing.assert_series_equal(result["middle"], sma)
                    
                    # Check if price touches or crosses bands occasionally
                    close = self.data['close']
                    upper_touches = ((close > result["upper"]) | 
                                   (close.shift(1) <= result["upper"].shift(1) & 
                                    close >= result["upper"])).mean()
                    lower_touches = ((close < result["lower"]) | 
                                   (close.shift(1) >= result["lower"].shift(1) & 
                                    close <= result["lower"])).mean()
                    
                    # Should touch bands around 5% of the time with 2 standard deviations
                    expected_touch_rate = 0.05 if dev == 2.0 else 0.01
                    self.assertGreater(upper_touches, expected_touch_rate * 0.5,
                                     "Price should occasionally touch upper band")
                    self.assertGreater(lower_touches, expected_touch_rate * 0.5,
                                     "Price should occasionally touch lower band")
    
    def test_average_true_range(self):
        """Test Average True Range calculation"""
        periods = [14, 21]
        
        for period in periods:
            with self.subTest(f"ATR-{period}"):
                result = self.assert_indicator_output(
                    volatility.average_true_range,
                    "above_zero",
                    {"period": period}
                )
                
                # Check if ATR is higher for volatile data than for ranging data
                atr_volatile = volatility.average_true_range(self.volatile_data, period=period)
                atr_ranging = volatility.average_true_range(self.ranging_data, period=period)
                
                self.assertGreater(atr_volatile.mean(), atr_ranging.mean(),
                                 "ATR should be higher for volatile data")
    
    def test_keltner_channels(self):
        """Test Keltner Channels calculation"""
        periods = [20]
        atrs = [2]
        
        for period in periods:
            for atr_multiple in atrs:
                with self.subTest(f"KC-{period}-{atr_multiple}"):
                    result = self.assert_indicator_output(
                        volatility.keltner_channels,
                        "follows_trend",
                        {"period": period, "atr_period": 10, "atr_multiple": atr_multiple}
                    )
                    
                    # Check if result has all expected bands
                    self.assertIn("middle", result.columns)
                    self.assertIn("upper", result.columns)
                    self.assertIn("lower", result.columns)
                    
                    # Check if channels widen during volatile periods
                    kc_volatile = volatility.keltner_channels(
                        self.volatile_data, 
                        period=period, 
                        atr_period=10, 
                        atr_multiple=atr_multiple
                    )
                    kc_ranging = volatility.keltner_channels(
                        self.ranging_data, 
                        period=period, 
                        atr_period=10, 
                        atr_multiple=atr_multiple
                    )
                    
                    volatile_width = (kc_volatile["upper"] - kc_volatile["lower"]).mean()
                    ranging_width = (kc_ranging["upper"] - kc_ranging["lower"]).mean()
                    
                    self.assertGreater(volatile_width, ranging_width,
                                     "Keltner Channels should be wider during volatile periods")


class VolumeTests(IndicatorTestCase):
    """Tests for volume indicators"""
    
    def test_on_balance_volume(self):
        """Test On-Balance Volume calculation"""
        result = self.assert_indicator_output(
            volume.on_balance_volume,
            "above_zero"
        )
        
        # Check if OBV changes direction with price
        obv_changes = result.diff().dropna()
        price_changes = self.data['close'].diff().dropna()
        
        # Direction agreement should be better than random
        direction_agreement = ((obv_changes > 0) == (price_changes > 0)).mean()
        self.assertGreater(direction_agreement, 0.5,
                         "OBV direction should agree with price direction more than 50% of the time")
    
    def test_volume_weighted_average_price(self):
        """Test VWAP calculation"""
        result = self.assert_indicator_output(
            volume.volume_weighted_average_price,
            "follows_trend"
        )
        
        # Check if VWAP is between high and low prices
        highs = self.data['high']
        lows = self.data['low']
        
        # VWAP should generally be between high and low for the day
        within_range = ((result >= lows) & (result <= highs)).mean()
        self.assertGreater(within_range, 0.9,
                         "VWAP should generally be between daily high and low")


class PatternTests(IndicatorTestCase):
    """Tests for pattern recognition indicators"""
    
    def test_engulfing_pattern(self):
        """Test engulfing pattern detection"""
        # Create data with known patterns
        dates = pd.date_range(start='2020-01-01', periods=5, freq='1d')
        
        # Create bullish engulfing pattern
        bullish_data = pd.DataFrame({
            'open': [100, 100, 100, 98, 95],
            'high': [105, 105, 102, 103, 105],
            'low': [95, 95, 98, 95, 94],
            'close': [102, 101, 99, 97, 103]  # Last bar engulfs previous
        }, index=dates)
        
        # Create bearish engulfing pattern
        bearish_data = pd.DataFrame({
            'open': [100, 100, 100, 102, 105],
            'high': [105, 105, 102, 107, 108],
            'low': [95, 95, 98, 98, 95],
            'close': [102, 101, 101, 103, 97]  # Last bar engulfs previous
        }, index=dates)
        
        from analysis_engine.analysis.pattern_recognition import detect_engulfing_patterns
        
        bullish_result = detect_engulfing_patterns(bullish_data)
        bearish_result = detect_engulfing_patterns(bearish_data)
        
        # Check if patterns were detected
        self.assertEqual(bullish_result.iloc[-1], 1,
                       "Should detect bullish engulfing pattern")
        self.assertEqual(bearish_result.iloc[-1], -1,
                       "Should detect bearish engulfing pattern")


class IndicatorRegistryTest(IndicatorTestCase):
    """Tests for indicator registry functionality"""
    
    def test_registry_functionality(self):
        """Test indicator registry registration and retrieval"""
        # Register indicators
        registry = indicator_registry.IndicatorRegistry()
        
        # Register a test indicator
        def test_indicator(data, param1=1, param2=2):
            return data['close'] * param1 + param2
        
        registry.register("test_indicator", test_indicator, 
                        description="Test indicator for unit testing",
                        category="test",
                        parameters={"param1": "Multiplier", "param2": "Addition"})
        
        # Check if indicator was registered
        self.assertIn("test_indicator", registry.get_all_indicators())
        
        # Test get indicator
        retrieved = registry.get_indicator("test_indicator")
        self.assertEqual(retrieved["function"], test_indicator)
        
        # Test call indicator
        result = registry.calculate("test_indicator", self.data, param1=3, param2=4)
        expected = self.data['close'] * 3 + 4
        pd.testing.assert_series_equal(result, expected)


def run_indicator_tests():
    """Run all indicator tests"""
    unittest.main()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_indicator_tests()
"""
