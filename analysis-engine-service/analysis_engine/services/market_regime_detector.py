"""
Market Regime Detection Service

This module provides functionality to detect and classify market regimes
based on price action, volatility, and other market characteristics.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

from analysis_engine.services.tool_effectiveness import MarketRegime


class MarketRegimeAnalyzer:
    """
    Detects and classifies market regimes based on price action characteristics
    and technical indicators.

    Market regimes are macro states of the market such as trending (up/down),
    ranging, volatile, or transitional periods.
    """

    def __init__(self):
        """Initialize the market regime detector"""
        self.logger = logging.getLogger(__name__)
        self.history: List[Dict[str, Any]] = []  # track past detections
        self.transition_history: List[Dict[str, Any]] = []  # track regime transitions

    def detect_regime(
        self,
        price_data: pd.DataFrame,
        lookback_periods: int = 100,
        short_window: int = 20,
        medium_window: int = 50,
        long_window: int = 100,
        volatility_window: int = 20,
        atr_window: int = 14,
        trend_strength_threshold: float = 0.25,
        volatility_threshold: float = 1.5,
        breakout_threshold: float = 2.0
    ) -> Dict[str, Any]:
        """
        Detect the current market regime based on price data.

        Args:
            price_data: DataFrame with OHLC price data and volume (if available)
            lookback_periods: Number of periods to consider for regime detection
            short_window: Window for short-term moving average
            medium_window: Window for medium-term moving average
            long_window: Window for long-term moving average
            volatility_window: Window for volatility calculation
            atr_window: Window for Average True Range calculation
            trend_strength_threshold: Threshold for directional movement index (ADX)
            volatility_threshold: Threshold multiplier for volatility regime classification
            breakout_threshold: Threshold for breakout detection as multiple of ATR

        Returns:
            Dictionary with detected regime and supporting metrics
        """
        if len(price_data) < lookback_periods:
            self.logger.warning(f"Insufficient data for regime detection. Need at least {lookback_periods} periods.")
            return {
                "regime": MarketRegime.UNKNOWN,
                "confidence": 0.0,
                "metrics": {}
            }

        # Ensure we have the required columns
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col.lower() in price_data.columns for col in required_columns):
            self.logger.error(f"Price data must contain {required_columns} columns.")
            return {
                "regime": MarketRegime.UNKNOWN,
                "confidence": 0.0,
                "metrics": {}
            }

        # Standardize column names to lower case
        price_data_copy = price_data.copy()
        price_data_copy.columns = [col.lower() for col in price_data_copy.columns]

        # Use only the most recent data based on lookback_periods
        data = price_data_copy.iloc[-lookback_periods:].reset_index(drop=True)

        # Calculate indicators for regime detection
        regime_metrics = self._calculate_regime_metrics(
            data,
            short_window=short_window,
            medium_window=medium_window,
            long_window=long_window,
            volatility_window=volatility_window,
            atr_window=atr_window
        )

        # Detect market regime based on metrics
        regime_result = self._classify_regime(
            regime_metrics,
            trend_strength_threshold=trend_strength_threshold,
            volatility_threshold=volatility_threshold,
            breakout_threshold=breakout_threshold
        )
        # Get current timestamp
        current_time = datetime.now()

        # Record to history
        current_regime = regime_result.get("regime")
        current_confidence = regime_result.get("confidence")

        self.history.append({
            "timestamp": current_time,
            "regime": current_regime,
            "confidence": current_confidence
        })

        # Check for regime transition
        if len(self.history) >= 2:
            previous_entry = self.history[-2]
            previous_regime = previous_entry.get("regime")

            # If regime has changed, record the transition
            if previous_regime != current_regime:
                self.transition_history.append({
                    "timestamp": current_time,
                    "from_regime": previous_regime,
                    "to_regime": current_regime,
                    "from_confidence": previous_entry.get("confidence"),
                    "to_confidence": current_confidence,
                    "transition_metrics": {
                        "volatility_ratio": regime_result.get("metrics", {}).get("volatility_ratio"),
                        "adx": regime_result.get("metrics", {}).get("adx"),
                        "ma_alignment": regime_result.get("metrics", {}).get("ma_alignment")
                    }
                })

                # Keep transition history limited to last 50 entries
                if len(self.transition_history) > 50:
                    self.transition_history = self.transition_history[-50:]

        return regime_result

    def _calculate_regime_metrics(
        self,
        data: pd.DataFrame,
        short_window: int = 20,
        medium_window: int = 50,
        long_window: int = 100,
        volatility_window: int = 20,
        atr_window: int = 14
    ) -> Dict[str, Any]:
        """
        Calculate technical indicators and metrics used for regime detection.

        Args:
            data: Price data DataFrame
            short_window: Window for short-term moving average
            medium_window: Window for medium-term moving average
            long_window: Window for long-term moving average
            volatility_window: Window for volatility calculation
            atr_window: Window for Average True Range calculation

        Returns:
            Dictionary with calculated metrics
        """
        close = data['close']

        # Moving Averages for trend direction
        ma_short = close.rolling(window=short_window).mean()
        ma_medium = close.rolling(window=medium_window).mean()
        ma_long = close.rolling(window=long_window).mean()

        # Calculate Moving Average slopes (as percentage change)
        ma_short_slope = self._calculate_slope(ma_short, periods=5)
        ma_medium_slope = self._calculate_slope(ma_medium, periods=10)
        ma_long_slope = self._calculate_slope(ma_long, periods=20)

        # MA Alignment - indicates trend strength and direction
        ma_alignment = self._calculate_ma_alignment(ma_short, ma_medium, ma_long)

        # Volatility measures
        std_dev = close.rolling(window=volatility_window).std()
        atr = self._calculate_atr(data, window=atr_window)

        # Normalize volatility by price level (as percentage)
        norm_std_dev = (std_dev / close) * 100
        norm_atr = (atr / close) * 100

        # Calculate recent volatility vs historical volatility
        recent_volatility = norm_std_dev.iloc[-5:].mean()
        historical_volatility = norm_std_dev.iloc[-volatility_window:-5].mean() if len(norm_std_dev) > volatility_window else norm_std_dev.mean()
        volatility_ratio = recent_volatility / historical_volatility if historical_volatility > 0 else 1.0

        # Calculate range-based metrics
        recent_range = self._calculate_price_range(data.iloc[-20:])
        historical_range = self._calculate_price_range(data.iloc[-lookback_periods:-20]) if len(data) > 20 else recent_range
        range_ratio = recent_range / historical_range if historical_range > 0 else 1.0

        # Calculate breakout detection
        is_breakout, breakout_direction = self._detect_breakout(data, atr, window=20, threshold=2.0)

        # Calculate trend strength using Directional Movement Index (ADX)
        adx = self._calculate_adx(data, window=14)
        recent_adx = adx.iloc[-1] if not adx.empty else 0

        # Calculate price momentum
        momentum = self._calculate_momentum(close, periods=10)

        # Return all calculated metrics
        return {
            'ma_short': ma_short.iloc[-1] if not ma_short.empty else None,
            'ma_medium': ma_medium.iloc[-1] if not ma_medium.empty else None,
            'ma_long': ma_long.iloc[-1] if not ma_long.empty else None,
            'ma_short_slope': ma_short_slope,
            'ma_medium_slope': ma_medium_slope,
            'ma_long_slope': ma_long_slope,
            'ma_alignment': ma_alignment,
            'std_dev': std_dev.iloc[-1] if not std_dev.empty else None,
            'atr': atr.iloc[-1] if not atr.empty else None,
            'norm_std_dev': norm_std_dev.iloc[-1] if not norm_std_dev.empty else None,
            'norm_atr': norm_atr.iloc[-1] if not norm_atr.empty else None,
            'recent_volatility': recent_volatility,
            'historical_volatility': historical_volatility,
            'volatility_ratio': volatility_ratio,
            'recent_range': recent_range,
            'historical_range': historical_range,
            'range_ratio': range_ratio,
            'is_breakout': is_breakout,
            'breakout_direction': breakout_direction,
            'adx': recent_adx,
            'momentum': momentum
        }

    def _classify_regime(
        self,
        metrics: Dict[str, Any],
        trend_strength_threshold: float = 25.0,
        volatility_threshold: float = 1.5,
        breakout_threshold: float = 2.0
    ) -> Dict[str, Any]:
        """
        Classify the market regime based on calculated metrics.

        Args:
            metrics: Dictionary with regime metrics from _calculate_regime_metrics
            trend_strength_threshold: ADX threshold for trend strength
            volatility_threshold: Volatility ratio threshold for volatile regime
            breakout_threshold: Threshold for breakout detection

        Returns:
            Dictionary with regime classification and confidence score
        """
        # Initialize confidence scores for each regime type
        regime_scores = {
            MarketRegime.TRENDING_UP: 0.0,
            MarketRegime.TRENDING_DOWN: 0.0,
            MarketRegime.RANGING: 0.0,
            MarketRegime.VOLATILE: 0.0,
            MarketRegime.CHOPPY: 0.0,
            MarketRegime.BREAKOUT: 0.0
        }

        # 1. Trending Market Detection
        adx = metrics.get('adx', 0)
        ma_alignment = metrics.get('ma_alignment', 0)

        # Strong trend indicated by high ADX
        if adx >= trend_strength_threshold:
            # Determine trend direction from MA slopes
            if ma_alignment > 0:
                # Uptrend
                regime_scores[MarketRegime.TRENDING_UP] += 0.6 + min((adx - trend_strength_threshold) / 100, 0.3)
            elif ma_alignment < 0:
                # Downtrend
                regime_scores[MarketRegime.TRENDING_DOWN] += 0.6 + min((adx - trend_strength_threshold) / 100, 0.3)

        # 2. Ranging Market Detection
        range_ratio = metrics.get('range_ratio', 1.0)
        volatility_ratio = metrics.get('volatility_ratio', 1.0)

        if adx < trend_strength_threshold and range_ratio < 1.2 and volatility_ratio < 1.2:
            regime_scores[MarketRegime.RANGING] += 0.7

        # 3. Volatile Market Detection
        if volatility_ratio > volatility_threshold:
            regime_scores[MarketRegime.VOLATILE] += 0.5 + min((volatility_ratio - volatility_threshold) / 2, 0.4)

        # 4. Choppy Market Detection
        ma_short_slope = metrics.get('ma_short_slope', 0)
        ma_medium_slope = metrics.get('ma_medium_slope', 0)

        # Choppy when short-term MA slope changes direction frequently while medium MA is flat
        if abs(ma_medium_slope) < 0.05 and abs(ma_short_slope) > 0.1:
            regime_scores[MarketRegime.CHOPPY] += 0.6

        # 5. Breakout Detection
        is_breakout = metrics.get('is_breakout', False)

        if is_breakout:
            regime_scores[MarketRegime.BREAKOUT] += 0.8

        # Find the regime with the highest score
        max_regime = max(regime_scores, key=regime_scores.get)
        max_score = regime_scores[max_regime]

        # If no strong signal, default to unknown
        if max_score < 0.4:
            max_regime = MarketRegime.UNKNOWN
            max_score = 0.3

        # Return the detected regime with confidence score and supporting metrics
        return {
            "regime": max_regime,
            "confidence": max_score,
            "sub_scores": regime_scores,
            "metrics": metrics
        }

    def _calculate_slope(self, series: pd.Series, periods: int = 5) -> float:
        """
        Calculate the slope of a time series over the specified periods.
        Returns slope as percentage change.
        """
        if len(series) < periods + 1 or series.iloc[-periods-1:].isna().any():
            return 0

        # Calculate percentage change over the period
        start_value = series.iloc[-periods-1]
        end_value = series.iloc[-1]

        if start_value == 0:
            return 0

        return ((end_value / start_value) - 1) * 100

    def _calculate_ma_alignment(self, ma_short: pd.Series, ma_medium: pd.Series, ma_long: pd.Series) -> float:
        """
        Calculate moving average alignment indicator.
        Positive values indicate bullish alignment (short > medium > long)
        Negative values indicate bearish alignment (short < medium < long)
        Values close to 0 indicate no clear alignment
        """
        if ma_short.isna().any() or ma_medium.isna().any() or ma_long.isna().any():
            return 0

        short_val = ma_short.iloc[-1]
        medium_val = ma_medium.iloc[-1]
        long_val = ma_long.iloc[-1]

        # Calculate normalized differences
        short_medium_diff = (short_val - medium_val) / medium_val
        medium_long_diff = (medium_val - long_val) / long_val

        # Combine the differences to get alignment
        # Perfect bullish alignment: short > medium > long
        # Perfect bearish alignment: short < medium < long
        alignment = (short_medium_diff + medium_long_diff) / 2

        # Scale to a reasonable range (-1 to 1)
        return np.clip(alignment * 10, -1, 1)

    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR)
        """
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=window).mean()

        return atr

    def _calculate_price_range(self, data: pd.DataFrame) -> float:
        """
        Calculate the average price range as percentage of price
        """
        if len(data) < 2:
            return 0

        high = data['high']
        low = data['low']
        close = data['close']

        # Calculate daily ranges as percentage of price
        ranges = (high - low) / close

        return ranges.mean() * 100

    def _detect_breakout(
        self,
        data: pd.DataFrame,
        atr: pd.Series,
        window: int = 20,
        threshold: float = 2.0
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect if a breakout has occurred in the recent price action

        Args:
            data: Price data DataFrame
            atr: Series with Average True Range values
            window: Lookback window to establish the range
            threshold: Multiplier of ATR to consider a move as breakout

        Returns:
            Tuple of (is_breakout, direction)
        """
        if len(data) < window + 2 or atr.isna().any():
            return False, None

        # Calculate the recent high and low
        recent_data = data.iloc[-window-1:-1]
        recent_high = recent_data['high'].max()
        recent_low = recent_data['low'].min()

        # Get the latest price and ATR value
        latest_price = data['close'].iloc[-1]
        latest_atr = atr.iloc[-1]

        # Check for breakout
        threshold_value = latest_atr * threshold

        if latest_price > recent_high + threshold_value:
            return True, "up"
        elif latest_price < recent_low - threshold_value:
            return True, "down"
        else:
            return False, None

    def _calculate_adx(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate the Average Directional Index (ADX) - a measure of trend strength
        """
        if len(data) < window * 2:
            return pd.Series([])

        # Calculate +DI and -DI first
        high = data['high']
        low = data['low']
        close = data['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)

        # +DM and -DM
        plus_dm = high.diff()
        minus_dm = low.diff() * -1

        # When +DM < 0 or +DM < -DM, +DM = 0
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)

        # When -DM < 0 or -DM < +DM, -DM = 0
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)

        # Smooth the TR, +DM and -DM using Wilder's smoothing technique
        tr_smoothed = tr.ewm(alpha=1/window, adjust=False).mean()
        plus_dm_smoothed = plus_dm.ewm(alpha=1/window, adjust=False).mean()
        minus_dm_smoothed = minus_dm.ewm(alpha=1/window, adjust=False).mean()

        # Calculate +DI and -DI
        plus_di = 100 * plus_dm_smoothed / tr_smoothed
        minus_di = 100 * minus_dm_smoothed / tr_smoothed

        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/window, adjust=False).mean()

        return adx

    def _calculate_momentum(self, series: pd.Series, periods: int = 10) -> float:
        """
        Calculate momentum as the rate of change over the specified periods
        """
        if len(series) < periods + 1:
            return 0

        # Calculate percentage change
        momentum = (series.iloc[-1] / series.iloc[-periods-1] - 1) * 100

        return momentum

    def get_transition_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the history of regime transitions.

        Args:
            limit: Maximum number of transitions to return

        Returns:
            List of regime transition events
        """
        # Return the most recent transitions up to the limit
        return self.transition_history[-limit:] if self.transition_history else []

    def get_transition_frequency(self, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Calculate the frequency of regime transitions over a period.

        Args:
            lookback_days: Number of days to look back

        Returns:
            Dictionary with transition frequency statistics
        """
        if not self.transition_history:
            return {
                "total_transitions": 0,
                "transitions_per_day": 0,
                "most_common_transition": None,
                "transition_counts": {}
            }

        # Calculate the cutoff time
        cutoff_time = datetime.now() - timedelta(days=lookback_days)

        # Filter transitions within the lookback period
        recent_transitions = [t for t in self.transition_history if t["timestamp"] >= cutoff_time]

        if not recent_transitions:
            return {
                "total_transitions": 0,
                "transitions_per_day": 0,
                "most_common_transition": None,
                "transition_counts": {}
            }

        # Count transitions by type
        transition_counts = {}
        for transition in recent_transitions:
            from_regime = transition["from_regime"]
            to_regime = transition["to_regime"]
            key = f"{from_regime} -> {to_regime}"

            if key not in transition_counts:
                transition_counts[key] = 0

            transition_counts[key] += 1

        # Find the most common transition
        most_common = max(transition_counts.items(), key=lambda x: x[1]) if transition_counts else (None, 0)

        # Calculate transitions per day
        total_transitions = len(recent_transitions)
        transitions_per_day = total_transitions / lookback_days

        return {
            "total_transitions": total_transitions,
            "transitions_per_day": transitions_per_day,
            "most_common_transition": most_common[0],
            "most_common_count": most_common[1],
            "transition_counts": transition_counts
        }


class MarketRegimeService:
    """Service for detecting market regimes and maintaining regime history"""

    def __init__(self):
        """Initialize the market regime service"""
        self.detector = MarketRegimeAnalyzer()
        self.regime_history = {}
        self.logger = logging.getLogger(__name__)

    def detect_current_regime(
        self,
        symbol: str,
        timeframe: str,
        price_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect the current market regime for a specific symbol and timeframe

        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            price_data: OHLC price data

        Returns:
            Dictionary with detected regime and supporting metrics
        """
        try:
            # Detect regime
            result = self.detector.detect_regime(price_data)

            # Store in history
            key = f"{symbol}_{timeframe}"
            if key not in self.regime_history:
                self.regime_history[key] = []

            self.regime_history[key].append({
                "timestamp": datetime.now(),
                "regime": result["regime"],
                "confidence": result["confidence"]
            })

            # Keep history limited to last 100 entries
            if len(self.regime_history[key]) > 100:
                self.regime_history[key] = self.regime_history[key][-100:]

            return result
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            return {
                "regime": MarketRegime.UNKNOWN,
                "confidence": 0.0,
                "metrics": {},
                "error": str(e)
            }

    def get_regime_history(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get historical regime data for a specific symbol and timeframe

        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            limit: Maximum number of historical entries to return

        Returns:
            List of historical regime entries
        """
        key = f"{symbol}_{timeframe}"
        history = self.regime_history.get(key, [])

        # Return last 'limit' entries
        return history[-limit:]

    def get_dominant_regime(
        self,
        symbol: str,
        timeframe: str,
        lookback_periods: int = 5
    ) -> Dict[str, Any]:
        """
        Get the dominant market regime over recent history

        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            lookback_periods: Number of historical entries to consider

        Returns:
            Dictionary with dominant regime and confidence
        """
        # Get recent history
        history = self.get_regime_history(symbol, timeframe, lookback_periods)

        if not history:
            return {
                "regime": MarketRegime.UNKNOWN,
                "confidence": 0.0
            }

        # Count occurrences of each regime
        regime_counts = {}
        total_confidence = {}

        for entry in history:
            regime = entry["regime"]
            confidence = entry["confidence"]

            if regime not in regime_counts:
                regime_counts[regime] = 0
                total_confidence[regime] = 0

            regime_counts[regime] += 1
            total_confidence[regime] += confidence

        # Find the most frequent regime
        dominant_regime = max(regime_counts, key=regime_counts.get)

        # Calculate average confidence for the dominant regime
        avg_confidence = total_confidence[dominant_regime] / regime_counts[dominant_regime]

        return {
            "regime": dominant_regime,
            "confidence": avg_confidence,
            "occurrence_rate": regime_counts[dominant_regime] / len(history)
        }

    def get_regime_transitions(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get the history of regime transitions for a specific symbol and timeframe

        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            limit: Maximum number of transitions to return

        Returns:
            List of regime transition events
        """
        return self.detector.get_transition_history(limit)

    def get_transition_statistics(
        self,
        symbol: str,
        timeframe: str,
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """
        Get statistics about regime transitions for a specific symbol and timeframe

        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            lookback_days: Number of days to look back

        Returns:
            Dictionary with transition statistics
        """
        stats = self.detector.get_transition_frequency(lookback_days)

        # Add symbol and timeframe information
        stats["symbol"] = symbol
        stats["timeframe"] = timeframe

        return stats
