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
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

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
        self.history: List[Dict[str, Any]] = []
        self.transition_history: List[Dict[str, Any]] = []

    def detect_regime(self, price_data: pd.DataFrame, lookback_periods: int
        =100, short_window: int=20, medium_window: int=50, long_window: int
        =100, volatility_window: int=20, atr_window: int=14,
        trend_strength_threshold: float=0.25, volatility_threshold: float=
        1.5, breakout_threshold: float=2.0) ->Dict[str, Any]:
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
            self.logger.warning(
                f'Insufficient data for regime detection. Need at least {lookback_periods} periods.'
                )
            return {'regime': MarketRegime.UNKNOWN, 'confidence': 0.0,
                'metrics': {}}
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col.lower() in price_data.columns for col in
            required_columns):
            self.logger.error(
                f'Price data must contain {required_columns} columns.')
            return {'regime': MarketRegime.UNKNOWN, 'confidence': 0.0,
                'metrics': {}}
        price_data_copy = price_data.copy()
        price_data_copy.columns = [col.lower() for col in price_data_copy.
            columns]
        data = price_data_copy.iloc[-lookback_periods:].reset_index(drop=True)
        regime_metrics = self._calculate_regime_metrics(data, short_window=
            short_window, medium_window=medium_window, long_window=
            long_window, volatility_window=volatility_window, atr_window=
            atr_window)
        regime_result = self._classify_regime(regime_metrics,
            trend_strength_threshold=trend_strength_threshold,
            volatility_threshold=volatility_threshold, breakout_threshold=
            breakout_threshold)
        current_time = datetime.now()
        current_regime = regime_result.get('regime')
        current_confidence = regime_result.get('confidence')
        self.history.append({'timestamp': current_time, 'regime':
            current_regime, 'confidence': current_confidence})
        if len(self.history) >= 2:
            previous_entry = self.history[-2]
            previous_regime = previous_entry.get('regime')
            if previous_regime != current_regime:
                self.transition_history.append({'timestamp': current_time,
                    'from_regime': previous_regime, 'to_regime':
                    current_regime, 'from_confidence': previous_entry.get(
                    'confidence'), 'to_confidence': current_confidence,
                    'transition_metrics': {'volatility_ratio':
                    regime_result.get('metrics', {}).get('volatility_ratio'
                    ), 'adx': regime_result.get('metrics', {}).get('adx'),
                    'ma_alignment': regime_result.get('metrics', {}).get(
                    'ma_alignment')}})
                if len(self.transition_history) > 50:
                    self.transition_history = self.transition_history[-50:]
        return regime_result

    def _calculate_regime_metrics(self, data: pd.DataFrame, short_window:
        int=20, medium_window: int=50, long_window: int=100,
        volatility_window: int=20, atr_window: int=14) ->Dict[str, Any]:
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
        ma_short = close.rolling(window=short_window).mean()
        ma_medium = close.rolling(window=medium_window).mean()
        ma_long = close.rolling(window=long_window).mean()
        ma_short_slope = self._calculate_slope(ma_short, periods=5)
        ma_medium_slope = self._calculate_slope(ma_medium, periods=10)
        ma_long_slope = self._calculate_slope(ma_long, periods=20)
        ma_alignment = self._calculate_ma_alignment(ma_short, ma_medium,
            ma_long)
        std_dev = close.rolling(window=volatility_window).std()
        atr = self._calculate_atr(data, window=atr_window)
        norm_std_dev = std_dev / close * 100
        norm_atr = atr / close * 100
        recent_volatility = norm_std_dev.iloc[-5:].mean()
        historical_volatility = norm_std_dev.iloc[-volatility_window:-5].mean(
            ) if len(norm_std_dev) > volatility_window else norm_std_dev.mean()
        volatility_ratio = (recent_volatility / historical_volatility if 
            historical_volatility > 0 else 1.0)
        recent_range = self._calculate_price_range(data.iloc[-20:])
        historical_range = self._calculate_price_range(data.iloc[-
            lookback_periods:-20]) if len(data) > 20 else recent_range
        range_ratio = (recent_range / historical_range if historical_range >
            0 else 1.0)
        is_breakout, breakout_direction = self._detect_breakout(data, atr,
            window=20, threshold=2.0)
        adx = self._calculate_adx(data, window=14)
        recent_adx = adx.iloc[-1] if not adx.empty else 0
        momentum = self._calculate_momentum(close, periods=10)
        return {'ma_short': ma_short.iloc[-1] if not ma_short.empty else
            None, 'ma_medium': ma_medium.iloc[-1] if not ma_medium.empty else
            None, 'ma_long': ma_long.iloc[-1] if not ma_long.empty else
            None, 'ma_short_slope': ma_short_slope, 'ma_medium_slope':
            ma_medium_slope, 'ma_long_slope': ma_long_slope, 'ma_alignment':
            ma_alignment, 'std_dev': std_dev.iloc[-1] if not std_dev.empty else
            None, 'atr': atr.iloc[-1] if not atr.empty else None,
            'norm_std_dev': norm_std_dev.iloc[-1] if not norm_std_dev.empty
             else None, 'norm_atr': norm_atr.iloc[-1] if not norm_atr.empty
             else None, 'recent_volatility': recent_volatility,
            'historical_volatility': historical_volatility,
            'volatility_ratio': volatility_ratio, 'recent_range':
            recent_range, 'historical_range': historical_range,
            'range_ratio': range_ratio, 'is_breakout': is_breakout,
            'breakout_direction': breakout_direction, 'adx': recent_adx,
            'momentum': momentum}

    def _classify_regime(self, metrics: Dict[str, Any],
        trend_strength_threshold: float=25.0, volatility_threshold: float=
        1.5, breakout_threshold: float=2.0) ->Dict[str, Any]:
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
        regime_scores = {MarketRegime.TRENDING_UP: 0.0, MarketRegime.
            TRENDING_DOWN: 0.0, MarketRegime.RANGING: 0.0, MarketRegime.
            VOLATILE: 0.0, MarketRegime.CHOPPY: 0.0, MarketRegime.BREAKOUT: 0.0
            }
        adx = metrics.get('adx', 0)
        ma_alignment = metrics.get('ma_alignment', 0)
        if adx >= trend_strength_threshold:
            if ma_alignment > 0:
                regime_scores[MarketRegime.TRENDING_UP] += 0.6 + min((adx -
                    trend_strength_threshold) / 100, 0.3)
            elif ma_alignment < 0:
                regime_scores[MarketRegime.TRENDING_DOWN] += 0.6 + min((adx -
                    trend_strength_threshold) / 100, 0.3)
        range_ratio = metrics.get('range_ratio', 1.0)
        volatility_ratio = metrics.get('volatility_ratio', 1.0)
        if (adx < trend_strength_threshold and range_ratio < 1.2 and 
            volatility_ratio < 1.2):
            regime_scores[MarketRegime.RANGING] += 0.7
        if volatility_ratio > volatility_threshold:
            regime_scores[MarketRegime.VOLATILE] += 0.5 + min((
                volatility_ratio - volatility_threshold) / 2, 0.4)
        ma_short_slope = metrics.get('ma_short_slope', 0)
        ma_medium_slope = metrics.get('ma_medium_slope', 0)
        if abs(ma_medium_slope) < 0.05 and abs(ma_short_slope) > 0.1:
            regime_scores[MarketRegime.CHOPPY] += 0.6
        is_breakout = metrics.get('is_breakout', False)
        if is_breakout:
            regime_scores[MarketRegime.BREAKOUT] += 0.8
        max_regime = max(regime_scores, key=regime_scores.get)
        max_score = regime_scores[max_regime]
        if max_score < 0.4:
            max_regime = MarketRegime.UNKNOWN
            max_score = 0.3
        return {'regime': max_regime, 'confidence': max_score, 'sub_scores':
            regime_scores, 'metrics': metrics}

    def _calculate_slope(self, series: pd.Series, periods: int=5) ->float:
        """
        Calculate the slope of a time series over the specified periods.
        Returns slope as percentage change.
        """
        if len(series) < periods + 1 or series.iloc[-periods - 1:].isna().any(
            ):
            return 0
        start_value = series.iloc[-periods - 1]
        end_value = series.iloc[-1]
        if start_value == 0:
            return 0
        return (end_value / start_value - 1) * 100

    def _calculate_ma_alignment(self, ma_short: pd.Series, ma_medium: pd.
        Series, ma_long: pd.Series) ->float:
        """
        Calculate moving average alignment indicator.
        Positive values indicate bullish alignment (short > medium > long)
        Negative values indicate bearish alignment (short < medium < long)
        Values close to 0 indicate no clear alignment
        """
        if ma_short.isna().any() or ma_medium.isna().any() or ma_long.isna(
            ).any():
            return 0
        short_val = ma_short.iloc[-1]
        medium_val = ma_medium.iloc[-1]
        long_val = ma_long.iloc[-1]
        short_medium_diff = (short_val - medium_val) / medium_val
        medium_long_diff = (medium_val - long_val) / long_val
        alignment = (short_medium_diff + medium_long_diff) / 2
        return np.clip(alignment * 10, -1, 1)

    def _calculate_atr(self, data: pd.DataFrame, window: int=14) ->pd.Series:
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

    def _calculate_price_range(self, data: pd.DataFrame) ->float:
        """
        Calculate the average price range as percentage of price
        """
        if len(data) < 2:
            return 0
        high = data['high']
        low = data['low']
        close = data['close']
        ranges = (high - low) / close
        return ranges.mean() * 100

    def _detect_breakout(self, data: pd.DataFrame, atr: pd.Series, window:
        int=20, threshold: float=2.0) ->Tuple[bool, Optional[str]]:
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
        recent_data = data.iloc[-window - 1:-1]
        recent_high = recent_data['high'].max()
        recent_low = recent_data['low'].min()
        latest_price = data['close'].iloc[-1]
        latest_atr = atr.iloc[-1]
        threshold_value = latest_atr * threshold
        if latest_price > recent_high + threshold_value:
            return True, 'up'
        elif latest_price < recent_low - threshold_value:
            return True, 'down'
        else:
            return False, None

    def _calculate_adx(self, data: pd.DataFrame, window: int=14) ->pd.Series:
        """
        Calculate the Average Directional Index (ADX) - a measure of trend strength
        """
        if len(data) < window * 2:
            return pd.Series([])
        high = data['high']
        low = data['low']
        close = data['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        plus_dm = high.diff()
        minus_dm = low.diff() * -1
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
        tr_smoothed = tr.ewm(alpha=1 / window, adjust=False).mean()
        plus_dm_smoothed = plus_dm.ewm(alpha=1 / window, adjust=False).mean()
        minus_dm_smoothed = minus_dm.ewm(alpha=1 / window, adjust=False).mean()
        plus_di = 100 * plus_dm_smoothed / tr_smoothed
        minus_di = 100 * minus_dm_smoothed / tr_smoothed
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1 / window, adjust=False).mean()
        return adx

    def _calculate_momentum(self, series: pd.Series, periods: int=10) ->float:
        """
        Calculate momentum as the rate of change over the specified periods
        """
        if len(series) < periods + 1:
            return 0
        momentum = (series.iloc[-1] / series.iloc[-periods - 1] - 1) * 100
        return momentum

    @with_resilience('get_transition_history')
    def get_transition_history(self, limit: int=10) ->List[Dict[str, Any]]:
        """
        Get the history of regime transitions.

        Args:
            limit: Maximum number of transitions to return

        Returns:
            List of regime transition events
        """
        return self.transition_history[-limit:
            ] if self.transition_history else []

    @with_resilience('get_transition_frequency')
    def get_transition_frequency(self, lookback_days: int=30) ->Dict[str, Any]:
        """
        Calculate the frequency of regime transitions over a period.

        Args:
            lookback_days: Number of days to look back

        Returns:
            Dictionary with transition frequency statistics
        """
        if not self.transition_history:
            return {'total_transitions': 0, 'transitions_per_day': 0,
                'most_common_transition': None, 'transition_counts': {}}
        cutoff_time = datetime.now() - timedelta(days=lookback_days)
        recent_transitions = [t for t in self.transition_history if t[
            'timestamp'] >= cutoff_time]
        if not recent_transitions:
            return {'total_transitions': 0, 'transitions_per_day': 0,
                'most_common_transition': None, 'transition_counts': {}}
        transition_counts = {}
        for transition in recent_transitions:
            from_regime = transition['from_regime']
            to_regime = transition['to_regime']
            key = f'{from_regime} -> {to_regime}'
            if key not in transition_counts:
                transition_counts[key] = 0
            transition_counts[key] += 1
        most_common = max(transition_counts.items(), key=lambda x: x[1]
            ) if transition_counts else (None, 0)
        total_transitions = len(recent_transitions)
        transitions_per_day = total_transitions / lookback_days
        return {'total_transitions': total_transitions,
            'transitions_per_day': transitions_per_day,
            'most_common_transition': most_common[0], 'most_common_count':
            most_common[1], 'transition_counts': transition_counts}


class MarketRegimeService:
    """Service for detecting market regimes and maintaining regime history"""

    def __init__(self):
        """Initialize the market regime service"""
        self.detector = MarketRegimeAnalyzer()
        self.regime_history = {}
        self.logger = logging.getLogger(__name__)

    @with_exception_handling
    def detect_current_regime(self, symbol: str, timeframe: str, price_data:
        pd.DataFrame) ->Dict[str, Any]:
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
            result = self.detector.detect_regime(price_data)
            key = f'{symbol}_{timeframe}'
            if key not in self.regime_history:
                self.regime_history[key] = []
            self.regime_history[key].append({'timestamp': datetime.now(),
                'regime': result['regime'], 'confidence': result['confidence']}
                )
            if len(self.regime_history[key]) > 100:
                self.regime_history[key] = self.regime_history[key][-100:]
            return result
        except Exception as e:
            self.logger.error(f'Error detecting market regime: {str(e)}')
            return {'regime': MarketRegime.UNKNOWN, 'confidence': 0.0,
                'metrics': {}, 'error': str(e)}

    @with_resilience('get_regime_history')
    def get_regime_history(self, symbol: str, timeframe: str, limit: int=10
        ) ->List[Dict[str, Any]]:
        """
        Get historical regime data for a specific symbol and timeframe

        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            limit: Maximum number of historical entries to return

        Returns:
            List of historical regime entries
        """
        key = f'{symbol}_{timeframe}'
        history = self.regime_history.get(key, [])
        return history[-limit:]

    @with_resilience('get_dominant_regime')
    def get_dominant_regime(self, symbol: str, timeframe: str,
        lookback_periods: int=5) ->Dict[str, Any]:
        """
        Get the dominant market regime over recent history

        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            lookback_periods: Number of historical entries to consider

        Returns:
            Dictionary with dominant regime and confidence
        """
        history = self.get_regime_history(symbol, timeframe, lookback_periods)
        if not history:
            return {'regime': MarketRegime.UNKNOWN, 'confidence': 0.0}
        regime_counts = {}
        total_confidence = {}
        for entry in history:
            regime = entry['regime']
            confidence = entry['confidence']
            if regime not in regime_counts:
                regime_counts[regime] = 0
                total_confidence[regime] = 0
            regime_counts[regime] += 1
            total_confidence[regime] += confidence
        dominant_regime = max(regime_counts, key=regime_counts.get)
        avg_confidence = total_confidence[dominant_regime] / regime_counts[
            dominant_regime]
        return {'regime': dominant_regime, 'confidence': avg_confidence,
            'occurrence_rate': regime_counts[dominant_regime] / len(history)}

    @with_resilience('get_regime_transitions')
    def get_regime_transitions(self, symbol: str, timeframe: str, limit: int=10
        ) ->List[Dict[str, Any]]:
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

    @with_resilience('get_transition_statistics')
    def get_transition_statistics(self, symbol: str, timeframe: str,
        lookback_days: int=30) ->Dict[str, Any]:
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
        stats['symbol'] = symbol
        stats['timeframe'] = timeframe
        return stats
