"""
Multi-Timeframe Analysis Module

This module provides tools for analyzing forex price data across multiple timeframes,
identifying alignment, and confirming signals through timeframe confluence.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from analysis_engine.analysis.advanced_ta.base import AdvancedAnalysisBase, ConfidenceLevel, MarketDirection, AnalysisTimeframe
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class TimeframeAlignment(Enum):
    """Possible alignments across timeframes"""
    STRONGLY_BULLISH = 'strongly_bullish'
    WEAKLY_BULLISH = 'weakly_bullish'
    MIXED = 'mixed'
    WEAKLY_BEARISH = 'weakly_bearish'
    STRONGLY_BEARISH = 'strongly_bearish'


class MultiTimeFrameAnalyzer(AdvancedAnalysisBase):
    """
    Multi-timeframe analysis for forex price data

    This class analyzes indicators and patterns across multiple timeframes
    to identify alignment and confirm trading signals.
    Optimized for performance with caching and parallel processing.
    """
    _analysis_cache = {}
    _cache_timestamps = {}
    _cache_max_size = 50

    def __init__(self, timeframes: List[str]=None, indicators: List[str]=
        None, enable_caching: bool=True, cache_ttl: int=300,
        use_parallel_processing: bool=True):
        """
        Initialize the multi-timeframe analyzer with optimized performance

        Args:
            timeframes: List of timeframes to analyze (e.g., ['1h', '4h', '1d'])
            indicators: List of indicators to use in the analysis
            enable_caching: If True, cache analysis results for improved performance
            cache_ttl: Time-to-live for cached results in seconds
            use_parallel_processing: If True, use parallel processing for better performance
        """
        default_timeframes = ['5m', '15m', '1h', '4h', '1d']
        default_indicators = ['ma', 'rsi', 'macd']
        parameters = {'timeframes': timeframes or default_timeframes,
            'indicators': indicators or default_indicators,
            'weight_higher_timeframes': True, 'alignment_threshold': 0.7,
            'enable_caching': enable_caching, 'cache_ttl': cache_ttl,
            'use_parallel_processing': use_parallel_processing}
        super().__init__('Multi-Timeframe Analysis', parameters)
        self.sorted_timeframes = sorted(self.parameters['timeframes'], key=
            lambda x: self._timeframe_to_minutes(x))
        self.indicator_analyzers = {'ma': self._analyze_moving_averages,
            'rsi': self._analyze_rsi, 'macd': self._analyze_macd}

    def analyze(self, data_dict: Dict[str, pd.DataFrame]) ->Dict[str, Any]:
        """
        Analyze price data across multiple timeframes with optimized performance

        Args:
            data_dict: Dictionary mapping timeframe strings to DataFrames

        Returns:
            Analysis results
        """
        if not data_dict:
            return {'error': 'No data provided for multi-timeframe analysis'}
        missing_timeframes = [tf for tf in self.parameters['timeframes'] if
            tf not in data_dict]
        if missing_timeframes:
            return {'error':
                f'Missing data for timeframes: {missing_timeframes}',
                'available': list(data_dict.keys())}
        if self.parameters['enable_caching']:
            cache_key = self._get_cache_key(data_dict)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result
        results = {'timeframes_analyzed': list(data_dict.keys()),
            'indicator_signals': {}, 'trend_alignment': {},
            'overall_alignment': None, 'confidence_level': None}
        if self.parameters['use_parallel_processing']:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                indicator_tasks = {indicator: executor.submit(self.
                    _analyze_indicator, data_dict, indicator) for indicator in
                    self.parameters['indicators']}
                for indicator, task in indicator_tasks.items():
                    results['indicator_signals'][indicator] = task.result()
                trend_task = executor.submit(self.
                    _determine_trend_alignment, data_dict)
                results['trend_alignment'] = trend_task.result()
        else:
            for indicator in self.parameters['indicators']:
                indicator_results = self._analyze_indicator(data_dict,
                    indicator)
                results['indicator_signals'][indicator] = indicator_results
            results['trend_alignment'] = self._determine_trend_alignment(
                data_dict)
        results['overall_alignment'], results['confidence_level'
            ] = self._calculate_overall_alignment(results)
        results['bullish_signal'] = results['overall_alignment'] in [
            TimeframeAlignment.STRONGLY_BULLISH, TimeframeAlignment.
            WEAKLY_BULLISH]
        results['bearish_signal'] = results['overall_alignment'] in [
            TimeframeAlignment.STRONGLY_BEARISH, TimeframeAlignment.
            WEAKLY_BEARISH]
        results['confirmation_strength'
            ] = self._calculate_confirmation_strength(results)
        if self.parameters['enable_caching']:
            self._add_to_cache(cache_key, results)
        return results

    def _analyze_indicator(self, data_dict: Dict[str, pd.DataFrame],
        indicator: str) ->Dict[str, Any]:
        """
        Analyze a specific indicator across timeframes

        Args:
            data_dict: Dictionary mapping timeframe strings to DataFrames
            indicator: Indicator to analyze (e.g., 'ma', 'rsi')

        Returns:
            Dictionary with indicator analysis results
        """
        if indicator in self.indicator_analyzers:
            return self.indicator_analyzers[indicator](data_dict)
        else:
            return {}

    def _analyze_moving_averages(self, data_dict: Dict[str, pd.DataFrame]
        ) ->Dict[str, Any]:
        """
        Analyze moving averages across timeframes

        Args:
            data_dict: Dictionary mapping timeframe strings to DataFrames

        Returns:
            Dictionary with MA analysis results
        """
        ma_results = {}
        fast_ma_period = 10
        slow_ma_period = 50
        for timeframe, df in data_dict.items():
            if timeframe not in self.parameters['timeframes']:
                continue
            fast_ma_col = f'ma_{fast_ma_period}'
            slow_ma_col = f'ma_{slow_ma_period}'
            if fast_ma_col not in df.columns:
                df[fast_ma_col] = df['close'].rolling(window=fast_ma_period
                    ).mean()
            if slow_ma_col not in df.columns:
                df[slow_ma_col] = df['close'].rolling(window=slow_ma_period
                    ).mean()
            latest_fast = df[fast_ma_col].iloc[-1]
            latest_slow = df[slow_ma_col].iloc[-1]
            latest_close = df['close'].iloc[-1]
            if latest_fast > latest_slow:
                trend = 'bullish'
                if latest_close > latest_fast:
                    strength = 'strong'
                else:
                    strength = 'moderate'
            else:
                trend = 'bearish'
                if latest_close < latest_fast:
                    strength = 'strong'
                else:
                    strength = 'moderate'
            ma_distance_pct = abs(latest_fast - latest_slow
                ) / latest_slow * 100
            crossovers = []
            for i in range(max(5, len(df) - 10), len(df) - 1):
                prev_diff = df[fast_ma_col].iloc[i - 1] - df[slow_ma_col].iloc[
                    i - 1]
                curr_diff = df[fast_ma_col].iloc[i] - df[slow_ma_col].iloc[i]
                if prev_diff <= 0 and curr_diff > 0:
                    crossovers.append({'type': 'bullish', 'index': i})
                elif prev_diff >= 0 and curr_diff < 0:
                    crossovers.append({'type': 'bearish', 'index': i})
            ma_results[timeframe] = {'trend': trend, 'strength': strength,
                'fast_value': latest_fast, 'slow_value': latest_slow,
                'ma_distance_pct': ma_distance_pct, 'recent_crossovers':
                crossovers}
        return ma_results

    def _analyze_rsi(self, data_dict: Dict[str, pd.DataFrame]) ->Dict[str, Any
        ]:
        """
        Analyze RSI across timeframes

        Args:
            data_dict: Dictionary mapping timeframe strings to DataFrames

        Returns:
            Dictionary with RSI analysis results
        """
        rsi_results = {}
        rsi_period = 14
        for timeframe, df in data_dict.items():
            if timeframe not in self.parameters['timeframes']:
                continue
            rsi_col = f'rsi_{rsi_period}'
            if rsi_col not in df.columns:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=rsi_period).mean()
                avg_loss = loss.rolling(window=rsi_period).mean()
                rs = avg_gain / avg_loss
                df[rsi_col] = 100 - 100 / (1 + rs)
            latest_rsi = df[rsi_col].iloc[-1]
            if latest_rsi > 70:
                trend = 'bearish'
                strength = 'strong' if latest_rsi > 80 else 'moderate'
            elif latest_rsi < 30:
                trend = 'bullish'
                strength = 'strong' if latest_rsi < 20 else 'moderate'
            elif latest_rsi > 50:
                trend = 'bullish'
                strength = 'weak'
            else:
                trend = 'bearish'
                strength = 'weak'
            price_direction = 'up' if df['close'].iloc[-1] > df['close'].iloc[
                -5] else 'down'
            rsi_direction = 'up' if df[rsi_col].iloc[-1] > df[rsi_col].iloc[-5
                ] else 'down'
            divergence = None
            if price_direction != rsi_direction:
                divergence = ('bullish' if price_direction == 'down' and 
                    rsi_direction == 'up' else 'bearish')
            rsi_results[timeframe] = {'value': latest_rsi, 'trend': trend,
                'strength': strength, 'condition': 'overbought' if 
                latest_rsi > 70 else 'oversold' if latest_rsi < 30 else
                'neutral', 'divergence': divergence}
        return rsi_results

    def _analyze_macd(self, data_dict: Dict[str, pd.DataFrame]) ->Dict[str, Any
        ]:
        """
        Analyze MACD across timeframes

        Args:
            data_dict: Dictionary mapping timeframe strings to DataFrames

        Returns:
            Dictionary with MACD analysis results
        """
        macd_results = {}
        fast_period = 12
        slow_period = 26
        signal_period = 9
        for timeframe, df in data_dict.items():
            if timeframe not in self.parameters['timeframes']:
                continue
            macd_col = 'macd'
            signal_col = 'macd_signal'
            hist_col = 'macd_hist'
            if macd_col not in df.columns:
                ema_fast = df['close'].ewm(span=fast_period, adjust=False
                    ).mean()
                ema_slow = df['close'].ewm(span=slow_period, adjust=False
                    ).mean()
                df[macd_col] = ema_fast - ema_slow
                df[signal_col] = df[macd_col].ewm(span=signal_period,
                    adjust=False).mean()
                df[hist_col] = df[macd_col] - df[signal_col]
            latest_macd = df[macd_col].iloc[-1]
            latest_signal = df[signal_col].iloc[-1]
            latest_hist = df[hist_col].iloc[-1]
            if latest_macd > latest_signal:
                trend = 'bullish'
            else:
                trend = 'bearish'
            if abs(latest_hist) > abs(latest_macd * 0.1):
                strength = 'strong'
            else:
                strength = 'moderate'
            zero_cross = None
            if df[macd_col].iloc[-2] < 0 and latest_macd >= 0:
                zero_cross = 'bullish'
            elif df[macd_col].iloc[-2] > 0 and latest_macd <= 0:
                zero_cross = 'bearish'
            signal_cross = None
            if df[macd_col].iloc[-2] < df[signal_col].iloc[-2
                ] and latest_macd > latest_signal:
                signal_cross = 'bullish'
            elif df[macd_col].iloc[-2] > df[signal_col].iloc[-2
                ] and latest_macd < latest_signal:
                signal_cross = 'bearish'
            macd_results[timeframe] = {'macd_value': latest_macd,
                'signal_value': latest_signal, 'histogram': latest_hist,
                'trend': trend, 'strength': strength, 'zero_cross':
                zero_cross, 'signal_cross': signal_cross}
        return macd_results

    def _determine_trend_alignment(self, data_dict: Dict[str, pd.DataFrame]
        ) ->Dict[str, str]:
        """
        Determine trend alignment across timeframes

        Args:
            data_dict: Dictionary mapping timeframe strings to DataFrames

        Returns:
            Dictionary with trend direction for each timeframe
        """
        trend_alignment = {}
        for timeframe, df in data_dict.items():
            if timeframe not in self.parameters['timeframes']:
                continue
            ma50_col = 'ma_50'
            if ma50_col not in df.columns:
                df[ma50_col] = df['close'].rolling(window=50).mean()
            latest_close = df['close'].iloc[-1]
            latest_ma50 = df[ma50_col].iloc[-1]
            if latest_close > latest_ma50:
                trend_direction = 'bullish'
            else:
                trend_direction = 'bearish'
            distance_pct = abs(latest_close - latest_ma50) / latest_ma50 * 100
            if distance_pct > 2.0:
                strength = 'strong'
            elif distance_pct > 0.5:
                strength = 'moderate'
            else:
                strength = 'weak'
            trend_alignment[timeframe] = {'direction': trend_direction,
                'strength': strength, 'distance_pct': distance_pct}
        return trend_alignment

    def _calculate_overall_alignment(self, results: Dict[str, Any]) ->Tuple[
        TimeframeAlignment, ConfidenceLevel]:
        """
        Calculate overall timeframe alignment and confidence

        Args:
            results: Analysis results dictionary

        Returns:
            Tuple of (alignment, confidence_level)
        """
        bullish_count = 0
        bearish_count = 0
        total_count = 0
        timeframe_weights = {}
        timeframes_sorted = sorted(self.parameters['timeframes'], key=lambda
            x: self._timeframe_to_minutes(x))
        if self.parameters['weight_higher_timeframes']:
            for i, tf in enumerate(timeframes_sorted):
                timeframe_weights[tf] = (i + 1) / len(timeframes_sorted)
        else:
            for tf in timeframes_sorted:
                timeframe_weights[tf] = 1.0
        weight_sum = sum(timeframe_weights.values())
        for tf in timeframe_weights:
            timeframe_weights[tf] /= weight_sum
        alignment_score = 0.0
        for timeframe, data in results['trend_alignment'].items():
            if timeframe in timeframe_weights:
                weight = timeframe_weights[timeframe]
                direction = data['direction']
                strength = data['strength']
                strength_multiplier = 1.0
                if strength == 'strong':
                    strength_multiplier = 1.0
                elif strength == 'moderate':
                    strength_multiplier = 0.7
                else:
                    strength_multiplier = 0.4
                if direction == 'bullish':
                    alignment_score += weight * strength_multiplier
                else:
                    alignment_score -= weight * strength_multiplier
        if alignment_score > 0.7:
            alignment = TimeframeAlignment.STRONGLY_BULLISH
        elif alignment_score > 0.3:
            alignment = TimeframeAlignment.WEAKLY_BULLISH
        elif alignment_score < -0.7:
            alignment = TimeframeAlignment.STRONGLY_BEARISH
        elif alignment_score < -0.3:
            alignment = TimeframeAlignment.WEAKLY_BEARISH
        else:
            alignment = TimeframeAlignment.MIXED
        abs_score = abs(alignment_score)
        if abs_score > 0.8:
            confidence = ConfidenceLevel.VERY_HIGH
        elif abs_score > 0.6:
            confidence = ConfidenceLevel.HIGH
        elif abs_score > 0.4:
            confidence = ConfidenceLevel.MEDIUM
        else:
            confidence = ConfidenceLevel.LOW
        return alignment, confidence

    def _calculate_confirmation_strength(self, results: Dict[str, Any]
        ) ->float:
        """
        Calculate the confirmation strength as a percentage

        Args:
            results: Analysis results dictionary

        Returns:
            Confirmation strength from 0.0 to 1.0
        """
        if results['overall_alignment'] == TimeframeAlignment.STRONGLY_BULLISH:
            base_strength = 1.0
        elif results['overall_alignment'] == TimeframeAlignment.WEAKLY_BULLISH:
            base_strength = 0.7
        elif results['overall_alignment'
            ] == TimeframeAlignment.STRONGLY_BEARISH:
            base_strength = 1.0
        elif results['overall_alignment'] == TimeframeAlignment.WEAKLY_BEARISH:
            base_strength = 0.7
        else:
            base_strength = 0.3
        indicator_agreement = 0.0
        for indicator, results_by_tf in results['indicator_signals'].items():
            bullish_count = 0
            bearish_count = 0
            total_count = 0
            for tf, indicator_data in results_by_tf.items():
                trend = indicator_data.get('trend', '')
                if trend == 'bullish':
                    bullish_count += 1
                elif trend == 'bearish':
                    bearish_count += 1
                total_count += 1
            if total_count > 0:
                max_alignment = max(bullish_count, bearish_count) / total_count
                indicator_agreement += max_alignment
        if results['indicator_signals']:
            indicator_agreement /= len(results['indicator_signals'])
            final_strength = base_strength * 0.7 + indicator_agreement * 0.3
        else:
            final_strength = base_strength
        return min(1.0, final_strength)

    @with_exception_handling
    def _timeframe_to_minutes(self, timeframe: str) ->int:
        """
        Convert timeframe string to minutes

        Args:
            timeframe: Timeframe string like '1h', '4h', '1d'

        Returns:
            Minutes representation
        """
        if not timeframe:
            return 0
        if timeframe[-1].isdigit():
            return int(timeframe)
        try:
            num = int(''.join([c for c in timeframe if c.isdigit()]))
            unit = timeframe[-1].lower()
            if unit == 'm':
                return num
            elif unit == 'h':
                return num * 60
            elif unit == 'd':
                return num * 60 * 24
            elif unit == 'w':
                return num * 60 * 24 * 7
            else:
                return num
        except:
            return 0

    def _get_cache_key(self, data_dict: Dict[str, pd.DataFrame]) ->str:
        """
        Generate a cache key for the given data.

        Args:
            data_dict: Dictionary of DataFrames by timeframe

        Returns:
            Cache key string
        """
        key_parts = []
        for tf in sorted(data_dict.keys()):
            df = data_dict[tf]
            if len(df) > 0:
                last_idx = min(5, len(df))
                last_timestamps = [ts.isoformat() for ts in df.index[-
                    last_idx:]]
                last_values = df['close'].iloc[-last_idx:].round(4).tolist()
                tf_key = (
                    f'{tf}_{len(df)}_{last_timestamps[-1]}_{last_values[-1]}')
                key_parts.append(tf_key)
        indicators_key = '_'.join(sorted(self.parameters['indicators']))
        params_key = (
            f"w{int(self.parameters['weight_higher_timeframes'])}_t{self.parameters['alignment_threshold']}"
            )
        return f"mta_{indicators_key}_{params_key}_{'_'.join(key_parts)}"

    def _get_from_cache(self, cache_key: str) ->Optional[Dict[str, Any]]:
        """
        Get a result from the cache if available and not expired.

        Args:
            cache_key: Cache key

        Returns:
            Cached result if available, None otherwise
        """
        if not self.parameters['enable_caching']:
            return None
        if cache_key in self._analysis_cache:
            timestamp = self._cache_timestamps.get(cache_key)
            if timestamp:
                current_time = pd.Timestamp.now()
                if (current_time - timestamp).total_seconds(
                    ) < self.parameters['cache_ttl']:
                    return self._analysis_cache[cache_key]
        return None

    def _add_to_cache(self, cache_key: str, result: Dict[str, Any]) ->None:
        """
        Add a result to the cache.

        Args:
            cache_key: Cache key
            result: Result to cache
        """
        if not self.parameters['enable_caching']:
            return
        self._analysis_cache[cache_key] = result
        self._cache_timestamps[cache_key] = pd.Timestamp.now()
        if len(self._analysis_cache) > self._cache_max_size:
            self._cleanup_cache()

    def _cleanup_cache(self) ->None:
        """Clean up the cache by removing the oldest entries."""
        sorted_keys = sorted(self._cache_timestamps.keys(), key=lambda k:
            self._cache_timestamps[k])
        keys_to_remove = sorted_keys[:len(sorted_keys) // 2]
        for key in keys_to_remove:
            if key in self._analysis_cache:
                del self._analysis_cache[key]
            if key in self._cache_timestamps:
                del self._cache_timestamps[key]

    @with_resilience('update_incremental')
    def update_incremental(self, data_dict: Dict[str, pd.DataFrame],
        previous_results: Dict[str, Any]) ->Dict[str, Any]:
        """
        Update analysis incrementally with new data

        Args:
            data_dict: Dictionary mapping timeframe strings to DataFrames with new data
            previous_results: Results from previous analysis

        Returns:
            Updated analysis results
        """
        if (previous_results and 'timeframes_analyzed' in previous_results and
            set(previous_results['timeframes_analyzed']) == set(data_dict.
            keys())):
            can_update_incrementally = True
            for tf in data_dict.keys():
                if tf in previous_results.get('last_analyzed_indices', {}):
                    last_idx = previous_results['last_analyzed_indices'][tf]
                    current_len = len(data_dict[tf])
                    if current_len - last_idx > 0.05 * current_len:
                        can_update_incrementally = False
                        break
                else:
                    can_update_incrementally = False
                    break
            if can_update_incrementally:
                updated_results = self._update_analysis(data_dict,
                    previous_results)
                updated_results['last_analyzed_indices'] = {tf: len(df) for
                    tf, df in data_dict.items()}
                return updated_results
        results = self.analyze(data_dict)
        results['last_analyzed_indices'] = {tf: len(df) for tf, df in
            data_dict.items()}
        return results

    def _update_analysis(self, data_dict: Dict[str, pd.DataFrame],
        previous_results: Dict[str, Any]) ->Dict[str, Any]:
        """
        Update analysis results with new data incrementally.

        Args:
            data_dict: Dictionary mapping timeframe strings to DataFrames with new data
            previous_results: Results from previous analysis

        Returns:
            Updated analysis results
        """
        updated_results = previous_results.copy()
        for indicator in self.parameters['indicators']:
            if indicator in self.indicator_analyzers:
                updated_indicator = self.indicator_analyzers[indicator](
                    data_dict)
                updated_results['indicator_signals'][indicator
                    ] = updated_indicator
        updated_results['trend_alignment'] = self._determine_trend_alignment(
            data_dict)
        updated_results['overall_alignment'], updated_results[
            'confidence_level'] = self._calculate_overall_alignment(
            updated_results)
        updated_results['bullish_signal'] = updated_results['overall_alignment'
            ] in [TimeframeAlignment.STRONGLY_BULLISH, TimeframeAlignment.
            WEAKLY_BULLISH]
        updated_results['bearish_signal'] = updated_results['overall_alignment'
            ] in [TimeframeAlignment.STRONGLY_BEARISH, TimeframeAlignment.
            WEAKLY_BEARISH]
        updated_results['confirmation_strength'
            ] = self._calculate_confirmation_strength(updated_results)
        return updated_results
