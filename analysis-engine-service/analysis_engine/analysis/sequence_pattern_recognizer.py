"""
Sequence Pattern Recognizer Module

This module provides functionality for identifying complex patterns that span multiple timeframes,
which can improve prediction accuracy by detecting fractal patterns that repeat across timeframes.

Part of Phase 3 implementation to enhance sequence pattern recognition.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import math
from analysis_engine.caching.cache_service import cache_result
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class PatternType(Enum):
    """Types of sequence patterns that can be detected"""
    DOUBLE_TOP = 'double_top'
    DOUBLE_BOTTOM = 'double_bottom'
    HEAD_AND_SHOULDERS = 'head_and_shoulders'
    INVERSE_HEAD_AND_SHOULDERS = 'inverse_head_and_shoulders'
    ASCENDING_TRIANGLE = 'ascending_triangle'
    DESCENDING_TRIANGLE = 'descending_triangle'
    SYMMETRICAL_TRIANGLE = 'symmetrical_triangle'
    RISING_WEDGE = 'rising_wedge'
    FALLING_WEDGE = 'falling_wedge'
    CHANNEL_UP = 'channel_up'
    CHANNEL_DOWN = 'channel_down'
    FLAG = 'flag'
    PENNANT = 'pennant'
    CUP_AND_HANDLE = 'cup_and_handle'
    CUSTOM = 'custom'


class TimeframeLevel(Enum):
    """Timeframe levels for pattern detection"""
    MICRO = 'micro'
    SHORT = 'short'
    MEDIUM = 'medium'
    LONG = 'long'
    MACRO = 'macro'


class SequencePatternRecognizer:
    """
    SequencePatternRecognizer class.
    
    Attributes:
        Add attributes here
    """


    def __init__(self, timeframe_mapping: Optional[Dict[str, TimeframeLevel
        ]]=None, min_pattern_quality: float=0.7, use_ml_validation: bool=
        False, pattern_types: Optional[List[PatternType]]=None,
        model_retraining_service: Optional['ModelRetrainingService']=None):
        """
        Initialize the sequence pattern recognizer.

        Args:
            timeframe_mapping: Mapping of timeframe strings to timeframe levels
            min_pattern_quality: Minimum quality threshold for pattern detection
            use_ml_validation: Whether to use machine learning for pattern validation
            pattern_types: List of pattern types to detect (if None, detects all)
        """
        self.timeframe_mapping = timeframe_mapping or {'1m': TimeframeLevel
            .MICRO, '5m': TimeframeLevel.MICRO, '15m': TimeframeLevel.SHORT,
            '30m': TimeframeLevel.SHORT, '1h': TimeframeLevel.SHORT, '4h':
            TimeframeLevel.MEDIUM, '8h': TimeframeLevel.MEDIUM, '1d':
            TimeframeLevel.LONG, '1w': TimeframeLevel.LONG, '1M':
            TimeframeLevel.MACRO}
        self.min_pattern_quality = min_pattern_quality
        self.use_ml_validation = use_ml_validation
        self.pattern_types = pattern_types or list(PatternType)
        self.model_retraining_service = model_retraining_service
        self.logger = logging.getLogger(__name__)
        self.pattern_history = {}
        self.logger.info(
            f'SequencePatternRecognizer initialized with {len(self.pattern_types)} pattern types'
            )

    @cache_result(ttl=1800)
    @with_exception_handling
    def detect_patterns(self, symbol: str, price_data: Dict[str, pd.
        DataFrame], timeframes: Optional[List[str]]=None) ->Dict[str, Any]:
        """
        Detect patterns across multiple timeframes.

        Args:
            symbol: Trading symbol
            price_data: Dictionary mapping timeframes to price DataFrames
            timeframes: Optional list of timeframes to analyze (if None, uses all available)

        Returns:
            Dictionary with detected patterns and their properties
        """
        if not price_data:
            return {'error': 'No price data provided'}
        if timeframes:
            available_timeframes = [tf for tf in timeframes if tf in price_data
                ]
        else:
            available_timeframes = list(price_data.keys())
        if not available_timeframes:
            return {'error': 'No valid timeframes available for analysis'}
        timeframe_levels = {}
        for tf in available_timeframes:
            level = self.timeframe_mapping.get(tf, TimeframeLevel.MEDIUM)
            if level not in timeframe_levels:
                timeframe_levels[level] = []
            timeframe_levels[level].append(tf)
        individual_patterns = {}
        for tf in available_timeframes:
            if price_data[tf].empty:
                continue
            tf_patterns = self._detect_timeframe_patterns(price_data[tf], tf)
            if tf_patterns:
                individual_patterns[tf] = tf_patterns
        sequence_patterns = self._find_sequence_patterns(price_data,
            individual_patterns, timeframe_levels)
        validated_patterns = self._validate_patterns(sequence_patterns,
            price_data)
        final_patterns = self._calculate_confidence_scores(validated_patterns)
        self._update_pattern_history(final_patterns)
        result = {'timeframes_analyzed': available_timeframes,
            'individual_patterns': individual_patterns, 'sequence_patterns':
            final_patterns, 'pattern_count': len(final_patterns),
            'timestamp': datetime.now().isoformat()}
        if self.use_ml_validation and self.model_retraining_service:
            try:
                self.model_retraining_service.check_and_trigger_retraining(
                    'sequence_pattern_model')
            except Exception:
                self.logger.error(
                    'Failed to trigger ML retraining for sequence patterns',
                    exc_info=True)
        return result

    def _detect_timeframe_patterns(self, price_data: pd.DataFrame,
        timeframe: str) ->List[Dict[str, Any]]:
        """
        Detect patterns in a single timeframe.

        Args:
            price_data: Price DataFrame for a specific timeframe
            timeframe: The timeframe being analyzed

        Returns:
            List of detected patterns
        """
        patterns = []
        required_cols = ['open', 'high', 'low', 'close']
        available_cols = [col.lower() for col in price_data.columns]
        if not all(col in available_cols for col in required_cols):
            self.logger.warning(
                f'Missing required columns for pattern detection in {timeframe}'
                )
            return patterns
        col_mapping = {}
        for req_col in required_cols:
            for col in price_data.columns:
                if col.lower() == req_col:
                    col_mapping[req_col] = col
                    break
        open_prices = price_data[col_mapping['open']]
        high_prices = price_data[col_mapping['high']]
        low_prices = price_data[col_mapping['low']]
        close_prices = price_data[col_mapping['close']]
        for pattern_type in self.pattern_types:
            if pattern_type == PatternType.DOUBLE_TOP:
                detected = self._detect_double_top(high_prices, close_prices)
                if detected:
                    patterns.append({**detected, 'timeframe': timeframe})
            elif pattern_type == PatternType.DOUBLE_BOTTOM:
                detected = self._detect_double_bottom(low_prices, close_prices)
                if detected:
                    patterns.append({**detected, 'timeframe': timeframe})
            elif pattern_type == PatternType.HEAD_AND_SHOULDERS:
                detected = self._detect_head_and_shoulders(high_prices,
                    close_prices)
                if detected:
                    patterns.append({**detected, 'timeframe': timeframe})
            elif pattern_type == PatternType.INVERSE_HEAD_AND_SHOULDERS:
                detected = self._detect_inverse_head_and_shoulders(low_prices,
                    close_prices)
                if detected:
                    patterns.append({**detected, 'timeframe': timeframe})
            elif pattern_type == PatternType.ASCENDING_TRIANGLE:
                detected = self._detect_ascending_triangle(high_prices,
                    low_prices, close_prices)
                if detected:
                    patterns.append({**detected, 'timeframe': timeframe})
            elif pattern_type == PatternType.DESCENDING_TRIANGLE:
                detected = self._detect_descending_triangle(high_prices,
                    low_prices, close_prices)
                if detected:
                    patterns.append({**detected, 'timeframe': timeframe})
            elif pattern_type == PatternType.SYMMETRICAL_TRIANGLE:
                detected = self._detect_symmetrical_triangle(high_prices,
                    low_prices, close_prices)
                if detected:
                    patterns.append({**detected, 'timeframe': timeframe})
        return patterns

    def _detect_double_top(self, high_prices: pd.Series, close_prices: pd.
        Series) ->Optional[Dict[str, Any]]:
        """
        Detect double top pattern.

        Args:
            high_prices: Series of high prices
            close_prices: Series of close prices

        Returns:
            Pattern details or None if not detected
        """
        if len(high_prices) < 30:
            return None
        window_size = min(30, len(high_prices) - 1)
        peaks = []
        for i in range(5, window_size):
            if high_prices.iloc[-i] > high_prices.iloc[-i - 1
                ] and high_prices.iloc[-i] > high_prices.iloc[-i + 1]:
                if high_prices.iloc[-i] > high_prices.iloc[-i - 5:].mean(
                    ) * 1.005:
                    peaks.append((-i, high_prices.iloc[-i]))
        if len(peaks) < 2:
            return None
        peaks.sort(key=lambda x: x[1], reverse=True)
        if len(peaks) >= 2:
            peak1 = peaks[0]
            peak2 = peaks[1]
            height_diff_pct = abs(peak1[1] - peak2[1]) / peak1[1]
            if height_diff_pct <= 0.02:
                peak_indices = sorted([peak1[0], peak2[0]])
                trough_idx = min(range(peak_indices[0], peak_indices[1] + 1
                    ), key=lambda i: low_prices.iloc[i])
                quality = 1.0 - height_diff_pct
                pattern_completed = close_prices.iloc[-1] < close_prices.iloc[
                    trough_idx]
                return {'type': PatternType.DOUBLE_TOP.value, 'direction':
                    'bearish', 'peak1_idx': peak1[0], 'peak2_idx': peak2[0],
                    'peak1_value': peak1[1], 'peak2_value': peak2[1],
                    'trough_idx': trough_idx, 'trough_value': close_prices.
                    iloc[trough_idx], 'quality': quality, 'completed':
                    pattern_completed}
        return None

    def _detect_double_bottom(self, low_prices: pd.Series, close_prices: pd
        .Series) ->Optional[Dict[str, Any]]:
        """
        Detect double bottom pattern.

        Args:
            low_prices: Series of low prices
            close_prices: Series of close prices

        Returns:
            Pattern details or None if not detected
        """
        if len(low_prices) < 30:
            return None
        window_size = min(30, len(low_prices) - 1)
        troughs = []
        for i in range(5, window_size):
            if low_prices.iloc[-i] < low_prices.iloc[-i - 1
                ] and low_prices.iloc[-i] < low_prices.iloc[-i + 1]:
                if low_prices.iloc[-i] < low_prices.iloc[-i - 5:].mean(
                    ) * 0.995:
                    troughs.append((-i, low_prices.iloc[-i]))
        if len(troughs) < 2:
            return None
        troughs.sort(key=lambda x: x[1])
        if len(troughs) >= 2:
            trough1 = troughs[0]
            trough2 = troughs[1]
            height_diff_pct = abs(trough1[1] - trough2[1]) / trough1[1
                ] if trough1[1] != 0 else 1.0
            if height_diff_pct <= 0.02:
                trough_indices = sorted([trough1[0], trough2[0]])
                peak_idx = max(range(trough_indices[0], trough_indices[1] +
                    1), key=lambda i: high_prices.iloc[i])
                quality = 1.0 - height_diff_pct
                pattern_completed = close_prices.iloc[-1] > close_prices.iloc[
                    peak_idx]
                return {'type': PatternType.DOUBLE_BOTTOM.value,
                    'direction': 'bullish', 'trough1_idx': trough1[0],
                    'trough2_idx': trough2[0], 'trough1_value': trough1[1],
                    'trough2_value': trough2[1], 'peak_idx': peak_idx,
                    'peak_value': close_prices.iloc[peak_idx], 'quality':
                    quality, 'completed': pattern_completed}
        return None

    def _detect_head_and_shoulders(self, high_prices: pd.Series,
        close_prices: pd.Series) ->Optional[Dict[str, Any]]:
        """
        Detect head and shoulders pattern.

        Args:
            high_prices: Series of high prices
            close_prices: Series of close prices

        Returns:
            Pattern details or None if not detected
        """
        return None

    def _detect_inverse_head_and_shoulders(self, low_prices: pd.Series,
        close_prices: pd.Series) ->Optional[Dict[str, Any]]:
        """
        Detect inverse head and shoulders pattern.

        Args:
            low_prices: Series of low prices
            close_prices: Series of close prices

        Returns:
            Pattern details or None if not detected
        """
        return None

    def _detect_ascending_triangle(self, high_prices: pd.Series, low_prices:
        pd.Series, close_prices: pd.Series) ->Optional[Dict[str, Any]]:
        """
        Detect ascending triangle pattern.

        Args:
            high_prices: Series of high prices
            low_prices: Series of low prices
            close_prices: Series of close prices

        Returns:
            Pattern details or None if not detected
        """
        return None

    def _detect_descending_triangle(self, high_prices: pd.Series,
        low_prices: pd.Series, close_prices: pd.Series) ->Optional[Dict[str,
        Any]]:
        """
        Detect descending triangle pattern.

        Args:
            high_prices: Series of high prices
            low_prices: Series of low prices
            close_prices: Series of close prices

        Returns:
            Pattern details or None if not detected
        """
        return None

    def _detect_symmetrical_triangle(self, high_prices: pd.Series,
        low_prices: pd.Series, close_prices: pd.Series) ->Optional[Dict[str,
        Any]]:
        """
        Detect symmetrical triangle pattern.

        Args:
            high_prices: Series of high prices
            low_prices: Series of low prices
            close_prices: Series of close prices

        Returns:
            Pattern details or None if not detected
        """
        return None

    def _find_sequence_patterns(self, price_data: Dict[str, pd.DataFrame],
        individual_patterns: Dict[str, List[Dict[str, Any]]],
        timeframe_levels: Dict[TimeframeLevel, List[str]]) ->List[Dict[str,
        Any]]:
        """
        Find patterns that form sequences across multiple timeframes.

        Args:
            price_data: Dictionary mapping timeframes to price DataFrames
            individual_patterns: Dictionary mapping timeframes to detected patterns
            timeframe_levels: Dictionary mapping timeframe levels to timeframes

        Returns:
            List of sequence patterns
        """
        sequence_patterns = []
        pattern_types = defaultdict(list)
        for tf, patterns in individual_patterns.items():
            for pattern in patterns:
                pattern_types[pattern['type']].append({**pattern,
                    'timeframe': tf})
        for pattern_type, patterns in pattern_types.items():
            patterns_by_level = defaultdict(list)
            for pattern in patterns:
                tf = pattern['timeframe']
                level = None
                for l, timeframes in timeframe_levels.items():
                    if tf in timeframes:
                        level = l
                        break
                if level:
                    patterns_by_level[level].append(pattern)
            if len(patterns_by_level) >= 2:
                for level1, patterns1 in patterns_by_level.items():
                    for level2, patterns2 in patterns_by_level.items():
                        if level1 != level2:
                            for pattern1 in patterns1:
                                for pattern2 in patterns2:
                                    if pattern1.get('direction'
                                        ) == pattern2.get('direction'):
                                        sequence = {'type': pattern_type,
                                            'direction': pattern1.get(
                                            'direction'), 'levels': [level1.
                                            value, level2.value], 'timeframes':
                                            [pattern1['timeframe'], pattern2[
                                            'timeframe']], 'patterns': [
                                            pattern1, pattern2], 'quality': (
                                            pattern1.get('quality', 0.5) +
                                            pattern2.get('quality', 0.5)) / 2}
                                        sequence_patterns.append(sequence)
        return sequence_patterns

    def _validate_patterns(self, sequence_patterns: List[Dict[str, Any]],
        price_data: Dict[str, pd.DataFrame]) ->List[Dict[str, Any]]:
        """
        Validate detected sequence patterns using statistical methods.

        Args:
            sequence_patterns: List of detected sequence patterns
            price_data: Dictionary mapping timeframes to price DataFrames

        Returns:
            List of validated patterns
        """
        validated_patterns = []
        for pattern in sequence_patterns:
            if pattern.get('quality', 0) < self.min_pattern_quality:
                continue
            validation_score = self._calculate_validation_score(pattern,
                price_data)
            if validation_score >= 0.6:
                validated_patterns.append({**pattern, 'validation_score':
                    validation_score})
        return validated_patterns

    def _calculate_validation_score(self, pattern: Dict[str, Any],
        price_data: Dict[str, pd.DataFrame]) ->float:
        """
        Calculate a validation score for a pattern.

        Args:
            pattern: The pattern to validate
            price_data: Dictionary mapping timeframes to price DataFrames

        Returns:
            Validation score (0.0 to 1.0)
        """
        base_score = pattern.get('quality', 0.5)
        completion_bonus = 0.0
        for p in pattern.get('patterns', []):
            if p.get('completed', False):
                completion_bonus += 0.1
        alignment_bonus = 0.0
        if len(pattern.get('timeframes', [])) >= 2:
            alignment_bonus = 0.1
        type_reliability = {PatternType.DOUBLE_TOP.value: 0.8, PatternType.
            DOUBLE_BOTTOM.value: 0.8, PatternType.HEAD_AND_SHOULDERS.value:
            0.7, PatternType.INVERSE_HEAD_AND_SHOULDERS.value: 0.7,
            PatternType.ASCENDING_TRIANGLE.value: 0.75, PatternType.
            DESCENDING_TRIANGLE.value: 0.75, PatternType.
            SYMMETRICAL_TRIANGLE.value: 0.65}
        reliability_factor = type_reliability.get(pattern.get('type'), 0.5)
        validation_score = (base_score * 0.6 + completion_bonus +
            alignment_bonus + reliability_factor * 0.2)
        return max(0.0, min(1.0, validation_score))

    def _calculate_confidence_scores(self, patterns: List[Dict[str, Any]]
        ) ->List[Dict[str, Any]]:
        """
        Calculate confidence scores for validated patterns.

        Args:
            patterns: List of validated patterns

        Returns:
            List of patterns with confidence scores
        """
        if not patterns:
            return []
        for pattern in patterns:
            quality = pattern.get('quality', 0.5)
            validation = pattern.get('validation_score', 0.5)
            confidence = quality * 0.4 + validation * 0.6
            timeframe_count = len(pattern.get('timeframes', []))
            if timeframe_count > 1:
                confidence = min(1.0, confidence * (1.0 + (timeframe_count -
                    1) * 0.1))
            pattern['confidence'] = confidence
        return sorted(patterns, key=lambda p: p.get('confidence', 0),
            reverse=True)

    def _update_pattern_history(self, patterns: List[Dict[str, Any]]) ->None:
        """
        Update the history of detected patterns.

        Args:
            patterns: List of detected patterns
        """
        timestamp = datetime.now()
        for pattern in patterns:
            pattern_type = pattern.get('type')
            if pattern_type not in self.pattern_history:
                self.pattern_history[pattern_type] = []
            self.pattern_history[pattern_type].append({'timestamp':
                timestamp, 'pattern': pattern})
            if len(self.pattern_history[pattern_type]) > 100:
                self.pattern_history[pattern_type] = self.pattern_history[
                    pattern_type][-100:]

    @with_resilience('get_pattern_history')
    def get_pattern_history(self, pattern_type: Optional[str]=None,
        lookback_hours: Optional[int]=None) ->Dict[str, List[Dict[str, Any]]]:
        """
        Get the history of detected patterns.

        Args:
            pattern_type: Optional pattern type to filter by
            lookback_hours: Optional number of hours to look back

        Returns:
            Dictionary mapping pattern types to lists of historical patterns
        """
        if pattern_type and pattern_type in self.pattern_history:
            history = {pattern_type: self.pattern_history[pattern_type]}
        else:
            history = self.pattern_history
        if lookback_hours is not None:
            cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
            filtered_history = {}
            for pt, patterns in history.items():
                filtered_history[pt] = [p for p in patterns if p[
                    'timestamp'] >= cutoff_time]
            return filtered_history
        return history
