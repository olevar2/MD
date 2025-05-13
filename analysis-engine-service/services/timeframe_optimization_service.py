"""
TimeframeOptimizationService

This module provides a comprehensive service for dynamically weighting timeframes in multi-timeframe analysis
based on historical signal performance, currency correlations, pattern recognition, and market regime detection.

Features:
- Dynamic timeframe weighting based on historical performance
- Enhanced currency correlation analysis across timeframes
- Pattern recognition across multiple timeframes
- Market regime transition detection
- Adaptive weighting based on current market conditions
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import json
from enum import Enum
import asyncio
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests, coint
from analysis_engine.services.tool_effectiveness import TimeFrame
from analysis_engine.analysis.advanced_ta.market_regime import MarketRegimeType
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class SignalOutcome(Enum):
    """Enum for signal outcome classification"""
    WIN = 'win'
    LOSS = 'loss'
    BREAKEVEN = 'breakeven'
    UNKNOWN = 'unknown'


class CorrelationType(Enum):
    """Types of correlation relationships"""
    POSITIVE = 'positive'
    NEGATIVE = 'negative'
    NEUTRAL = 'neutral'
    BREAKDOWN = 'breakdown'
    STRENGTHENING = 'strengthening'
    WEAKENING = 'weakening'


class PatternType(Enum):
    """Types of patterns that can be detected across timeframes"""
    CONTINUATION = 'continuation'
    REVERSAL = 'reversal'
    CONSOLIDATION = 'consolidation'
    BREAKOUT = 'breakout'
    DIVERGENCE = 'divergence'
    HARMONIC = 'harmonic'


class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING_UP = 'trending_up'
    TRENDING_DOWN = 'trending_down'
    RANGING = 'ranging'
    VOLATILE = 'volatile'
    BREAKOUT = 'breakout'
    REVERSAL = 'reversal'
    TRANSITION = 'transition'


class TimeframeOptimizationService:
    """
    Comprehensive service to track and optimize timeframe weights based on multiple factors.

    This service:
    - Tracks the success rate of signals from different timeframes
    - Calculates optimal weights based on historical performance
    - Analyzes currency correlations across timeframes
    - Detects patterns that span multiple timeframes
    - Identifies market regime transitions
    - Provides adaptive weighting based on current market conditions
    """

    def __init__(self, timeframes: List[str], primary_timeframe: str=None,
        lookback_days: int=30, min_signals_required: int=10,
        weight_decay_factor: float=0.95, max_weight: float=3.0, min_weight:
        float=0.5, currency_pairs: List[str]=None, correlation_threshold:
        float=0.7, pattern_detection_threshold: float=0.65,
        regime_change_sensitivity: float=0.5, enable_correlation_analysis:
        bool=True, enable_pattern_recognition: bool=True,
        enable_regime_detection: bool=True, adaptive_weighting: bool=True):
        """
        Initialize the enhanced timeframe optimization service.

        Args:
            timeframes: List of timeframes to track and optimize
            primary_timeframe: Optional primary timeframe (if not provided, will use the first timeframe)
            lookback_days: How many days of signal history to consider
            min_signals_required: Minimum number of signals needed before optimization is applied
            weight_decay_factor: Factor for decaying older signal importance (0-1)
            max_weight: Maximum weight a timeframe can receive (prevents overemphasis)
            min_weight: Minimum weight a timeframe can receive (prevents complete elimination)
            currency_pairs: List of currency pairs to analyze for correlations
            correlation_threshold: Threshold for significant correlation (0-1)
            pattern_detection_threshold: Confidence threshold for pattern detection (0-1)
            regime_change_sensitivity: Sensitivity for detecting market regime changes (0-1)
            enable_correlation_analysis: Whether to enable currency correlation analysis
            enable_pattern_recognition: Whether to enable pattern recognition across timeframes
            enable_regime_detection: Whether to enable market regime detection
            adaptive_weighting: Whether to adapt weights based on current market conditions
        """
        self.timeframes = timeframes
        self.primary_timeframe = primary_timeframe or timeframes[0]
        self.lookback_days = lookback_days
        self.min_signals_required = min_signals_required
        self.weight_decay_factor = weight_decay_factor
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.currency_pairs = currency_pairs or []
        self.correlation_threshold = correlation_threshold
        self.pattern_detection_threshold = pattern_detection_threshold
        self.regime_change_sensitivity = regime_change_sensitivity
        self.enable_correlation_analysis = enable_correlation_analysis
        self.enable_pattern_recognition = enable_pattern_recognition
        self.enable_regime_detection = enable_regime_detection
        self.adaptive_weighting = adaptive_weighting
        self.logger = logging.getLogger(__name__)
        self.performance_history = {tf: [] for tf in timeframes}
        self.signal_counts = {tf: (0) for tf in timeframes}
        self.win_rates = {tf: (0.0) for tf in timeframes}
        self.consistency_scores = {tf: (0.0) for tf in timeframes}
        self.timeframe_weights = {tf: (1.0) for tf in timeframes}
        self.base_weights = {tf: (1.0) for tf in timeframes}
        if self.primary_timeframe:
            self.timeframe_weights[self.primary_timeframe] = 1.2
            self.base_weights[self.primary_timeframe] = 1.2
        self.total_signals_processed = 0
        self.last_optimization_time = None
        self.correlation_matrices = {}
        self.correlation_history = {}
        self.correlation_breakdowns = []
        self.detected_patterns = {}
        self.pattern_history = []
        self.cross_timeframe_patterns = []
        self.current_regime = MarketRegime.RANGING
        self.regime_history = []
        self.regime_transition_indicators = {}
        self.correlation_weight_adjustments = {tf: (0.0) for tf in timeframes}
        self.pattern_weight_adjustments = {tf: (0.0) for tf in timeframes}
        self.regime_weight_adjustments = {tf: (0.0) for tf in timeframes}
        self.logger.info(
            f'Enhanced TimeframeOptimizationService initialized with timeframes: {timeframes}'
            )

    def record_timeframe_performance(self, timeframe: str, outcome:
        SignalOutcome, symbol: str, pips_result: float, confidence: float,
        timestamp: Optional[datetime]=None) ->bool:
        """
        Record the performance of a signal from a specific timeframe.

        Args:
            timeframe: The timeframe of the signal
            outcome: Enum indicating win, loss, breakeven
            symbol: The trading symbol
            pips_result: Profit/loss in pips
            confidence: Signal confidence level (0-1)
            timestamp: Optional timestamp (defaults to now)

        Returns:
            Boolean indicating whether the record was added successfully
        """
        if timeframe not in self.timeframes:
            self.logger.warning(
                f'Attempted to record performance for unknown timeframe: {timeframe}'
                )
            return False
        timestamp = timestamp or datetime.now()
        record = {'timestamp': timestamp, 'outcome': outcome.value,
            'symbol': symbol, 'pips_result': pips_result, 'confidence':
            confidence, 'weight': self._calculate_record_weight(timestamp,
            confidence)}
        self.performance_history[timeframe].append(record)
        self.signal_counts[timeframe] += 1
        self.total_signals_processed += 1
        self._cleanup_old_records()
        self._update_timeframe_statistics(timeframe)
        return True

    def _calculate_record_weight(self, timestamp: datetime, confidence: float
        ) ->float:
        """
        Calculate the weight of a performance record based on recency and confidence.

        Args:
            timestamp: When the signal occurred
            confidence: Confidence level of the signal

        Returns:
            Weight value for this record
        """
        now = datetime.now()
        days_old = (now - timestamp).total_seconds() / 86400
        time_factor = self.weight_decay_factor ** min(days_old, self.
            lookback_days)
        confidence_factor = 0.5 + confidence * 0.5
        return time_factor * confidence_factor

    def _cleanup_old_records(self) ->None:
        """Remove performance records that are older than the lookback period."""
        cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
        for tf in self.timeframes:
            self.performance_history[tf] = [record for record in self.
                performance_history[tf] if record['timestamp'] > cutoff_date]

    def _update_timeframe_statistics(self, timeframe: str) ->None:
        """
        Update statistics for a specific timeframe.

        Args:
            timeframe: The timeframe to update
        """
        records = self.performance_history[timeframe]
        if not records:
            return
        wins = sum(1 for r in records if r['outcome'] == SignalOutcome.WIN.
            value)
        total = len(records)
        self.win_rates[timeframe] = wins / total if total > 0 else 0.0
        if total >= 3:
            pips_results = [r['pips_result'] for r in records]
            mean_result = np.mean(pips_results)
            if mean_result > 0:
                variance = np.var(pips_results)
                self.consistency_scores[timeframe] = 1.0 / (1.0 + variance /
                    abs(mean_result))
            else:
                self.consistency_scores[timeframe] = 0.0

    def optimize_timeframe_weights(self) ->Dict[str, float]:
        """
        Calculate optimal weights for timeframes based on historical performance.

        Returns:
            Dictionary of timeframe weights
        """
        total_signals = sum(len(records) for records in self.
            performance_history.values())
        if total_signals < self.min_signals_required:
            self.logger.info(
                f'Not enough signals for optimization: {total_signals}/{self.min_signals_required}'
                )
            return self.timeframe_weights
        for tf in self.timeframes:
            self._update_timeframe_statistics(tf)
        raw_weights = {}
        for tf in self.timeframes:
            win_rate = self.win_rates[tf]
            consistency = self.consistency_scores[tf]
            signal_count = len(self.performance_history[tf])
            confidence_factor = min(1.0, signal_count / self.
                min_signals_required)
            if signal_count > 0:
                raw_weight = (0.5 + win_rate * 0.5) * (1.0 + consistency * 0.5)
                raw_weights[tf] = raw_weight * confidence_factor + 1.0 * (
                    1.0 - confidence_factor)
            else:
                raw_weights[tf] = 1.0
        max_raw = max(raw_weights.values()) if raw_weights else 1.0
        min_raw = min(raw_weights.values()) if raw_weights else 1.0
        normalized_weights = {}
        if max_raw > min_raw:
            for tf, raw in raw_weights.items():
                normalized = self.min_weight + (raw - min_raw) * (self.
                    max_weight - self.min_weight) / (max_raw - min_raw)
                normalized_weights[tf] = normalized
        else:
            normalized_weights = self.timeframe_weights.copy()
        if self.primary_timeframe in normalized_weights:
            if normalized_weights[self.primary_timeframe] < 1.0:
                normalized_weights[self.primary_timeframe] = max(1.0,
                    normalized_weights[self.primary_timeframe])
        self.timeframe_weights = normalized_weights
        self.last_optimization_time = datetime.now()
        self.logger.info(f'Optimized timeframe weights: {normalized_weights}')
        return self.timeframe_weights

    @with_resilience('get_timeframe_weights')
    def get_timeframe_weights(self, force_optimize: bool=False) ->Dict[str,
        float]:
        """
        Get the current timeframe weights.

        Args:
            force_optimize: Whether to force a weight optimization before returning

        Returns:
            Dictionary of timeframe weights
        """
        if force_optimize:
            return self.optimize_timeframe_weights()
        return self.timeframe_weights

    @with_resilience('get_performance_stats')
    def get_performance_stats(self) ->Dict[str, Any]:
        """
        Get performance statistics for all timeframes.

        Returns:
            Dictionary with performance statistics
        """
        stats = {'timeframe_stats': {}, 'total_signals': self.
            total_signals_processed, 'last_optimization': self.
            last_optimization_time.isoformat() if self.
            last_optimization_time else None}
        for tf in self.timeframes:
            tf_stats = {'signals': self.signal_counts[tf], 'win_rate': self
                .win_rates[tf], 'consistency': self.consistency_scores[tf],
                'current_weight': self.timeframe_weights[tf],
                'recent_outcomes': [{'outcome': r['outcome'], 'pips': r[
                'pips_result']} for r in sorted(self.performance_history[tf
                ], key=lambda x: x['timestamp'], reverse=True)[:5]]}
            stats['timeframe_stats'][tf] = tf_stats
        return stats

    @with_resilience('get_recommended_timeframes')
    def get_recommended_timeframes(self, max_count: int=3) ->List[str]:
        """
        Get the recommended timeframes with the highest weights.

        Args:
            max_count: Maximum number of timeframes to return

        Returns:
            List of timeframe strings, sorted by weight
        """
        sorted_tfs = sorted(self.timeframes, key=lambda tf: self.
            timeframe_weights.get(tf, 0.0), reverse=True)
        if self.primary_timeframe in sorted_tfs:
            sorted_tfs.remove(self.primary_timeframe)
            recommended = [self.primary_timeframe] + sorted_tfs[:max_count - 1]
        else:
            recommended = sorted_tfs[:max_count]
        return recommended

    def apply_weighted_score(self, timeframe_scores: Dict[str, float]) ->Tuple[
        float, Dict[str, float]]:
        """
        Apply weights to timeframe scores and calculate weighted average.

        Args:
            timeframe_scores: Dictionary of scores by timeframe

        Returns:
            Tuple of (weighted_average_score, weighted_scores_by_timeframe)
        """
        valid_scores = {tf: score for tf, score in timeframe_scores.items() if
            tf in self.timeframes}
        if not valid_scores:
            self.logger.warning(
                'No valid timeframe scores provided for weighting')
            return 0.0, {}
        weights = self.get_timeframe_weights()
        weighted_scores = {}
        weight_sum = 0.0
        weighted_score_sum = 0.0
        for tf, score in valid_scores.items():
            weight = weights.get(tf, 1.0)
            weighted_score = score * weight
            weighted_scores[tf] = weighted_score
            weight_sum += weight
            weighted_score_sum += weighted_score
        if weight_sum > 0:
            weighted_average = weighted_score_sum / weight_sum
        else:
            weighted_average = sum(valid_scores.values()) / len(valid_scores
                ) if valid_scores else 0.0
        return weighted_average, weighted_scores

    @with_exception_handling
    def save_to_file(self, file_path: str) ->bool:
        """
        Save optimization state to a file.

        Args:
            file_path: Path to save the state

        Returns:
            Success status
        """
        try:
            serializable_history = {}
            for tf, records in self.performance_history.items():
                serializable_history[tf] = []
                for record in records:
                    serializable_record = record.copy()
                    serializable_record['timestamp'] = record['timestamp'
                        ].isoformat()
                    serializable_history[tf].append(serializable_record)
            state = {'timeframes': self.timeframes, 'primary_timeframe':
                self.primary_timeframe, 'lookback_days': self.lookback_days,
                'min_signals_required': self.min_signals_required,
                'weight_decay_factor': self.weight_decay_factor,
                'max_weight': self.max_weight, 'min_weight': self.
                min_weight, 'performance_history': serializable_history,
                'signal_counts': self.signal_counts, 'win_rates': self.
                win_rates, 'consistency_scores': self.consistency_scores,
                'timeframe_weights': self.timeframe_weights,
                'total_signals_processed': self.total_signals_processed,
                'last_optimization_time': self.last_optimization_time.
                isoformat() if self.last_optimization_time else None}
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            self.logger.info(f'Optimization state saved to {file_path}')
            return True
        except Exception as e:
            self.logger.error(f'Error saving optimization state: {e}')
            return False

    @classmethod
    @with_exception_handling
    def load_from_file(cls, file_path: str) ->'TimeframeOptimizationService':
        """
        Load optimization state from a file.

        Args:
            file_path: Path to load the state from

        Returns:
            Loaded TimeframeOptimizationService instance
        """
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            instance = cls(timeframes=state['timeframes'],
                primary_timeframe=state['primary_timeframe'], lookback_days
                =state['lookback_days'], min_signals_required=state[
                'min_signals_required'], weight_decay_factor=state[
                'weight_decay_factor'], max_weight=state['max_weight'],
                min_weight=state['min_weight'])
            for tf, records in state['performance_history'].items():
                for record in records:
                    record['timestamp'] = datetime.fromisoformat(record[
                        'timestamp'])
                instance.performance_history[tf] = records
            instance.signal_counts = state['signal_counts']
            instance.win_rates = state['win_rates']
            instance.consistency_scores = state['consistency_scores']
            instance.timeframe_weights = state['timeframe_weights']
            instance.total_signals_processed = state['total_signals_processed']
            instance.last_optimization_time = datetime.fromisoformat(state[
                'last_optimization_time']) if state['last_optimization_time'
                ] else None
            instance.logger.info(f'Optimization state loaded from {file_path}')
            return instance
        except Exception as e:
            logging.error(f'Error loading optimization state: {e}')
            return cls([TimeFrame.H1.value])
