"""
Adaptive Signal Quality Integration

This module enhances the Adaptive Layer with signal quality evaluation capabilities,
creating a feedback loop between signal quality assessment and parameter adaptation.
"""
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json
from analysis_engine.services.tool_effectiveness import MarketRegime
from analysis_engine.services.signal_quality_evaluator import SignalQualityEvaluator
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class AdaptiveSignalQualityIntegration:
    """
    Integrates signal quality evaluation with the adaptive layer to create
    a feedback loop for continuous self-improvement of trading strategies.
    """

    def __init__(self):
        """Initialize the adaptive signal quality integration module"""
        self.logger = logging.getLogger(__name__)
        self.quality_evaluator = SignalQualityEvaluator()
        self.quality_metrics_cache = {}
        self.cache_expiry = {}
        self.cache_ttl = timedelta(minutes=15)
        self.high_quality_threshold = 0.7
        self.low_quality_threshold = 0.4

    @with_exception_handling
    def evaluate_signal_quality(self, signal_event: Dict[str, Any],
        market_context: Dict[str, Any], historical_performance: Optional[
        Dict[str, Any]]=None, additional_signals: Optional[List[Dict[str,
        Any]]]=None) ->Dict[str, Any]:
        """
        Evaluate the quality of a trading signal
        
        Args:
            signal_event: Signal event data
            market_context: Current market context 
            historical_performance: Historical performance data for the tool
            additional_signals: Other signals in the same time window
            
        Returns:
            Quality evaluation result
        """
        try:
            if isinstance(signal_event, dict):
                from analysis_engine.services.tool_effectiveness import SignalEvent
                converted_signal = SignalEvent(tool_name=signal_event.get(
                    'tool_id', ''), signal_type=signal_event.get(
                    'signal_type', ''), direction=signal_event.get(
                    'direction', signal_event.get('signal_type', '')),
                    strength=signal_event.get('confidence', 0.5), timestamp
                    =signal_event.get('timestamp', datetime.now()), symbol=
                    signal_event.get('instrument', ''), timeframe=
                    signal_event.get('timeframe', ''), price_at_signal=
                    signal_event.get('price', 0.0), metadata=signal_event.
                    get('metadata', {}), market_context=market_context)
                converted_additional_signals = None
                if additional_signals:
                    converted_additional_signals = []
                    for signal in additional_signals:
                        converted_additional_signals.append(SignalEvent(
                            tool_name=signal.get('tool_id', ''),
                            signal_type=signal.get('signal_type', ''),
                            direction=signal.get('direction', signal.get(
                            'signal_type', '')), strength=signal.get(
                            'confidence', 0.5), timestamp=signal.get(
                            'timestamp', datetime.now()), symbol=signal.get
                            ('instrument', ''), timeframe=signal.get(
                            'timeframe', ''), price_at_signal=signal.get(
                            'price', 0.0), metadata=signal.get('metadata',
                            {}), market_context=market_context))
                quality_metrics = (self.quality_evaluator.
                    evaluate_signal_quality(signal=converted_signal,
                    market_context=market_context, additional_signals=
                    converted_additional_signals, historical_performance=
                    historical_performance))
                cache_key = self._generate_cache_key(signal_event)
                self.quality_metrics_cache[cache_key] = quality_metrics
                self.cache_expiry[cache_key] = datetime.now() + self.cache_ttl
                return quality_metrics
            else:
                quality_metrics = (self.quality_evaluator.
                    evaluate_signal_quality(signal=signal_event,
                    market_context=market_context, additional_signals=
                    additional_signals, historical_performance=
                    historical_performance))
                return quality_metrics
        except Exception as e:
            self.logger.error(f'Error evaluating signal quality: {str(e)}')
            return {'base_quality': 0.5, 'timing_quality': 0.5,
                'overall_quality': 0.5, 'error': str(e)}

    @with_resilience('get_cached_quality')
    def get_cached_quality(self, signal_event: Dict[str, Any]) ->Optional[Dict
        [str, Any]]:
        """
        Get cached quality metrics for a signal if available
        
        Args:
            signal_event: Signal event data
            
        Returns:
            Cached quality metrics or None if not available
        """
        cache_key = self._generate_cache_key(signal_event)
        if cache_key in self.quality_metrics_cache:
            if datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
                return self.quality_metrics_cache[cache_key]
            else:
                if cache_key in self.quality_metrics_cache:
                    del self.quality_metrics_cache[cache_key]
                if cache_key in self.cache_expiry:
                    del self.cache_expiry[cache_key]
        return None

    def _generate_cache_key(self, signal_event: Dict[str, Any]) ->str:
        """Generate a cache key for a signal event"""
        return (
            f"{signal_event.get('tool_id')}-{signal_event.get('signal_type')}-{signal_event.get('instrument')}-{signal_event.get('timestamp')}"
            )

    def filter_signals_by_quality(self, signals: List[Dict[str, Any]],
        market_context: Dict[str, Any], min_quality_threshold: float=0.0
        ) ->List[Dict[str, Any]]:
        """
        Filter signals based on their quality
        
        Args:
            signals: List of signal events to filter
            market_context: Current market context
            min_quality_threshold: Minimum quality threshold to include
            
        Returns:
            Filtered list of signals with quality metrics added
        """
        result = []
        for signal in signals:
            quality_metrics = self.get_cached_quality(signal)
            if quality_metrics is None:
                quality_metrics = self.evaluate_signal_quality(signal,
                    market_context)
            overall_quality = quality_metrics.get('overall_quality', 0.0)
            if overall_quality >= min_quality_threshold:
                signal_with_quality = signal.copy()
                signal_with_quality['quality_metrics'] = quality_metrics
                result.append(signal_with_quality)
        return result

    def adjust_signal_weights_by_quality(self, signals: List[Dict[str, Any]
        ], base_weights: Dict[str, float]) ->Dict[str, float]:
        """
        Adjust signal weights based on evaluated signal quality
        
        Args:
            signals: List of signals with quality metrics
            base_weights: Base weights for each tool
            
        Returns:
            Adjusted weights
        """
        adjusted_weights = base_weights.copy()
        for signal in signals:
            if 'quality_metrics' not in signal:
                continue
            tool_id = signal.get('tool_id')
            if tool_id not in adjusted_weights:
                continue
            quality = signal['quality_metrics'].get('overall_quality', 0.5)
            if quality > self.high_quality_threshold:
                boost_factor = 1.0 + (quality - self.high_quality_threshold
                    ) / (1.0 - self.high_quality_threshold)
                adjusted_weights[tool_id] *= boost_factor
            elif quality < self.low_quality_threshold:
                reduction_factor = quality / self.low_quality_threshold
                adjusted_weights[tool_id] *= reduction_factor
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            for tool_id in adjusted_weights:
                adjusted_weights[tool_id] /= total_weight
        return adjusted_weights

    def generate_quality_based_recommendations(self, signals: List[Dict[str,
        Any]], market_regime: MarketRegime) ->Dict[str, Any]:
        """
        Generate recommendations based on signal quality analysis
        
        Args:
            signals: List of signals with quality metrics
            market_regime: Current market regime
            
        Returns:
            Dictionary with recommendations
        """
        quality_values = [s['quality_metrics'].get('overall_quality', 0.0) for
            s in signals if 'quality_metrics' in s]
        if not quality_values:
            return {'recommendation': 'insufficient_data', 'confidence': 
                0.0, 'explanation': 'No quality metrics available for analysis'
                }
        avg_quality = np.mean(quality_values)
        quality_std = np.std(quality_values)
        quality_consistency = 1.0 - min(1.0, quality_std)
        directions = {}
        for signal in signals:
            if 'quality_metrics' not in signal:
                continue
            direction = signal.get('direction', signal.get('signal_type', ''))
            quality = signal['quality_metrics'].get('overall_quality', 0.0)
            if direction not in directions:
                directions[direction] = []
            directions[direction].append(quality)
        conflicting_signals = False
        high_quality_directions = []
        for direction, qualities in directions.items():
            if any(q >= self.high_quality_threshold for q in qualities):
                high_quality_directions.append(direction)
        conflicting_signals = len(high_quality_directions) > 1
        if (avg_quality >= self.high_quality_threshold and 
            quality_consistency >= 0.7 and not conflicting_signals):
            recommendation = {'recommendation': 'increase_position_size',
                'confidence': min(0.9, avg_quality), 'explanation':
                'High quality and consistent signals detected'}
        elif avg_quality >= self.high_quality_threshold and conflicting_signals:
            recommendation = {'recommendation': 'wait_for_confirmation',
                'confidence': 0.6, 'explanation':
                'High quality but conflicting signals detected'}
        elif avg_quality < self.low_quality_threshold:
            recommendation = {'recommendation': 'reduce_position_size',
                'confidence': 1.0 - avg_quality, 'explanation':
                'Low quality signals detected'}
        elif market_regime in [MarketRegime.VOLATILE, MarketRegime.CHOPPY]:
            recommendation = {'recommendation': 'use_tight_stops',
                'confidence': 0.7, 'explanation':
                f'Medium quality signals in {market_regime} regime'}
        else:
            recommendation = {'recommendation': 'standard_position_size',
                'confidence': avg_quality, 'explanation':
                'Average signal quality detected'}
        recommendation['quality_stats'] = {'average': avg_quality,
            'std_dev': quality_std, 'consistency': quality_consistency,
            'conflicting_signals': conflicting_signals}
        return recommendation
