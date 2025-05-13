"""
Adaptive Layer Service

This module provides functionality to dynamically adjust trading parameters
based on market conditions, tool effectiveness metrics, and other feedback.
It implements the self-evolution aspect by creating feedback loops between
performance measurement and parameter adaptation.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Set
from datetime import datetime, timedelta
import json
from enum import Enum
from analysis_engine.services.market_regime_detector import MarketRegimeAnalyzer, MarketRegimeService
from analysis_engine.services.tool_effectiveness import MarketRegime
from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository
from analysis_engine.services.adaptive_signal_quality import AdaptiveSignalQualityIntegration


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class AdaptationStrategy(str, Enum):
    """Strategies for adapting parameters based on effectiveness data"""
    CONSERVATIVE = 'conservative'
    MODERATE = 'moderate'
    AGGRESSIVE = 'aggressive'
    EXPERIMENTAL = 'experimental'


class AdaptiveLayer:
    """
    Dynamically adjusts trading parameters based on market conditions,
    tool effectiveness metrics, and signal quality evaluation.
    
    This class is a core component of the platform's self-evolution capability,
    creating feedback loops that allow the system to optimize itself over time.
    """

    def __init__(self, tool_effectiveness_repository:
        ToolEffectivenessRepository, market_regime_service:
        MarketRegimeService=None, adaptation_strategy: AdaptationStrategy=
        AdaptationStrategy.MODERATE, signal_quality_integration:
        AdaptiveSignalQualityIntegration=None):
        """
        Initialize the adaptive layer
        
        Args:
            tool_effectiveness_repository: Repository for accessing tool effectiveness data
            market_regime_service: Service for detecting market regimes
            adaptation_strategy: Strategy for parameter adaptation
            signal_quality_integration: Component for signal quality evaluation integration
        """
        self.repository = tool_effectiveness_repository
        self.market_regime_service = (market_regime_service or
            MarketRegimeService())
        self.adaptation_strategy = adaptation_strategy
        self.signal_quality = (signal_quality_integration or
            AdaptiveSignalQualityIntegration())
        self.logger = logging.getLogger(__name__)
        self.metrics_cache = {}
        self.cache_expiry = {}
        self.cache_duration = timedelta(minutes=30)
        self.adaptation_rates = {AdaptationStrategy.CONSERVATIVE: 0.1,
            AdaptationStrategy.MODERATE: 0.25, AdaptationStrategy.
            AGGRESSIVE: 0.5, AdaptationStrategy.EXPERIMENTAL: 0.75}
        self.min_signal_quality = 0.3
        self.quality_weight_factor = 0.4

    @with_exception_handling
    def generate_adaptive_parameters(self, symbol: str, timeframe: str,
        price_data: pd.DataFrame, available_tools: List[str],
        recent_signals: Optional[List[Dict[str, Any]]]=None) ->Dict[str, Any]:
        """
        Generate adaptive parameters based on current market conditions, tool effectiveness,
        and signal quality metrics.
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            price_data: OHLC price data
            available_tools: List of tool IDs available for the strategy
            recent_signals: Optional list of recent signals for quality assessment
            
        Returns:
            Dictionary with adaptive parameters
        """
        try:
            regime_result = self.market_regime_service.detect_current_regime(
                symbol=symbol, timeframe=timeframe, price_data=price_data)
            current_regime = regime_result['regime']
            regime_confidence = regime_result['confidence']
            dominant_regime = self.market_regime_service.get_dominant_regime(
                symbol=symbol, timeframe=timeframe)
            if dominant_regime['occurrence_rate'] > 0.6 and dominant_regime[
                'confidence'] > 0.5:
                final_regime = dominant_regime['regime']
                regime_certainty = dominant_regime['confidence'
                    ] * dominant_regime['occurrence_rate']
            else:
                final_regime = current_regime
                regime_certainty = regime_confidence
            market_context = {'regime': final_regime, 'price_data':
                price_data.tail(100).to_dict('records'), 'symbol': symbol,
                'timeframe': timeframe, 'regime_confidence': regime_certainty}
            adaptive_params = self._generate_parameters(symbol=symbol,
                timeframe=timeframe, regime=final_regime, regime_certainty=
                regime_certainty, available_tools=available_tools,
                market_context=market_context, recent_signals=recent_signals)
            adaptive_params['detected_regime'] = {'current': {'regime':
                current_regime, 'confidence': regime_confidence},
                'dominant': {'regime': dominant_regime['regime'],
                'confidence': dominant_regime['confidence'],
                'occurrence_rate': dominant_regime['occurrence_rate']},
                'final_regime': final_regime, 'certainty': regime_certainty}
            return adaptive_params
        except Exception as e:
            self.logger.error(f'Error generating adaptive parameters: {str(e)}'
                )
            return self._get_default_parameters(available_tools)

    def _generate_parameters(self, symbol: str, timeframe: str, regime:
        MarketRegime, regime_certainty: float, available_tools: List[str],
        market_context: Dict[str, Any]=None, recent_signals: List[Dict[str,
        Any]]=None) ->Dict[str, Any]:
        """
        Generate parameter adjustments based on effectiveness metrics, market regime,
        and signal quality assessment
        
        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            regime: Current market regime
            regime_certainty: Confidence in the regime detection
            available_tools: List of tool IDs available
            market_context: Current market context for signal quality evaluation
            recent_signals: Recent signals for quality assessment
            
        Returns:
            Dictionary with adaptive parameters
        """
        tool_metrics = {}
        for tool_id in available_tools:
            metrics = self._get_cached_effectiveness_metrics(tool_id=
                tool_id, timeframe=timeframe, instrument=symbol,
                market_regime=regime)
            tool_metrics[tool_id] = metrics
        signal_weights = self._calculate_signal_weights(tool_metrics, regime)
        quality_recommendations = None
        if recent_signals and market_context:
            evaluated_signals = []
            for signal in recent_signals:
                if signal.get('tool_id') not in available_tools:
                    continue
                historical_performance = tool_metrics.get(signal.get(
                    'tool_id'), {})
                quality_metrics = self.signal_quality.evaluate_signal_quality(
                    signal_event=signal, market_context=market_context,
                    historical_performance=historical_performance)
                signal_with_quality = signal.copy()
                signal_with_quality['quality_metrics'] = quality_metrics
                evaluated_signals.append(signal_with_quality)
            if evaluated_signals:
                adjusted_weights = (self.signal_quality.
                    adjust_signal_weights_by_quality(signals=
                    evaluated_signals, base_weights=signal_weights))
                quality_recommendations = (self.signal_quality.
                    generate_quality_based_recommendations(
                    evaluated_signals, regime))
                signal_weights = adjusted_weights
        timeframe_weights = self._determine_timeframe_weights(regime)
        risk_adjustment_factor = self._calculate_risk_adjustment(regime=
            regime, regime_certainty=regime_certainty, tool_metrics=
            tool_metrics)
        if quality_recommendations:
            recommendation = quality_recommendations.get('recommendation')
            confidence = quality_recommendations.get('confidence', 0.5)
            if recommendation == 'increase_position_size':
                risk_adjustment_factor *= 1.0 + 0.2 * confidence
            elif recommendation == 'reduce_position_size':
                risk_adjustment_factor *= 0.8 * confidence
            elif recommendation == 'use_tight_stops':
                pass
        preferred_model = self._determine_preferred_model(regime, tool_metrics)
        parameter_adjustments = self._calculate_parameter_adjustments(regime
            =regime, tool_metrics=tool_metrics)
        result = {'signal_weights': signal_weights, 'timeframe_weights':
            timeframe_weights, 'risk_adjustment_factor':
            risk_adjustment_factor, 'preferred_model': preferred_model,
            'parameter_adjustments': parameter_adjustments, 'timestamp':
            datetime.now().isoformat()}
        if quality_recommendations:
            result['quality_recommendations'] = quality_recommendations
        return result

    @with_exception_handling
    def _get_cached_effectiveness_metrics(self, tool_id: str, timeframe:
        Optional[str]=None, instrument: Optional[str]=None, market_regime:
        Optional[MarketRegime]=None) ->Dict[str, Any]:
        """
        Get effectiveness metrics from cache or repository
        
        Args:
            tool_id: ID of the tool
            timeframe: Chart timeframe filter
            instrument: Symbol filter
            market_regime: Market regime filter
            
        Returns:
            Dictionary with effectiveness metrics
        """
        cache_key = f'{tool_id}_{timeframe}_{instrument}_{market_regime}'
        if (cache_key in self.metrics_cache and cache_key in self.
            cache_expiry and datetime.now() < self.cache_expiry[cache_key]):
            return self.metrics_cache[cache_key]
        try:
            win_rate, signal_count, outcome_count = (self.repository.
                get_tool_win_rate(tool_id=tool_id, timeframe=timeframe,
                instrument=instrument, market_regime=market_regime))
            latest_metrics = self.repository.get_latest_tool_metrics(tool_id)
            regime_metrics = {}
            if market_regime:
                effectiveness_metrics = (self.repository.
                    get_effectiveness_metrics(tool_id=tool_id, metric_type=
                    'reliability_by_regime'))
                if effectiveness_metrics:
                    latest_metric = max(effectiveness_metrics, key=lambda x:
                        x.created_at)
                    if hasattr(latest_metric, 'details'
                        ) and latest_metric.details:
                        details = json.loads(latest_metric.details
                            ) if isinstance(latest_metric.details, str
                            ) else latest_metric.details
                        regime_metrics = details.get('by_regime', {})
            metrics = {'win_rate': win_rate, 'signal_count': signal_count,
                'outcome_count': outcome_count, 'profit_factor':
                latest_metrics.get('profit_factor'), 'expected_payoff':
                latest_metrics.get('expected_payoff'), 'regime_metrics':
                regime_metrics, 'fetched_at': datetime.now().isoformat()}
            self.metrics_cache[cache_key] = metrics
            self.cache_expiry[cache_key] = datetime.now() + self.cache_duration
            return metrics
        except Exception as e:
            self.logger.error(
                f'Error fetching effectiveness metrics for {tool_id}: {str(e)}'
                )
            return {'win_rate': 50.0, 'signal_count': 0, 'outcome_count': 0,
                'profit_factor': 1.0, 'expected_payoff': 0.0,
                'regime_metrics': {}, 'error': str(e)}

    def _calculate_signal_weights(self, tool_metrics: Dict[str, Dict[str,
        Any]], current_regime: MarketRegime) ->Dict[str, float]:
        """
        Calculate optimal signal weights based on tool effectiveness
        
        Args:
            tool_metrics: Dictionary of effectiveness metrics for each tool
            current_regime: Current market regime
            
        Returns:
            Dictionary with signal weights for each tool
        """
        weights = {}
        total_score = 0
        for tool_id, metrics in tool_metrics.items():
            score = 1.0
            win_rate = metrics.get('win_rate', 50.0)
            if win_rate > 50:
                score *= 1 + (win_rate - 50) / 50
            else:
                score *= win_rate / 50
            profit_factor = metrics.get('profit_factor', 1.0) or 1.0
            score *= min(profit_factor, 3.0) / 1.5
            expected_payoff = metrics.get('expected_payoff', 0.0) or 0.0
            if expected_payoff > 0:
                score *= 1 + min(expected_payoff / 100, 0.5)
            regime_metrics = metrics.get('regime_metrics', {})
            regime_data = regime_metrics.get(str(current_regime), {})
            if regime_data:
                regime_win_rate = regime_data.get('win_rate', win_rate / 100
                    ) * 100
                if regime_win_rate > win_rate:
                    boost_factor = 1 + min((regime_win_rate - win_rate) / 
                        50, 1.0)
                    score *= boost_factor
                elif regime_win_rate < win_rate:
                    penalty_factor = max(0.5, regime_win_rate / win_rate
                        ) if win_rate > 0 else 0.5
                    score *= penalty_factor
            score = max(0.1, score)
            weights[tool_id] = score
            total_score += score
        if total_score > 0:
            for tool_id in weights:
                weights[tool_id] /= total_score
        else:
            equal_weight = 1.0 / len(tool_metrics) if tool_metrics else 0
            weights = {tool_id: equal_weight for tool_id in tool_metrics}
        return weights

    def _determine_timeframe_weights(self, regime: MarketRegime) ->Dict[str,
        float]:
        """
        Determine optimal timeframe weights based on the current market regime
        
        Args:
            regime: Current market regime
            
        Returns:
            Dictionary with weights for each timeframe
        """
        default_weights = {'1m': 0.05, '5m': 0.1, '15m': 0.15, '30m': 0.15,
            '1h': 0.2, '4h': 0.2, '1d': 0.1, '1w': 0.05}
        if (regime == MarketRegime.TRENDING_UP or regime == MarketRegime.
            TRENDING_DOWN):
            return {'1m': 0.03, '5m': 0.07, '15m': 0.1, '30m': 0.15, '1h': 
                0.25, '4h': 0.25, '1d': 0.1, '1w': 0.05}
        elif regime == MarketRegime.RANGING:
            return {'1m': 0.05, '5m': 0.1, '15m': 0.2, '30m': 0.25, '1h': 
                0.2, '4h': 0.1, '1d': 0.05, '1w': 0.05}
        elif regime == MarketRegime.VOLATILE:
            return {'1m': 0.1, '5m': 0.15, '15m': 0.2, '30m': 0.2, '1h': 
                0.15, '4h': 0.1, '1d': 0.05, '1w': 0.05}
        elif regime == MarketRegime.BREAKOUT:
            return {'1m': 0.15, '5m': 0.15, '15m': 0.1, '30m': 0.1, '1h': 
                0.1, '4h': 0.15, '1d': 0.15, '1w': 0.1}
        elif regime == MarketRegime.CHOPPY:
            return {'1m': 0.01, '5m': 0.04, '15m': 0.05, '30m': 0.1, '1h': 
                0.2, '4h': 0.3, '1d': 0.2, '1w': 0.1}
        return default_weights

    def _calculate_risk_adjustment(self, regime: MarketRegime,
        regime_certainty: float, tool_metrics: Dict[str, Dict[str, Any]]
        ) ->float:
        """
        Calculate risk adjustment factor based on market regime and tool effectiveness
        
        Args:
            regime: Current market regime
            regime_certainty: Confidence in the regime detection
            tool_metrics: Dictionary of effectiveness metrics for each tool
            
        Returns:
            Risk adjustment factor (1.0 is normal, <1.0 reduces risk, >1.0 increases risk)
        """
        base_risk = 1.0
        regime_factors = {MarketRegime.TRENDING_UP: 1.2, MarketRegime.
            TRENDING_DOWN: 0.8, MarketRegime.RANGING: 1.0, MarketRegime.
            VOLATILE: 0.7, MarketRegime.CHOPPY: 0.6, MarketRegime.BREAKOUT:
            1.1, MarketRegime.UNKNOWN: 0.8}
        regime_risk = regime_factors.get(regime, 0.8)
        regime_risk_scaled = 1.0 + (regime_risk - 1.0) * regime_certainty
        win_rates = [metrics.get('win_rate', 50.0) for metrics in
            tool_metrics.values()]
        avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 50.0
        win_rate_factor = 1.0 + (avg_win_rate - 50.0) / 50.0
        adaptation_rate = self.adaptation_rates[self.adaptation_strategy]
        risk_adjustment = (base_risk * (1.0 - adaptation_rate) + 
            regime_risk_scaled * adaptation_rate * 0.6 + win_rate_factor *
            adaptation_rate * 0.4)
        return max(0.5, min(1.5, risk_adjustment))

    def _determine_preferred_model(self, regime: MarketRegime, tool_metrics:
        Dict[str, Dict[str, Any]]) ->str:
        """
        Determine the preferred machine learning model for the current conditions
        
        Args:
            regime: Current market regime
            tool_metrics: Dictionary of effectiveness metrics for each tool
            
        Returns:
            ID of the preferred model
        """
        regime_model_map = {MarketRegime.TRENDING_UP:
            'trend_following_model', MarketRegime.TRENDING_DOWN:
            'trend_following_model', MarketRegime.RANGING:
            'mean_reversion_model', MarketRegime.VOLATILE:
            'volatility_model', MarketRegime.CHOPPY: 'filter_model',
            MarketRegime.BREAKOUT: 'breakout_model', MarketRegime.UNKNOWN:
            'ensemble_model'}
        return regime_model_map.get(regime, 'default_model')

    def _calculate_parameter_adjustments(self, regime: MarketRegime,
        tool_metrics: Dict[str, Dict[str, Any]]) ->Dict[str, Dict[str, Any]]:
        """
        Calculate optimal parameters adjustments for technical indicators
        
        Args:
            regime: Current market regime
            tool_metrics: Dictionary of effectiveness metrics for each tool
            
        Returns:
            Dictionary with parameter adjustments for various indicators
        """
        adjustments = {'moving_averages': {'fast_period': 0, 'slow_period':
            0}, 'oscillators': {'rsi_period': 0, 'overbought_level': 0,
            'oversold_level': 0}, 'volatility': {'atr_period': 0,
            'bollinger_period': 0, 'bollinger_std': 0},
            'support_resistance': {'sensitivity': 0, 'lookback_periods': 0}}
        if (regime == MarketRegime.TRENDING_UP or regime == MarketRegime.
            TRENDING_DOWN):
            adjustments['moving_averages']['fast_period'] = 2
            adjustments['moving_averages']['slow_period'] = 5
            adjustments['oscillators']['overbought_level'] = 5
            adjustments['oscillators']['oversold_level'] = -5
        elif regime == MarketRegime.RANGING:
            adjustments['moving_averages']['fast_period'] = -2
            adjustments['moving_averages']['slow_period'] = -3
            adjustments['oscillators']['overbought_level'] = -3
            adjustments['oscillators']['oversold_level'] = 3
        elif regime == MarketRegime.VOLATILE:
            adjustments['moving_averages']['fast_period'] = 3
            adjustments['moving_averages']['slow_period'] = 7
            adjustments['oscillators']['rsi_period'] = 4
            adjustments['volatility']['atr_period'] = 5
            adjustments['volatility']['bollinger_std'] = 0.5
        elif regime == MarketRegime.CHOPPY:
            adjustments['moving_averages']['fast_period'] = 5
            adjustments['moving_averages']['slow_period'] = 10
            adjustments['oscillators']['rsi_period'] = 7
            adjustments['volatility']['bollinger_period'] = 5
            adjustments['volatility']['bollinger_std'] = 0.7
        elif regime == MarketRegime.BREAKOUT:
            adjustments['support_resistance']['sensitivity'] = 5
            adjustments['volatility']['bollinger_std'] = -0.5
        adaptation_rate = self.adaptation_rates[self.adaptation_strategy]
        for category in adjustments:
            for param in adjustments[category]:
                adjustments[category][param] *= adaptation_rate
        return adjustments

    def _get_default_parameters(self, available_tools: List[str]) ->Dict[
        str, Any]:
        """
        Get default parameters when effectiveness metrics are unavailable
        
        Args:
            available_tools: List of available tool IDs
            
        Returns:
            Dictionary with default parameters
        """
        equal_weight = 1.0 / len(available_tools) if available_tools else 0
        signal_weights = {tool_id: equal_weight for tool_id in available_tools}
        timeframe_weights = {'1m': 0.05, '5m': 0.1, '15m': 0.15, '30m': 
            0.15, '1h': 0.2, '4h': 0.2, '1d': 0.1, '1w': 0.05}
        return {'signal_weights': signal_weights, 'timeframe_weights':
            timeframe_weights, 'risk_adjustment_factor': 1.0,
            'preferred_model': 'default_model', 'parameter_adjustments': {
            'moving_averages': {'fast_period': 0, 'slow_period': 0},
            'oscillators': {'rsi_period': 0, 'overbought_level': 0,
            'oversold_level': 0}, 'volatility': {'atr_period': 0,
            'bollinger_period': 0, 'bollinger_std': 0},
            'support_resistance': {'sensitivity': 0, 'lookback_periods': 0}
            }, 'detected_regime': {'current': {'regime': MarketRegime.
            UNKNOWN, 'confidence': 0.0}, 'dominant': {'regime':
            MarketRegime.UNKNOWN, 'confidence': 0.0, 'occurrence_rate': 0.0
            }, 'final_regime': MarketRegime.UNKNOWN, 'certainty': 0.0},
            'timestamp': datetime.now().isoformat()}
