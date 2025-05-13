"""
Adaptive Weight Calculator

This module provides functionality to calculate adaptive weights for signals and strategies
based on their historical effectiveness and the current market regime.
"""
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from analysis_engine.services.tool_effectiveness import MarketRegime


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class AdaptiveWeightCalculator:
    """
    Calculates adaptive weights for different signals and strategies based on
    historical effectiveness metrics, market regimes, and statistical significance.
    """

    def __init__(self, confidence_threshold: float=0.8):
        """
        Initialize the adaptive weight calculator.
        
        Args:
            confidence_threshold: Threshold for statistical significance (0.0-1.0)
        """
        self.logger = logging.getLogger(__name__)
        self.confidence_threshold = confidence_threshold

    @with_analysis_resilience('calculate_signal_weights')
    def calculate_signal_weights(self, effectiveness_metrics: Dict[str,
        Dict[str, Any]], current_regime: MarketRegime, regime_certainty:
        float=1.0) ->Dict[str, float]:
        """
        Calculate adaptive weights for signals based on their effectiveness metrics.
        
        Args:
            effectiveness_metrics: Dictionary mapping tool IDs to their effectiveness metrics
            current_regime: Current market regime
            regime_certainty: Certainty level of the market regime detection (0.0-1.0)
            
        Returns:
            Dictionary mapping tool IDs to their calculated weights (0.0-1.0)
        """
        if not effectiveness_metrics:
            return {}
        weights = {}
        total_score = 0.0
        for tool_id, metrics in effectiveness_metrics.items():
            score = self._calculate_tool_score(metrics, current_regime)
            weights[tool_id] = score
            total_score += score
        if total_score > 0:
            normalized_weights = {k: (v / total_score) for k, v in weights.
                items()}
        else:
            equal_weight = 1.0 / len(effectiveness_metrics
                ) if effectiveness_metrics else 0.0
            normalized_weights = {tool_id: equal_weight for tool_id in
                effectiveness_metrics.keys()}
        if regime_certainty < 1.0:
            equal_weight = 1.0 / len(normalized_weights
                ) if normalized_weights else 0.0
            for tool_id in normalized_weights:
                normalized_weights[tool_id
                    ] = regime_certainty * normalized_weights[tool_id] + (
                    1.0 - regime_certainty) * equal_weight
        return normalized_weights

    def _calculate_tool_score(self, metrics: Dict[str, Any], current_regime:
        MarketRegime) ->float:
        """
        Calculate a score for a tool based on its metrics and the current market regime.
        
        Args:
            metrics: Effectiveness metrics for the tool
            current_regime: Current market regime
            
        Returns:
            Score for the tool (higher is better)
        """
        win_rate = metrics.get('win_rate', 50.0)
        profit_factor = metrics.get('profit_factor', 1.0)
        expected_payoff = metrics.get('expected_payoff', 0.0)
        sample_size = metrics.get('signal_count', 0)
        regime_metrics = metrics.get('regime_metrics', {}).get(str(
            current_regime.value), {})
        statistical_significance = min(1.0, sample_size / 30.0)
        if statistical_significance < 0.2:
            base_score = 0.5
        else:
            win_rate_component = win_rate / 100.0 if win_rate <= 100 else 1.0
            pf_component = min(profit_factor, 3.0) / 3.0
            ep_component = 1.0 / (1.0 + np.exp(-expected_payoff))
            base_score = (0.5 * win_rate_component + 0.3 * pf_component + 
                0.2 * ep_component)
        if regime_metrics:
            regime_win_rate = regime_metrics.get('win_rate', win_rate)
            regime_profit_factor = regime_metrics.get('profit_factor',
                profit_factor)
            regime_win_rate_component = (regime_win_rate / 100.0 if 
                regime_win_rate <= 100 else 1.0)
            regime_pf_component = min(regime_profit_factor, 3.0) / 3.0
            regime_score = (0.6 * regime_win_rate_component + 0.4 *
                regime_pf_component)
            regime_sample_size = regime_metrics.get('signal_count', 0)
            regime_significance = min(1.0, regime_sample_size / 20.0)
            combined_score = (1.0 - regime_significance
                ) * base_score + regime_significance * regime_score
            return combined_score
        return base_score

    @with_analysis_resilience('calculate_strategy_weights')
    def calculate_strategy_weights(self, strategies: Dict[str, Dict[str,
        Any]], market_regime: MarketRegime) ->Dict[str, float]:
        """
        Calculate weights for different trading strategies based on their performance.
        
        Args:
            strategies: Dictionary mapping strategy IDs to their performance metrics
            market_regime: Current market regime
            
        Returns:
            Dictionary mapping strategy IDs to their weights
        """
        if not strategies:
            return {}
        weights = {}
        total_weight = 0.0
        for strategy_id, metrics in strategies.items():
            win_rate = metrics.get('win_rate', 50.0)
            profit_factor = metrics.get('profit_factor', 1.0)
            regime_performance = metrics.get('regime_performance', {}).get(str
                (market_regime.value), {})
            if regime_performance:
                regime_win_rate = regime_performance.get('win_rate', win_rate)
                regime_profit_factor = regime_performance.get('profit_factor',
                    profit_factor)
                weight = regime_win_rate / 100.0 * min(regime_profit_factor,
                    3.0) / 1.5
            else:
                weight = win_rate / 100.0 * min(profit_factor, 3.0) / 2.0
            trades = metrics.get('total_trades', 0)
            significance_factor = min(1.0, trades / 50.0)
            adjusted_weight = 0.5 + (weight - 0.5) * significance_factor
            weights[strategy_id] = adjusted_weight
            total_weight += adjusted_weight
        if total_weight > 0:
            return {k: (v / total_weight) for k, v in weights.items()}
        else:
            equal_weight = 1.0 / len(strategies) if strategies else 0.0
            return {strategy_id: equal_weight for strategy_id in strategies
                .keys()}

    def apply_trend_factors(self, weights: Dict[str, float], trend_factors:
        Dict[str, float]) ->Dict[str, float]:
        """
        Apply trend factors to adjust weights based on recent performance trends.
        
        Args:
            weights: Current weights
            trend_factors: Dictionary mapping tool/strategy IDs to trend factors
                           (>1.0 for improving trends, <1.0 for declining trends)
                           
        Returns:
            Dictionary with adjusted weights
        """
        if not weights or not trend_factors:
            return weights
        adjusted_weights = {}
        total_adjusted = 0.0
        for tool_id, weight in weights.items():
            trend_factor = trend_factors.get(tool_id, 1.0)
            adjusted = weight * trend_factor
            adjusted_weights[tool_id] = adjusted
            total_adjusted += adjusted
        if total_adjusted > 0:
            return {k: (v / total_adjusted) for k, v in adjusted_weights.
                items()}
        return adjusted_weights

    @with_analysis_resilience('calculate_statistical_significance')
    def calculate_statistical_significance(self, sample_size: int,
        required_confidence: float=None) ->float:
        """
        Calculate statistical significance factor based on sample size.
        
        Args:
            sample_size: Number of samples/trades
            required_confidence: Required confidence level (defaults to self.confidence_threshold)
            
        Returns:
            Statistical significance factor (0.0-1.0)
        """
        if required_confidence is None:
            required_confidence = self.confidence_threshold
        required_samples = int(30 * required_confidence * required_confidence)
        significance = min(1.0, sample_size / required_samples)
        return significance
