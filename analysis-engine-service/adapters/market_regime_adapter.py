"""
Market Regime Aware Adapter

This module provides functionality to adapt strategy parameters based on 
the current market regime, historical performance data, and statistical analysis.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

from analysis_engine.services.tool_effectiveness import MarketRegime


class AdaptationStrength(str, Enum):
    """Defines the strength of adaptation to apply"""
    MINIMAL = "minimal"     # Very small parameter adjustments
    CONSERVATIVE = "conservative"  # Conservative parameter adjustments
    MODERATE = "moderate"   # Moderate parameter adjustments
    AGGRESSIVE = "aggressive"  # Larger parameter adjustments
    EXPERIMENTAL = "experimental"  # Experimental, potentially higher risk adjustments


class MarketRegimeAwareAdapter:
    """
    Adapts strategy parameters based on the current market regime and historical 
    performance data, applying statistically validated adjustments.
    """
    
    def __init__(self, default_adaptation_strength: AdaptationStrength = AdaptationStrength.MODERATE):
        """
        Initialize the market regime adapter.
        
        Args:
            default_adaptation_strength: Default adaptation strength to use
        """
        self.logger = logging.getLogger(__name__)
        self.default_adaptation_strength = default_adaptation_strength
        
        # Base adaptation factors by strength level
        self.adaptation_factors = {
            AdaptationStrength.MINIMAL: 0.05,     # 5% adjustment
            AdaptationStrength.CONSERVATIVE: 0.10,  # 10% adjustment
            AdaptationStrength.MODERATE: 0.20,    # 20% adjustment
            AdaptationStrength.AGGRESSIVE: 0.35,  # 35% adjustment
            AdaptationStrength.EXPERIMENTAL: 0.50  # 50% adjustment
        }
        
    def adapt_for_regime(
        self,
        current_parameters: Dict[str, Any],
        market_regime: MarketRegime,
        regime_certainty: float,
        effectiveness_metrics: Dict[str, Any],
        adaptation_strength: AdaptationStrength = None
    ) -> Dict[str, Any]:
        """
        Adapt parameters based on the current market regime.
        
        Args:
            current_parameters: Current strategy parameters
            market_regime: Detected market regime
            regime_certainty: Confidence in the regime detection (0.0-1.0)
            effectiveness_metrics: Tool effectiveness metrics
            adaptation_strength: Strength of adaptation to apply (default: self.default_adaptation_strength)
            
        Returns:
            Adapted parameter dictionary
        """
        if adaptation_strength is None:
            adaptation_strength = self.default_adaptation_strength
            
        # Copy parameters to avoid modifying the original
        params = current_parameters.copy()
        
        # Select appropriate adaptation function based on market regime
        adaptation_func = self._get_adaptation_function(market_regime)
        
        # Apply the adaptation with appropriate strength
        adapted_params = adaptation_func(
            params, 
            effectiveness_metrics, 
            adaptation_strength,
            regime_certainty
        )
        
        # Log the adaptation for analysis
        self._log_adaptation(current_parameters, adapted_params, market_regime, adaptation_strength)
        
        return adapted_params
        
    def _get_adaptation_function(self, market_regime: MarketRegime):
        """
        Get the appropriate adaptation function for the given market regime.
        
        Args:
            market_regime: Market regime to adapt for
            
        Returns:
            Adaptation function for the regime
        """
        adaptation_map = {
            MarketRegime.TRENDING_UP: self._adapt_for_bullish_trend,
            MarketRegime.TRENDING_DOWN: self._adapt_for_bearish_trend,
            MarketRegime.RANGING: self._adapt_for_ranging_market,
            MarketRegime.VOLATILE: self._adapt_for_volatile_market,
            MarketRegime.BREAKOUT: self._adapt_for_breakout_market,
            MarketRegime.REVERSAL: self._adapt_for_reversal_market,
            MarketRegime.CHOPPY: self._adapt_for_choppy_market
        }
        
        return adaptation_map.get(market_regime, self._adapt_default)
        
    def _adapt_for_bullish_trend(
        self,
        params: Dict[str, Any],
        effectiveness_metrics: Dict[str, Any],
        adaptation_strength: AdaptationStrength,
        regime_certainty: float
    ) -> Dict[str, Any]:
        """
        Adapt parameters for bullish trending market.
        
        Args:
            params: Current parameters
            effectiveness_metrics: Effectiveness metrics
            adaptation_strength: Strength of adaptation
            regime_certainty: Confidence in regime detection
            
        Returns:
            Adapted parameters
        """
        # Get adaptation factor based on strength and certainty
        factor = self._calculate_adaptation_factor(adaptation_strength, regime_certainty)
        
        # Adjust tool weights if present
        if 'tool_weights' in params:
            weights = params['tool_weights'].copy()
            
            # Increase weight for trend-following tools
            for tool_id, weight in weights.items():
                # Check if we have effectiveness data for this tool
                tool_metrics = effectiveness_metrics.get('tools', {}).get(tool_id)
                
                if tool_id.startswith('trend_'):
                    # Increase weights for trend tools
                    weights[tool_id] = weight * (1 + 0.3 * factor)
                elif tool_id.startswith('osc_'):
                    # Decrease weights for oscillators in strong trends
                    weights[tool_id] = weight * (1 - 0.3 * factor)
                    
                # Further adjust based on tool effectiveness in this regime
                if tool_metrics:
                    regime_performance = tool_metrics.get('regime_metrics', {}).get(
                        str(MarketRegime.TRENDING_UP.value), {}
                    )
                    if regime_performance:
                        win_rate = regime_performance.get('win_rate', 50)
                        if win_rate > 60:  # Tool performs well in this regime
                            weights[tool_id] *= 1 + (win_rate - 60) / 100
                            
            params['tool_weights'] = weights
            
        # Adjust stop loss and take profit for trending markets
        if 'stop_loss_pips' in params and 'take_profit_pips' in params:
            # In bullish trends, consider wider stops and larger targets
            params['stop_loss_pips'] = params['stop_loss_pips'] * (1 + 0.1 * factor)
            params['take_profit_pips'] = params['take_profit_pips'] * (1 + 0.2 * factor)
            
        # Adjust entry parameters for bullish trend
        if 'entry_threshold' in params:
            # Lower entry threshold to capture more of the trend
            params['entry_threshold'] = params['entry_threshold'] * (1 - 0.15 * factor)
            
        # Adjust position sizing for trends
        if 'position_size_factor' in params:
            # Increase position size in established trends
            params['position_size_factor'] = params['position_size_factor'] * (1 + 0.1 * factor)
            
        return params
        
    def _adapt_for_bearish_trend(
        self,
        params: Dict[str, Any],
        effectiveness_metrics: Dict[str, Any],
        adaptation_strength: AdaptationStrength,
        regime_certainty: float
    ) -> Dict[str, Any]:
        """
        Adapt parameters for bearish trending market.
        
        Args:
            params: Current parameters
            effectiveness_metrics: Effectiveness metrics
            adaptation_strength: Strength of adaptation
            regime_certainty: Confidence in regime detection
            
        Returns:
            Adapted parameters
        """
        # Get adaptation factor based on strength and certainty
        factor = self._calculate_adaptation_factor(adaptation_strength, regime_certainty)
        
        # Adjust tool weights if present
        if 'tool_weights' in params:
            weights = params['tool_weights'].copy()
            
            # Increase weight for trend-following tools
            for tool_id, weight in weights.items():
                # Check if we have effectiveness data for this tool
                tool_metrics = effectiveness_metrics.get('tools', {}).get(tool_id)
                
                if tool_id.startswith('trend_'):
                    # Increase weights for trend tools
                    weights[tool_id] = weight * (1 + 0.3 * factor)
                elif tool_id.startswith('osc_'):
                    # Decrease weights for oscillators in strong trends
                    weights[tool_id] = weight * (1 - 0.3 * factor)
                    
                # Further adjust based on tool effectiveness in this regime
                if tool_metrics:
                    regime_performance = tool_metrics.get('regime_metrics', {}).get(
                        str(MarketRegime.TRENDING_DOWN.value), {}
                    )
                    if regime_performance:
                        win_rate = regime_performance.get('win_rate', 50)
                        if win_rate > 60:  # Tool performs well in this regime
                            weights[tool_id] *= 1 + (win_rate - 60) / 100
                            
            params['tool_weights'] = weights
            
        # Adjust stop loss and take profit for trending markets
        if 'stop_loss_pips' in params and 'take_profit_pips' in params:
            # In bearish trends, consider wider stops and larger targets
            params['stop_loss_pips'] = params['stop_loss_pips'] * (1 + 0.1 * factor)
            params['take_profit_pips'] = params['take_profit_pips'] * (1 + 0.2 * factor)
            
        # Adjust entry parameters for bearish trend
        if 'entry_threshold' in params:
            # Lower entry threshold to capture more of the trend
            params['entry_threshold'] = params['entry_threshold'] * (1 - 0.15 * factor)
            
        # Adjust position sizing for trends
        if 'position_size_factor' in params:
            # Increase position size in established trends
            params['position_size_factor'] = params['position_size_factor'] * (1 + 0.1 * factor)
            
        return params
        
    def _adapt_for_ranging_market(
        self,
        params: Dict[str, Any],
        effectiveness_metrics: Dict[str, Any],
        adaptation_strength: AdaptationStrength,
        regime_certainty: float
    ) -> Dict[str, Any]:
        """
        Adapt parameters for ranging market.
        
        Args:
            params: Current parameters
            effectiveness_metrics: Effectiveness metrics
            adaptation_strength: Strength of adaptation
            regime_certainty: Confidence in regime detection
            
        Returns:
            Adapted parameters
        """
        # Get adaptation factor based on strength and certainty
        factor = self._calculate_adaptation_factor(adaptation_strength, regime_certainty)
        
        # In ranging markets, decrease trend sensitivity
        if 'trend_sensitivity' in params:
            params['trend_sensitivity'] = params['trend_sensitivity'] * (1 - 0.2 * factor)
            
        # Adjust tool weights if present
        if 'tool_weights' in params:
            weights = params['tool_weights'].copy()
            
            # Favor oscillators over trend-following tools
            for tool_id, weight in weights.items():
                # Check if we have effectiveness data for this tool
                tool_metrics = effectiveness_metrics.get('tools', {}).get(tool_id)
                
                if tool_id.startswith('osc_'):
                    # Increase weights for oscillator tools
                    weights[tool_id] = weight * (1 + 0.4 * factor)
                elif tool_id.startswith('trend_'):
                    # Decrease weights for trend tools in ranging markets
                    weights[tool_id] = weight * (1 - 0.3 * factor)
                    
                # Further adjust based on tool effectiveness in this regime
                if tool_metrics:
                    regime_performance = tool_metrics.get('regime_metrics', {}).get(
                        str(MarketRegime.RANGING.value), {}
                    )
                    if regime_performance:
                        win_rate = regime_performance.get('win_rate', 50)
                        if win_rate > 60:  # Tool performs well in this regime
                            weights[tool_id] *= 1 + (win_rate - 60) / 100
                            
            params['tool_weights'] = weights
            
        # Adjust stop loss and take profit for ranging markets
        if 'stop_loss_pips' in params and 'take_profit_pips' in params:
            # In ranging markets, tighter stops and smaller targets
            params['stop_loss_pips'] = params['stop_loss_pips'] * (1 - 0.1 * factor)
            params['take_profit_pips'] = params['take_profit_pips'] * (1 - 0.1 * factor)
            
        # Adjust overbought/oversold levels for oscillators
        if 'overbought_level' in params and 'oversold_level' in params:
            # Make overbought/oversold levels more conservative in ranging markets
            params['overbought_level'] = params['overbought_level'] * (1 - 0.05 * factor)
            params['oversold_level'] = params['oversold_level'] * (1 - 0.05 * factor)
            
        return params
        
    def _adapt_for_volatile_market(
        self,
        params: Dict[str, Any],
        effectiveness_metrics: Dict[str, Any],
        adaptation_strength: AdaptationStrength,
        regime_certainty: float
    ) -> Dict[str, Any]:
        """
        Adapt parameters for volatile market.
        
        Args:
            params: Current parameters
            effectiveness_metrics: Effectiveness metrics
            adaptation_strength: Strength of adaptation
            regime_certainty: Confidence in regime detection
            
        Returns:
            Adapted parameters
        """
        # Get adaptation factor based on strength and certainty
        factor = self._calculate_adaptation_factor(adaptation_strength, regime_certainty)
        
        # In volatile markets, focus on volatility-based tools
        if 'tool_weights' in params:
            weights = params['tool_weights'].copy()
            
            for tool_id, weight in weights.items():
                # Check if we have effectiveness data for this tool
                tool_metrics = effectiveness_metrics.get('tools', {}).get(tool_id)
                
                if tool_id.startswith('vol_') or 'atr' in tool_id:
                    # Increase weights for volatility tools
                    weights[tool_id] = weight * (1 + 0.3 * factor)
                    
                # Further adjust based on tool effectiveness in this regime
                if tool_metrics:
                    regime_performance = tool_metrics.get('regime_metrics', {}).get(
                        str(MarketRegime.VOLATILE.value), {}
                    )
                    if regime_performance:
                        win_rate = regime_performance.get('win_rate', 50)
                        if win_rate > 60:  # Tool performs well in this regime
                            weights[tool_id] *= 1 + (win_rate - 60) / 100
                            
            params['tool_weights'] = weights
            
        # Adjust stop loss and take profit for volatile markets
        if 'stop_loss_pips' in params and 'take_profit_pips' in params:
            # In volatile markets, wider stops to avoid premature exit
            params['stop_loss_pips'] = params['stop_loss_pips'] * (1 + 0.2 * factor)
            
            # Also adjust take profit for volatility
            atr_multiplier = params.get('atr_stop_multiplier', 3.0)
            if atr_multiplier > 0:
                # Use ATR for dynamic stop loss if available
                params['stop_loss_atr_multiplier'] = atr_multiplier * (1 + 0.15 * factor)
                
        # Reduce position size in volatile markets
        if 'position_size_factor' in params:
            params['position_size_factor'] = params['position_size_factor'] * (1 - 0.25 * factor)
            
        # Adjust moving average periods
        if 'fast_ma_period' in params and 'slow_ma_period' in params:
            # Use longer periods to filter out noise
            params['fast_ma_period'] = params['fast_ma_period'] * (1 + 0.15 * factor)
            params['slow_ma_period'] = params['slow_ma_period'] * (1 + 0.1 * factor)
            
        return params
        
    def _adapt_for_breakout_market(
        self,
        params: Dict[str, Any],
        effectiveness_metrics: Dict[str, Any],
        adaptation_strength: AdaptationStrength,
        regime_certainty: float
    ) -> Dict[str, Any]:
        """
        Adapt parameters for breakout market.
        
        Args:
            params: Current parameters
            effectiveness_metrics: Effectiveness metrics
            adaptation_strength: Strength of adaptation
            regime_certainty: Confidence in regime detection
            
        Returns:
            Adapted parameters
        """
        # Get adaptation factor based on strength and certainty
        factor = self._calculate_adaptation_factor(adaptation_strength, regime_certainty)
        
        # In breakout markets, focus on momentum tools
        if 'tool_weights' in params:
            weights = params['tool_weights'].copy()
            
            for tool_id, weight in weights.items():
                # Check if we have effectiveness data for this tool
                tool_metrics = effectiveness_metrics.get('tools', {}).get(tool_id)
                
                if tool_id.startswith('breakout_') or 'momentum' in tool_id:
                    # Increase weights for breakout/momentum tools
                    weights[tool_id] = weight * (1 + 0.5 * factor)
                    
                # Further adjust based on tool effectiveness in this regime
                if tool_metrics:
                    regime_performance = tool_metrics.get('regime_metrics', {}).get(
                        str(MarketRegime.BREAKOUT.value), {}
                    )
                    if regime_performance:
                        win_rate = regime_performance.get('win_rate', 50)
                        if win_rate > 60:  # Tool performs well in this regime
                            weights[tool_id] *= 1 + (win_rate - 60) / 100
                            
            params['tool_weights'] = weights
            
        # Adjust stop loss and take profit for breakout markets
        if 'stop_loss_pips' in params and 'take_profit_pips' in params:
            # In breakout markets, tighter stops and larger targets for momentum capture
            params['stop_loss_pips'] = params['stop_loss_pips'] * (1 - 0.1 * factor)
            params['take_profit_pips'] = params['take_profit_pips'] * (1 + 0.3 * factor)
            
        # Adjust confirmation thresholds for breakouts
        if 'breakout_confirmation_threshold' in params:
            # Less confirmation needed to catch breakouts early
            params['breakout_confirmation_threshold'] = params['breakout_confirmation_threshold'] * (1 - 0.2 * factor)
            
        # Increase sensitivity to volume spikes
        if 'volume_spike_threshold' in params:
            params['volume_spike_threshold'] = params['volume_spike_threshold'] * (1 - 0.15 * factor)
            
        return params
        
    def _adapt_for_reversal_market(
        self,
        params: Dict[str, Any],
        effectiveness_metrics: Dict[str, Any],
        adaptation_strength: AdaptationStrength,
        regime_certainty: float
    ) -> Dict[str, Any]:
        """
        Adapt parameters for reversal market conditions.
        
        Args:
            params: Current parameters
            effectiveness_metrics: Effectiveness metrics
            adaptation_strength: Strength of adaptation
            regime_certainty: Confidence in regime detection
            
        Returns:
            Adapted parameters
        """
        # Get adaptation factor based on strength and certainty
        factor = self._calculate_adaptation_factor(adaptation_strength, regime_certainty)
        
        # In reversal markets, focus on confirmation before entry
        if 'reversal_confirmation_threshold' in params:
            params['reversal_confirmation_threshold'] = params['reversal_confirmation_threshold'] * (1 + 0.1 * factor)
        
        # Adjust tool weights if present
        if 'tool_weights' in params:
            weights = params['tool_weights'].copy()
            
            for tool_id, weight in weights.items():
                # Check if we have effectiveness data for this tool
                tool_metrics = effectiveness_metrics.get('tools', {}).get(tool_id)
                
                if tool_id.startswith('reversal_') or 'divergence' in tool_id:
                    # Increase weights for reversal pattern tools
                    weights[tool_id] = weight * (1 + 0.6 * factor)
                elif tool_id.startswith('trend_'):
                    # Decrease trend-following weights
                    weights[tool_id] = weight * (1 - 0.5 * factor)
                    
                # Further adjust based on tool effectiveness in this regime
                if tool_metrics:
                    regime_performance = tool_metrics.get('regime_metrics', {}).get(
                        str(MarketRegime.REVERSAL.value), {}
                    )
                    if regime_performance:
                        win_rate = regime_performance.get('win_rate', 50)
                        if win_rate > 60:  # Tool performs well in this regime
                            weights[tool_id] *= 1 + (win_rate - 60) / 100
                            
            params['tool_weights'] = weights
            
        # Adjust risk management for reversal markets
        if 'position_size_factor' in params:
            params['position_size_factor'] = params['position_size_factor'] * (1 - 0.2 * factor)
            
        # Adjust stop loss and take profit for reversal markets
        if 'stop_loss_pips' in params and 'take_profit_pips' in params:
            # In reversal markets, tighter stops for early exit if reversal fails
            params['stop_loss_pips'] = params['stop_loss_pips'] * (1 - 0.15 * factor)
            
        # Increase divergence sensitivity
        if 'divergence_threshold' in params:
            params['divergence_threshold'] = params['divergence_threshold'] * (1 - 0.1 * factor)
            
        return params
        
    def _adapt_for_choppy_market(
        self,
        params: Dict[str, Any],
        effectiveness_metrics: Dict[str, Any],
        adaptation_strength: AdaptationStrength,
        regime_certainty: float
    ) -> Dict[str, Any]:
        """
        Adapt parameters for choppy market conditions.
        
        Args:
            params: Current parameters
            effectiveness_metrics: Effectiveness metrics
            adaptation_strength: Strength of adaptation
            regime_certainty: Confidence in regime detection
            
        Returns:
            Adapted parameters
        """
        # Get adaptation factor based on strength and certainty
        factor = self._calculate_adaptation_factor(adaptation_strength, regime_certainty)
        
        # In choppy markets, reduce position size and increase filtering
        if 'position_size_factor' in params:
            params['position_size_factor'] = params['position_size_factor'] * (1 - 0.3 * factor)
            
        # Adjust tool weights if present
        if 'tool_weights' in params:
            weights = params['tool_weights'].copy()
            
            # In choppy markets, reduce all weights to minimize trading
            for tool_id, weight in weights.items():
                # Check if we have effectiveness data for this tool
                tool_metrics = effectiveness_metrics.get('tools', {}).get(tool_id)
                
                # Reduce all weights by default
                weights[tool_id] = weight * (1 - 0.2 * factor)
                
                # But favor tools that specifically filter noise
                if 'filter' in tool_id or 'smooth' in tool_id:
                    weights[tool_id] = weight * (1 + 0.3 * factor)
                
                # Further adjust based on tool effectiveness in this regime
                if tool_metrics:
                    regime_performance = tool_metrics.get('regime_metrics', {}).get(
                        str(MarketRegime.CHOPPY.value), {}
                    )
                    if regime_performance:
                        win_rate = regime_performance.get('win_rate', 50)
                        if win_rate > 60:  # Tool performs well in this regime
                            weights[tool_id] *= 1 + (win_rate - 60) / 100
                            
            params['tool_weights'] = weights
                        
        # Increase moving average periods to filter out noise
        if 'fast_ma_period' in params:
            params['fast_ma_period'] = params['fast_ma_period'] * (1 + 0.25 * factor)
            
        if 'slow_ma_period' in params:
            params['slow_ma_period'] = params['slow_ma_period'] * (1 + 0.25 * factor)
            
        # Widen oscillator thresholds
        if 'overbought_level' in params:
            params['overbought_level'] = params['overbought_level'] * (1 + 0.1 * factor)
            
        if 'oversold_level' in params:
            params['oversold_level'] = params['oversold_level'] * (1 + 0.1 * factor)
            
        # Increase signal thresholds to filter weak signals
        if 'signal_threshold' in params:
            params['signal_threshold'] = params['signal_threshold'] * (1 + 0.2 * factor)
            
        return params
        
    def _adapt_default(
        self,
        params: Dict[str, Any],
        effectiveness_metrics: Dict[str, Any],
        adaptation_strength: AdaptationStrength,
        regime_certainty: float
    ) -> Dict[str, Any]:
        """
        Default adaptation strategy when specific market regime is unknown.
        
        Args:
            params: Current parameters
            effectiveness_metrics: Effectiveness metrics
            adaptation_strength: Strength of adaptation
            regime_certainty: Confidence in regime detection
            
        Returns:
            Adapted parameters
        """
        # Apply minimal adjustments when regime is uncertain
        factor = self._calculate_adaptation_factor(adaptation_strength, regime_certainty) * 0.5
        
        # Use overall most effective tools without regime bias
        if 'tool_weights' in params and effectiveness_metrics:
            weights = params['tool_weights'].copy()
            tools = effectiveness_metrics.get('tools', {})
            
            # Find top performing tools
            tool_scores = [(tool_id, metrics.get('win_rate', 50)) 
                          for tool_id, metrics in tools.items()]
            
            # Sort by win rate
            tool_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Slightly increase weights for top tools regardless of type
            for tool_id, win_rate in tool_scores[:3]:  # Focus on top 3 tools
                if tool_id in weights:
                    weights[tool_id] = weights[tool_id] * (1 + 0.2 * factor)
                    
            params['tool_weights'] = weights
            
        # Use balanced risk settings
        if 'position_size_factor' in params:
            # Slightly conservative position sizing when regime is unclear
            params['position_size_factor'] = params['position_size_factor'] * (1 - 0.1 * factor)
            
        return params
        
    def _calculate_adaptation_factor(
        self, 
        adaptation_strength: AdaptationStrength,
        regime_certainty: float
    ) -> float:
        """
        Calculate adjustment factor based on adaptation strength and regime certainty.
        
        Args:
            adaptation_strength: Strength of adaptation
            regime_certainty: Confidence in regime detection (0.0-1.0)
            
        Returns:
            Adjustment factor (0.0-1.0)
        """
        base_factor = self.adaptation_factors.get(
            adaptation_strength, 
            self.adaptation_factors[AdaptationStrength.MODERATE]
        )
        
        # Scale by certainty - lower certainty means less aggressive adaptation
        return base_factor * regime_certainty
        
    def _log_adaptation(
        self,
        original_params: Dict[str, Any],
        adapted_params: Dict[str, Any],
        market_regime: MarketRegime,
        adaptation_strength: AdaptationStrength
    ) -> None:
        """
        Log parameter adaptations for analysis and tracking.
        
        Args:
            original_params: Original parameters before adaptation
            adapted_params: Parameters after adaptation
            market_regime: Market regime adaptation was applied for
            adaptation_strength: Strength of adaptation applied
        """
        changes = {}
        for key in adapted_params:
            if key in original_params:
                if original_params[key] != adapted_params[key]:
                    if isinstance(original_params[key], (int, float)) and isinstance(adapted_params[key], (int, float)):
                        pct_change = ((adapted_params[key] - original_params[key]) / original_params[key]) * 100
                        changes[key] = f"{pct_change:.1f}%"
                    else:
                        changes[key] = "changed"
        
        if changes:
            self.logger.info(
                f"Adapted parameters for {market_regime.value} regime "
                f"(strength: {adaptation_strength.value}): {changes}"
            )
