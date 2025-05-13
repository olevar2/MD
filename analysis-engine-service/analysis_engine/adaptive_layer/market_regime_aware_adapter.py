"""
Market Regime Aware Adapter

This module provides functionality to adjust parameters based on market regimes.
It serves as a key component of the adaptive layer, enabling strategies to
dynamically respond to changing market conditions.
"""
import logging
from typing import Dict, Any, Optional, Union, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from analysis_engine.services.tool_effectiveness import MarketRegime, TimeFrame
from analysis_engine.services.market_regime_detector import MarketRegimeService
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class MarketRegimeAwareAdapter:
    """
    Adapts strategy parameters based on detected market regimes.
    This component automatically adjusts various aspects of strategies
    to best suit the current market conditions.
    """

    def __init__(self, market_regime_service: MarketRegimeService):
        """
        Initialize the market regime aware adapter.
        
        Args:
            market_regime_service: Service for detecting market regimes
        """
        self.logger = logging.getLogger(__name__)
        self.market_regime_service = market_regime_service
        self._initialize_parameter_adjustments()

    def _initialize_parameter_adjustments(self):
        """Initialize the base parameter adjustments for different regimes"""
        self.risk_adjustments = {MarketRegime.TRENDING_UP: {
            'stop_loss_multiplier': 1.2, 'take_profit_multiplier': 1.5,
            'position_size_multiplier': 1.1, 'risk_reward_min': 1.5},
            MarketRegime.TRENDING_DOWN: {'stop_loss_multiplier': 1.2,
            'take_profit_multiplier': 1.5, 'position_size_multiplier': 1.1,
            'risk_reward_min': 1.5}, MarketRegime.RANGING: {
            'stop_loss_multiplier': 0.9, 'take_profit_multiplier': 0.9,
            'position_size_multiplier': 1.0, 'risk_reward_min': 1.2},
            MarketRegime.VOLATILE: {'stop_loss_multiplier': 1.5,
            'take_profit_multiplier': 1.2, 'position_size_multiplier': 0.7,
            'risk_reward_min': 1.8}, MarketRegime.CHOPPY: {
            'stop_loss_multiplier': 1.3, 'take_profit_multiplier': 1.0,
            'position_size_multiplier': 0.8, 'risk_reward_min': 1.5},
            MarketRegime.BREAKOUT: {'stop_loss_multiplier': 1.0,
            'take_profit_multiplier': 1.3, 'position_size_multiplier': 1.1,
            'risk_reward_min': 1.5}, MarketRegime.UNKNOWN: {
            'stop_loss_multiplier': 1.0, 'take_profit_multiplier': 1.0,
            'position_size_multiplier': 1.0, 'risk_reward_min': 1.3}}
        self.entry_adjustments = {MarketRegime.TRENDING_UP: {
            'confirmation_threshold': 0.7, 'signal_duration_multiplier': 
            1.3}, MarketRegime.TRENDING_DOWN: {'confirmation_threshold': 
            0.7, 'signal_duration_multiplier': 1.3}, MarketRegime.RANGING:
            {'confirmation_threshold': 0.8, 'signal_duration_multiplier': 
            0.8}, MarketRegime.VOLATILE: {'confirmation_threshold': 0.9,
            'signal_duration_multiplier': 0.6}, MarketRegime.CHOPPY: {
            'confirmation_threshold': 0.85, 'signal_duration_multiplier': 
            0.7}, MarketRegime.BREAKOUT: {'confirmation_threshold': 0.75,
            'signal_duration_multiplier': 1.0}, MarketRegime.UNKNOWN: {
            'confirmation_threshold': 0.8, 'signal_duration_multiplier': 1.0}}
        self.indicator_adjustments = {MarketRegime.TRENDING_UP: {
            'ma_period_multiplier': 0.8, 'atr_period_multiplier': 1.0,
            'oscillator_overbought': 80, 'oscillator_oversold': 30,
            'preferred_indicators': ['ema', 'macd', 'adx']}, MarketRegime.
            TRENDING_DOWN: {'ma_period_multiplier': 0.8,
            'atr_period_multiplier': 1.0, 'oscillator_overbought': 70,
            'oscillator_oversold': 20, 'preferred_indicators': ['ema',
            'macd', 'adx']}, MarketRegime.RANGING: {'ma_period_multiplier':
            1.0, 'atr_period_multiplier': 1.1, 'oscillator_overbought': 70,
            'oscillator_oversold': 30, 'preferred_indicators': ['sma',
            'rsi', 'stochastic', 'bollinger_bands']}, MarketRegime.VOLATILE:
            {'ma_period_multiplier': 1.5, 'atr_period_multiplier': 0.7,
            'oscillator_overbought': 75, 'oscillator_oversold': 25,
            'preferred_indicators': ['keltner_channels', 'atr',
            'volatility_bands']}, MarketRegime.CHOPPY: {
            'ma_period_multiplier': 1.3, 'atr_period_multiplier': 1.2,
            'oscillator_overbought': 65, 'oscillator_oversold': 35,
            'preferred_indicators': ['tema', 'hma', 'vidya', 'supertrend']},
            MarketRegime.BREAKOUT: {'ma_period_multiplier': 0.7,
            'atr_period_multiplier': 0.8, 'oscillator_overbought': 80,
            'oscillator_oversold': 20, 'preferred_indicators': [
            'keltner_channels', 'donchian', 'momentum']}, MarketRegime.
            UNKNOWN: {'ma_period_multiplier': 1.0, 'atr_period_multiplier':
            1.0, 'oscillator_overbought': 70, 'oscillator_oversold': 30,
            'preferred_indicators': ['sma', 'ema', 'rsi', 'macd']}}
        self.trade_mgmt_adjustments = {MarketRegime.TRENDING_UP: {
            'trailing_stop_activation': 0.5, 'partial_tp_levels': [0.382, 
            0.618, 1.0], 'time_stop_bars': 20, 'breakeven_threshold': 0.3},
            MarketRegime.TRENDING_DOWN: {'trailing_stop_activation': 0.5,
            'partial_tp_levels': [0.382, 0.618, 1.0], 'time_stop_bars': 20,
            'breakeven_threshold': 0.3}, MarketRegime.RANGING: {
            'trailing_stop_activation': 0.7, 'partial_tp_levels': [0.5, 1.0
            ], 'time_stop_bars': 12, 'breakeven_threshold': 0.5},
            MarketRegime.VOLATILE: {'trailing_stop_activation': 0.3,
            'partial_tp_levels': [0.236, 0.382, 0.618, 1.0],
            'time_stop_bars': 8, 'breakeven_threshold': 0.2}, MarketRegime.
            CHOPPY: {'trailing_stop_activation': 0.5, 'partial_tp_levels':
            [0.5, 1.0], 'time_stop_bars': 10, 'breakeven_threshold': 0.4},
            MarketRegime.BREAKOUT: {'trailing_stop_activation': 0.4,
            'partial_tp_levels': [0.382, 0.618, 1.0, 1.618],
            'time_stop_bars': 15, 'breakeven_threshold': 0.25},
            MarketRegime.UNKNOWN: {'trailing_stop_activation': 0.5,
            'partial_tp_levels': [0.5, 1.0], 'time_stop_bars': 15,
            'breakeven_threshold': 0.3}}

    @with_resilience('get_regime_adjusted_parameters')
    @with_exception_handling
    def get_regime_adjusted_parameters(self, base_parameters: Dict[str, Any
        ], symbol: str, timeframe: str, regime_override: Optional[
        MarketRegime]=None) ->Dict[str, Any]:
        """
        Generate parameters adapted to the current market regime.
        
        Args:
            base_parameters: Base parameters to adjust
            symbol: Trading symbol
            timeframe: Chart timeframe
            regime_override: Optional manual override for market regime
            
        Returns:
            Dictionary with adjusted parameters
        """
        try:
            if regime_override:
                current_regime = regime_override
                regime_certainty = 1.0
            else:
                regime_info = self.market_regime_service.detect_market_regime(
                    symbol, timeframe)
                current_regime = regime_info.get('final_regime',
                    MarketRegime.UNKNOWN)
                regime_certainty = regime_info.get('certainty', 0.7)
            adjusted_parameters = base_parameters.copy()
            risk_params = self.risk_adjustments.get(current_regime, self.
                risk_adjustments[MarketRegime.UNKNOWN])
            for key, adjustment in risk_params.items():
                if key in adjusted_parameters:
                    default_value = adjusted_parameters[key]
                    regime_value = default_value * adjustment
                    adjusted_parameters[key
                        ] = regime_value * regime_certainty + default_value * (
                        1 - regime_certainty)
                else:
                    adjusted_parameters[key] = adjustment
            entry_params = self.entry_adjustments.get(current_regime, self.
                entry_adjustments[MarketRegime.UNKNOWN])
            for key, adjustment in entry_params.items():
                if key in adjusted_parameters:
                    default_value = adjusted_parameters[key]
                    regime_value = default_value * adjustment if isinstance(
                        adjustment, float) else adjustment
                    adjusted_parameters[key
                        ] = regime_value * regime_certainty + default_value * (
                        1 - regime_certainty)
                else:
                    adjusted_parameters[key] = adjustment
            indicator_params = self.indicator_adjustments.get(current_regime,
                self.indicator_adjustments[MarketRegime.UNKNOWN])
            if 'indicators' not in adjusted_parameters:
                adjusted_parameters['indicators'] = {}
            adjusted_parameters['indicators']['preferred'
                ] = indicator_params.get('preferred_indicators')
            if 'ma_period' in adjusted_parameters:
                adjusted_parameters['ma_period'] = int(adjusted_parameters[
                    'ma_period'] * indicator_params.get(
                    'ma_period_multiplier', 1.0))
            if 'atr_period' in adjusted_parameters:
                adjusted_parameters['atr_period'] = int(adjusted_parameters
                    ['atr_period'] * indicator_params.get(
                    'atr_period_multiplier', 1.0))
            if 'oscillator_levels' in adjusted_parameters:
                adjusted_parameters['oscillator_levels']['overbought'
                    ] = indicator_params.get('oscillator_overbought', 70)
                adjusted_parameters['oscillator_levels']['oversold'
                    ] = indicator_params.get('oscillator_oversold', 30)
            else:
                adjusted_parameters['oscillator_levels'] = {'overbought':
                    indicator_params.get('oscillator_overbought', 70),
                    'oversold': indicator_params.get('oscillator_oversold', 30)
                    }
            trade_params = self.trade_mgmt_adjustments.get(current_regime,
                self.trade_mgmt_adjustments[MarketRegime.UNKNOWN])
            if 'trade_management' not in adjusted_parameters:
                adjusted_parameters['trade_management'] = {}
            adjusted_parameters['trade_management']['trailing_stop_activation'
                ] = trade_params.get('trailing_stop_activation')
            adjusted_parameters['trade_management']['partial_tp_levels'
                ] = trade_params.get('partial_tp_levels')
            adjusted_parameters['trade_management']['time_stop_bars'
                ] = trade_params.get('time_stop_bars')
            adjusted_parameters['trade_management']['breakeven_threshold'
                ] = trade_params.get('breakeven_threshold')
            adjusted_parameters['market_regime'] = {'current_regime': 
                current_regime.value if hasattr(current_regime, 'value') else
                str(current_regime), 'regime_certainty': regime_certainty,
                'adjusted_at': datetime.now().isoformat()}
            return adjusted_parameters
        except Exception as e:
            self.logger.error(
                f'Error adjusting parameters for market regime: {str(e)}',
                exc_info=True)
            return base_parameters

    def apply_volatility_adjustments(self, parameters: Dict[str, Any],
        recent_volatility: float, average_volatility: float) ->Dict[str, Any]:
        """
        Apply volatility-based adjustments to parameters.
        
        Args:
            parameters: Current parameters
            recent_volatility: Recent market volatility (e.g., ATR)
            average_volatility: Average historical volatility
            
        Returns:
            Dictionary with volatility-adjusted parameters
        """
        adjusted = parameters.copy()
        volatility_ratio = (recent_volatility / average_volatility if 
            average_volatility > 0 else 1.0)
        if 'stop_loss_pips' in adjusted:
            adjusted['stop_loss_pips'] = adjusted['stop_loss_pips'
                ] * volatility_ratio
        if 'take_profit_pips' in adjusted:
            adjusted['take_profit_pips'] = adjusted['take_profit_pips'
                ] * volatility_ratio
        if 'position_size' in adjusted and volatility_ratio > 1.0:
            adjusted['position_size'] = adjusted['position_size'
                ] / volatility_ratio
        if 'market_conditions' not in adjusted:
            adjusted['market_conditions'] = {}
        adjusted['market_conditions']['volatility_ratio'] = volatility_ratio
        adjusted['market_conditions']['volatility_adjusted'] = True
        return adjusted

    @with_resilience('get_recommended_strategy_types')
    def get_recommended_strategy_types(self, market_regime: MarketRegime
        ) ->List[str]:
        """
        Get recommended strategy types for the given market regime.
        
        Args:
            market_regime: Current market regime
            
        Returns:
            List of recommended strategy types
        """
        strategy_recommendations = {MarketRegime.TRENDING_UP: [
            'trend_following', 'breakout', 'momentum',
            'elliott_wave_impulse', 'fibonacci_extension'], MarketRegime.
            TRENDING_DOWN: ['trend_following', 'breakout', 'momentum',
            'elliott_wave_impulse', 'fibonacci_extension'], MarketRegime.
            RANGING: ['mean_reversion', 'support_resistance',
            'harmonic_pattern', 'elliott_wave_correction',
            'fibonacci_retracement'], MarketRegime.VOLATILE: [
            'volatility_expansion', 'options_based',
            'multi_timeframe_confluence', 'breakout_with_filter',
            'statistical_arbitrage'], MarketRegime.CHOPPY: ['filter_based',
            'multi_timeframe', 'statistical_edge', 'ml_pattern_recognition',
            'adaptive_moving_average'], MarketRegime.BREAKOUT: ['breakout',
            'momentum', 'fibonacci_extension', 'volatility_expansion',
            'multi_timeframe_confluence'], MarketRegime.UNKNOWN: [
            'multi_timeframe_confluence', 'adaptive_moving_average',
            'statistical_edge', 'filter_based']}
        return strategy_recommendations.get(market_regime,
            strategy_recommendations[MarketRegime.UNKNOWN])

    @with_resilience('get_strategy_specific_adjustments')
    def get_strategy_specific_adjustments(self, strategy_type: str,
        market_regime: MarketRegime) ->Dict[str, Any]:
        """
        Get strategy-specific parameter adjustments for the given market regime.
        
        Args:
            strategy_type: Type of strategy
            market_regime: Current market regime
            
        Returns:
            Dictionary with strategy-specific parameter adjustments
        """
        return {}
