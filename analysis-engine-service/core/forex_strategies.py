"""
Forex Asset Strategy Implementations

This module contains implementations of forex-specific trading strategies
that integrate with all analysis components.
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import numpy as np
import pandas as pd
from analysis_engine.multi_asset.asset_strategy_framework import BaseAssetStrategy, AssetStrategyType
from analysis_engine.multi_asset.asset_registry import AssetClass
from analysis_engine.integration.analysis_integration_service import AnalysisIntegrationService
from analysis_engine.models.market_data import MarketData
logger = logging.getLogger(__name__)


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class ForexTrendStrategy(BaseAssetStrategy):
    """
    Forex-specific trend-following strategy
    
    This strategy focuses on capturing medium to long-term trends in forex markets.
    It integrates multiple timeframe analysis, sentiment analysis, and pattern recognition.
    """

    def __init__(self, analysis_service: AnalysisIntegrationService=None,
        config: Dict[str, Any]=None):
        """
        Initialize a forex trend strategy
        
        Args:
            analysis_service: Analysis integration service
            config: Strategy configuration
        """
        super().__init__(strategy_type=AssetStrategyType.FOREX_TREND,
            asset_class=AssetClass.FOREX, analysis_service=analysis_service,
            config=config or {})
        self.default_config = {'timeframes': ['15m', '1h', '4h', '1d'],
            'primary_timeframe': '4h', 'trend_detection': {'ema_fast': 8,
            'ema_slow': 21, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal':
            9, 'atr_period': 14}, 'entry_filters': {'min_trend_strength': 
            0.6, 'min_signal_confidence': 0.7, 'min_pullback': 0.3,
            'rsi_oversold': 30, 'rsi_overbought': 70}, 'exit_rules': {
            'trailing_stop_atr_mult': 2.0, 'take_profit_atr_mult': 3.0,
            'max_holding_periods': 20}, 'risk_management': {
            'max_risk_per_trade': 0.01, 'max_correlated_exposure': 0.05,
            'session_filters_enabled': True}, 'currency_correlations': {
            'check_correlations': True, 'correlation_threshold': 0.7},
            'session_preferences': {'london_weight': 1.0, 'ny_weight': 1.0,
            'asia_weight': 0.7, 'sydney_weight': 0.7}}
        self.config = {**self.default_config, **self.config}
        self.regime_params = {'trending': {'entry_filters': {
            'min_trend_strength': 0.5, 'min_signal_confidence': 0.6,
            'min_pullback': 0.2}, 'exit_rules': {'trailing_stop_atr_mult': 
            2.5, 'take_profit_atr_mult': 4.0}}, 'ranging': {'entry_filters':
            {'min_trend_strength': 0.7, 'min_signal_confidence': 0.8,
            'min_pullback': 0.4}, 'exit_rules': {'trailing_stop_atr_mult': 
            1.5, 'take_profit_atr_mult': 2.0}}, 'volatile': {
            'entry_filters': {'min_trend_strength': 0.8,
            'min_signal_confidence': 0.85, 'min_pullback': 0.5},
            'exit_rules': {'trailing_stop_atr_mult': 3.0,
            'take_profit_atr_mult': 3.0}}}

    async def analyze(self, symbol: str, market_data: Dict[str, MarketData]
        ) ->Dict[str, Any]:
        """
        Analyze market data and generate trading signals
        
        Args:
            symbol: Asset symbol
            market_data: Dictionary of market data by timeframe
            
        Returns:
            Dictionary with analysis results and signals
        """
        if not self.validate_asset(symbol):
            return {'error':
                f'Symbol {symbol} is not a valid forex pair for this strategy'}
        components = self.get_required_components()
        analysis_results = await self.analysis_service.analyze_asset(symbol
            =symbol, market_data=market_data, include_components=components)
        if 'error' in analysis_results:
            return {'error': analysis_results['error']}
        market_regime = self._detect_market_regime(analysis_results)
        params = self.get_strategy_parameters(market_regime)
        adjusted_params = self.adjust_parameters(params, analysis_results)
        signals = self._generate_signals(symbol, analysis_results,
            adjusted_params)
        if signals['signal'] != 'neutral':
            signals['position_size'] = self.get_position_sizing(signals[
                'strength'], signals['confidence'])
        signals['market_regime'] = market_regime
        signals['parameters'] = adjusted_params
        signals['forex_specific'] = self._get_forex_specific_insights(symbol,
            analysis_results)
        return signals

    @with_resilience('get_strategy_parameters')
    def get_strategy_parameters(self, market_regime: str) ->Dict[str, Any]:
        """
        Get strategy parameters based on market regime
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Dictionary with strategy parameters
        """
        params = {**self.config}
        if market_regime in self.regime_params:
            regime_specific = self.regime_params[market_regime]
            for key, value in regime_specific.items():
                if isinstance(value, dict) and key in params and isinstance(
                    params[key], dict):
                    params[key] = {**params[key], **value}
                else:
                    params[key] = value
        return params

    def adjust_parameters(self, params: Dict[str, Any], market_context:
        Dict[str, Any]) ->Dict[str, Any]:
        """
        Adjust strategy parameters based on market context
        
        Args:
            params: Current strategy parameters
            market_context: Market context information
            
        Returns:
            Adjusted parameters
        """
        adjusted = {**params}
        if ('asset_specific' in market_context and 'session_activity' in
            market_context['asset_specific']):
            session_activity = market_context['asset_specific'][
                'session_activity']
            active_session = session_activity.get('active_session', 'none')
            if active_session == 'london_open' or active_session == 'ny_open':
                if 'entry_filters' in adjusted:
                    adjusted['entry_filters']['min_signal_confidence'] *= 0.9
            elif active_session == 'asia' or active_session == 'sydney':
                if 'entry_filters' in adjusted:
                    adjusted['entry_filters']['min_signal_confidence'] *= 1.1
        if ('asset_specific' in market_context and 'spread_viability' in
            market_context['asset_specific']):
            if not market_context['asset_specific']['spread_viability']:
                if 'entry_filters' in adjusted:
                    adjusted['entry_filters']['min_signal_confidence'] *= 1.2
                    adjusted['entry_filters']['min_trend_strength'] *= 1.2
        return adjusted

    @with_resilience('get_position_sizing')
    def get_position_sizing(self, signal_strength: float, confidence: float
        ) ->float:
        """
        Calculate position sizing based on signal strength and confidence
        
        Args:
            signal_strength: Strength of the trading signal
            confidence: Confidence in the signal
            
        Returns:
            Position size as a percentage of available capital
        """
        max_risk = self.config['risk_management']['max_risk_per_trade']
        position_scale = signal_strength * confidence
        position_size = max_risk * position_scale
        return min(position_size, max_risk)

    @with_resilience('get_required_components')
    def get_required_components(self) ->List[str]:
        """
        Get list of required analysis components for this strategy
        
        Returns:
            List of component names
        """
        return ['technical', 'pattern', 'multi_timeframe', 'sentiment',
            'market_regime']

    def _detect_market_regime(self, analysis_results: Dict[str, Any]) ->str:
        """Detect market regime from analysis results"""
        if ('components' in analysis_results and 'market_regime' in
            analysis_results['components']):
            regime = analysis_results['components']['market_regime'].get(
                'regime')
            if regime:
                return regime
        return 'trending'

    def _generate_signals(self, symbol: str, analysis_results: Dict[str,
        Any], params: Dict[str, Any]) ->Dict[str, Any]:
        """Generate trading signals from analysis results"""
        signal = {'symbol': symbol, 'signal': 'neutral', 'strength': 0.0,
            'confidence': 0.0, 'timestamp': datetime.now().isoformat()}
        if 'overall_signal' in analysis_results:
            signal['signal'] = analysis_results['overall_signal']
        if 'overall_confidence' in analysis_results:
            signal['confidence'] = analysis_results['overall_confidence']
        bullish_scores = []
        bearish_scores = []
        if 'confidence_scores' in analysis_results:
            for key, score in analysis_results['confidence_scores'].items():
                if 'bullish' in key:
                    bullish_scores.append(score)
                elif 'bearish' in key:
                    bearish_scores.append(score)
        if bullish_scores:
            bullish_strength = sum(bullish_scores) / len(bullish_scores)
        else:
            bullish_strength = 0.0
        if bearish_scores:
            bearish_strength = sum(bearish_scores) / len(bearish_scores)
        else:
            bearish_strength = 0.0
        if signal['signal'] == 'bullish':
            signal['strength'] = bullish_strength
        elif signal['signal'] == 'bearish':
            signal['strength'] = bearish_strength
        else:
            signal['strength'] = 0.0
        min_confidence = params['entry_filters']['min_signal_confidence']
        min_strength = params['entry_filters']['min_trend_strength']
        if signal['confidence'] < min_confidence or signal['strength'
            ] < min_strength:
            signal['signal'] = 'neutral'
            signal['explanation'] = (
                f"Signal below minimum thresholds (confidence: {signal['confidence']:.2f} < {min_confidence:.2f} or strength: {signal['strength']:.2f} < {min_strength:.2f})"
                )
        else:
            signal['explanation'] = (
                f"Signal meets criteria with confidence {signal['confidence']:.2f} and strength {signal['strength']:.2f}"
                )
        return signal

    def _get_forex_specific_insights(self, symbol: str, analysis_results:
        Dict[str, Any]) ->Dict[str, Any]:
        """Get forex-specific insights from analysis results"""
        insights = {}
        if ('asset_specific' in analysis_results and 'session_activity' in
            analysis_results['asset_specific']):
            insights['session'] = analysis_results['asset_specific'][
                'session_activity']
        if ('components' in analysis_results and 'correlation' in
            analysis_results['components']):
            insights['correlations'] = analysis_results['components'][
                'correlation'].get('correlations', {})
            if self.config['currency_correlations']['check_correlations']:
                threshold = self.config['currency_correlations'][
                    'correlation_threshold']
                insights['high_correlations'] = self._check_high_correlations(
                    symbol, insights['correlations'], threshold)
        return insights

    def _check_high_correlations(self, symbol: str, correlations: Dict[str,
        float], threshold: float) ->Dict[str, Any]:
        """Check for high correlations that might impact risk"""
        result = {'positive_correlated': [], 'negative_correlated': [],
            'has_risk_concentration': False}
        base_currency = symbol[:3]
        quote_currency = symbol[3:6]
        for pair, corr_value in correlations.items():
            if pair == symbol:
                continue
            if abs(corr_value) >= threshold:
                if corr_value > 0:
                    result['positive_correlated'].append({'pair': pair,
                        'correlation': corr_value})
                else:
                    result['negative_correlated'].append({'pair': pair,
                        'correlation': corr_value})
                if base_currency in pair or quote_currency in pair:
                    result['has_risk_concentration'] = True
        return result


class ForexRangeStrategy(BaseAssetStrategy):
    """
    Forex-specific range trading strategy
    
    This strategy is designed for range-bound markets, focusing on 
    support/resistance levels and mean reversion techniques.
    """

    def __init__(self, analysis_service: AnalysisIntegrationService=None,
        config: Dict[str, Any]=None):
        """
        Initialize a forex range strategy
        
        Args:
            analysis_service: Analysis integration service
            config: Strategy configuration
        """
        super().__init__(strategy_type=AssetStrategyType.FOREX_RANGE,
            asset_class=AssetClass.FOREX, analysis_service=analysis_service,
            config=config or {})
        self.default_config = {'timeframes': ['5m', '15m', '1h', '4h'],
            'primary_timeframe': '1h', 'range_detection': {
            'min_range_periods': 20, 'max_trend_strength': 30,
            'bollinger_periods': 20, 'bollinger_std_dev': 2.0, 'rsi_period':
            14}, 'entry_rules': {'rsi_oversold': 30, 'rsi_overbought': 70,
            'min_band_distance': 0.5, 'min_distance_to_level': 0.1,
            'min_level_touches': 2}, 'exit_rules': {'target_opposite_band':
            True, 'max_holding_periods': 10, 'trailing_stop_enabled': True,
            'trailing_stop_activation': 0.5}, 'risk_management': {
            'max_risk_per_trade': 0.005, 'max_correlated_exposure': 0.03}}
        self.config = {**self.default_config, **self.config}
        self.regime_params = {'ranging': {'entry_rules': {
            'min_band_distance': 0.4}, 'exit_rules': {
            'target_opposite_band': True}}, 'trending': {'entry_rules': {
            'min_band_distance': 0.7}, 'exit_rules': {
            'target_opposite_band': False, 'max_holding_periods': 5}},
            'volatile': {'entry_rules': {'min_band_distance': 0.8,
            'min_level_touches': 3}, 'risk_management': {
            'max_risk_per_trade': 0.003}}}

    async def analyze(self, symbol: str, market_data: Dict[str, MarketData]
        ) ->Dict[str, Any]:
        """Implementation for analyze method - similar to ForexTrendStrategy"""
        pass

    @with_resilience('get_strategy_parameters')
    def get_strategy_parameters(self, market_regime: str) ->Dict[str, Any]:
        """Implementation for get_strategy_parameters - similar to ForexTrendStrategy"""
        pass

    def adjust_parameters(self, params: Dict[str, Any], market_context:
        Dict[str, Any]) ->Dict[str, Any]:
        """Implementation for adjust_parameters - similar to ForexTrendStrategy"""
        pass

    @with_resilience('get_position_sizing')
    def get_position_sizing(self, signal_strength: float, confidence: float
        ) ->float:
        """Implementation for get_position_sizing - similar to ForexTrendStrategy"""
        pass


class ForexBreakoutStrategy(BaseAssetStrategy):
    """
    Forex-specific breakout strategy
    
    This strategy focuses on identifying and trading breakouts from consolidation
    periods, key support/resistance levels, and chart patterns.
    """

    def __init__(self, analysis_service: AnalysisIntegrationService=None,
        config: Dict[str, Any]=None):
        """
        Initialize a forex breakout strategy
        
        Args:
            analysis_service: Analysis integration service
            config: Strategy configuration
        """
        super().__init__(strategy_type=AssetStrategyType.FOREX_BREAKOUT,
            asset_class=AssetClass.FOREX, analysis_service=analysis_service,
            config=config or {})
        self.default_config = {'timeframes': ['15m', '1h', '4h', '1d'],
            'primary_timeframe': '1h', 'breakout_detection': {
            'consolidation_periods': 20, 'volatility_contraction': 0.5,
            'volume_increase_req': 1.5, 'min_level_touches': 2,
            'max_false_breakout': 0.3}, 'entry_rules': {'min_breakout_size':
            0.5, 'confirmation_candles': 1, 'use_momentum': True},
            'exit_rules': {'target_projection': 2.0,
            'trailing_stop_atr_mult': 1.5, 'max_holding_periods': 15},
            'risk_management': {'max_risk_per_trade': 0.01,
            'reduce_overnight_exposure': True}, 'pattern_preferences': {
            'favor_chart_patterns': True, 'min_pattern_quality': 0.7}}
        self.config = {**self.default_config, **self.config}
        self.regime_params = {'trending': {'entry_rules': {
            'min_breakout_size': 0.4, 'confirmation_candles': 1},
            'exit_rules': {'target_projection': 2.5}}, 'ranging': {
            'entry_rules': {'min_breakout_size': 0.7,
            'confirmation_candles': 2}, 'exit_rules': {'target_projection':
            1.5}}, 'volatile': {'entry_rules': {'min_breakout_size': 0.9,
            'confirmation_candles': 2}, 'risk_management': {
            'max_risk_per_trade': 0.007}}}

    async def analyze(self, symbol: str, market_data: Dict[str, MarketData]
        ) ->Dict[str, Any]:
        """Implementation for analyze method - similar to ForexTrendStrategy"""
        pass

    @with_resilience('get_strategy_parameters')
    def get_strategy_parameters(self, market_regime: str) ->Dict[str, Any]:
        """Implementation for get_strategy_parameters - similar to ForexTrendStrategy"""
        pass

    def adjust_parameters(self, params: Dict[str, Any], market_context:
        Dict[str, Any]) ->Dict[str, Any]:
        """Implementation for adjust_parameters - similar to ForexTrendStrategy"""
        pass

    @with_resilience('get_position_sizing')
    def get_position_sizing(self, signal_strength: float, confidence: float
        ) ->float:
        """Implementation for get_position_sizing - similar to ForexTrendStrategy"""
        pass
