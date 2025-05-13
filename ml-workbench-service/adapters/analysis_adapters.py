"""
Analysis Engine Adapters Module

This module provides adapter implementations for analysis engine interfaces,
helping to break circular dependencies between services.
"""
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import logging
import asyncio
import json
from common_lib.analysis.interfaces import IAnalysisEngine, IMarketRegimeAnalyzer, IMultiAssetAnalyzer, MarketRegimeType, AnalysisTimeframe
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)


from ml_workbench_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class AnalysisEngineAdapter(IAnalysisEngine):
    """
    Adapter for analysis engine that implements the common interface.
    
    This adapter can either wrap an actual analysis engine instance or provide
    standalone functionality to avoid circular dependencies.
    """

    def __init__(self, analysis_engine_instance=None):
        """
        Initialize the adapter.
        
        Args:
            analysis_engine_instance: Optional actual analysis engine instance to wrap
        """
        self.analysis_engine = analysis_engine_instance
        self.analysis_cache = {}

    @async_with_exception_handling
    async def get_technical_analysis(self, symbol: str, timeframe: Union[
        str, AnalysisTimeframe], indicators: List[Dict[str, Any]],
        lookback_periods: int=100) ->Dict[str, Any]:
        """
        Get technical analysis for a symbol.
        
        Args:
            symbol: The trading symbol
            timeframe: The timeframe to analyze
            indicators: List of indicators to calculate
            lookback_periods: Number of periods to analyze
            
        Returns:
            Dictionary with technical analysis results
        """
        if self.analysis_engine:
            try:
                return await self.analysis_engine.get_technical_analysis(symbol
                    =symbol, timeframe=timeframe, indicators=indicators,
                    lookback_periods=lookback_periods)
            except Exception as e:
                logger.warning(f'Error getting technical analysis: {str(e)}')
        logger.info(
            f'Using fallback technical analysis for {symbol} {timeframe}')
        result = {'symbol': symbol, 'timeframe': str(timeframe),
            'timestamp': datetime.now().isoformat(), 'indicators': {},
            'is_fallback': True}
        for indicator in indicators:
            indicator_name = indicator.get('name', 'unknown')
            indicator_params = indicator.get('params', {})
            if indicator_name.upper() in ['SMA', 'EMA', 'MA']:
                period = indicator_params.get('period', 20)
                result['indicators'][indicator_name] = {'values': [(100.0 +
                    i * 0.01) for i in range(10)], 'parameters':
                    indicator_params, 'current_value': 100.5}
            elif indicator_name.upper() in ['RSI']:
                period = indicator_params.get('period', 14)
                result['indicators'][indicator_name] = {'values': [(50.0 + 
                    i * 0.5) for i in range(10)], 'parameters':
                    indicator_params, 'current_value': 55.0}
            elif indicator_name.upper() in ['MACD']:
                result['indicators'][indicator_name] = {'macd_line': [(0.1 +
                    i * 0.01) for i in range(10)], 'signal_line': [(0.05 + 
                    i * 0.01) for i in range(10)], 'histogram': [(0.05 + i *
                    0.005) for i in range(10)], 'parameters':
                    indicator_params, 'current_values': {'macd': 0.2,
                    'signal': 0.15, 'histogram': 0.05}}
            else:
                result['indicators'][indicator_name] = {'values': [(i * 0.1
                    ) for i in range(10)], 'parameters': indicator_params,
                    'current_value': 0.5}
        cache_key = (
            f'{symbol}_{timeframe}_{len(indicators)}_{lookback_periods}')
        self.analysis_cache[cache_key] = result
        return result

    @async_with_exception_handling
    async def get_confluence_analysis(self, symbol: str, timeframe: Union[
        str, AnalysisTimeframe], lookback_periods: int=100) ->Dict[str, Any]:
        """
        Get confluence analysis for a symbol.
        
        Args:
            symbol: The trading symbol
            timeframe: The timeframe to analyze
            lookback_periods: Number of periods to analyze
            
        Returns:
            Dictionary with confluence analysis results
        """
        if self.analysis_engine:
            try:
                return await self.analysis_engine.get_confluence_analysis(
                    symbol=symbol, timeframe=timeframe, lookback_periods=
                    lookback_periods)
            except Exception as e:
                logger.warning(f'Error getting confluence analysis: {str(e)}')
        logger.info(
            f'Using fallback confluence analysis for {symbol} {timeframe}')
        return {'symbol': symbol, 'timeframe': str(timeframe), 'timestamp':
            datetime.now().isoformat(), 'confluence_score': 0.65,
            'direction': 'bullish', 'supporting_signals': [{'name':
            'trend_following', 'score': 0.7, 'direction': 'bullish'}, {
            'name': 'momentum', 'score': 0.6, 'direction': 'bullish'}],
            'opposing_signals': [{'name': 'overbought', 'score': 0.3,
            'direction': 'bearish'}], 'is_fallback': True}

    @async_with_exception_handling
    async def get_multi_timeframe_analysis(self, symbol: str, timeframes:
        List[Union[str, AnalysisTimeframe]], indicators: List[Dict[str, Any]]
        ) ->Dict[str, Any]:
        """
        Get multi-timeframe analysis for a symbol.
        
        Args:
            symbol: The trading symbol
            timeframes: List of timeframes to analyze
            indicators: List of indicators to calculate
            
        Returns:
            Dictionary with multi-timeframe analysis results
        """
        if self.analysis_engine:
            try:
                return await self.analysis_engine.get_multi_timeframe_analysis(
                    symbol=symbol, timeframes=timeframes, indicators=indicators
                    )
            except Exception as e:
                logger.warning(
                    f'Error getting multi-timeframe analysis: {str(e)}')
        logger.info(f'Using fallback multi-timeframe analysis for {symbol}')
        result = {'symbol': symbol, 'timestamp': datetime.now().isoformat(),
            'timeframes': {}, 'summary': {'trend_alignment': 0.7,
            'overall_direction': 'bullish', 'confidence': 0.65},
            'is_fallback': True}
        for tf in timeframes:
            tf_str = str(tf)
            result['timeframes'][tf_str] = {'indicators': {}, 'trend': 
                'bullish' if tf_str in ['1h', '4h', '1d'] else 'neutral',
                'strength': 0.6 + (0.1 if tf_str in ['1h', '4h'] else 0)}
            for indicator in indicators:
                indicator_name = indicator.get('name', 'unknown')
                result['timeframes'][tf_str]['indicators'][indicator_name] = {
                    'current_value': 100.0 + (5.0 if tf_str in ['1h', '4h']
                     else 0), 'signal': 'buy' if tf_str in ['1h', '4h',
                    '1d'] else 'neutral'}
        return result

    @async_with_exception_handling
    async def get_integrated_analysis(self, symbol: str, timeframe: Union[
        str, AnalysisTimeframe], include_components: Optional[List[str]]=None
        ) ->Dict[str, Any]:
        """
        Get integrated analysis from multiple components.
        
        Args:
            symbol: The trading symbol
            timeframe: The timeframe to analyze
            include_components: Optional list of components to include
            
        Returns:
            Dictionary with integrated analysis results
        """
        if self.analysis_engine:
            try:
                return await self.analysis_engine.get_integrated_analysis(
                    symbol=symbol, timeframe=timeframe, include_components=
                    include_components)
            except Exception as e:
                logger.warning(f'Error getting integrated analysis: {str(e)}')
        logger.info(
            f'Using fallback integrated analysis for {symbol} {timeframe}')
        if not include_components:
            include_components = ['technical', 'pattern', 'regime',
                'sentiment', 'ml']
        result = {'symbol': symbol, 'timeframe': str(timeframe),
            'timestamp': datetime.now().isoformat(), 'components': {},
            'integrated_signal': {'direction': 'bullish', 'strength': 0.65,
            'confidence': 0.7}, 'is_fallback': True}
        if 'technical' in include_components:
            result['components']['technical'] = {'trend': 'bullish',
                'momentum': 'increasing', 'volatility': 'moderate',
                'signals': [{'name': 'moving_averages', 'signal': 'buy'}, {
                'name': 'oscillators', 'signal': 'neutral'}]}
        if 'pattern' in include_components:
            result['components']['pattern'] = {'detected_patterns': [{
                'name': 'bullish_engulfing', 'confidence': 0.8, 'signal':
                'buy'}], 'support_resistance': {'support_levels': [1.105, 
                1.1], 'resistance_levels': [1.115, 1.12]}}
        if 'regime' in include_components:
            result['components']['regime'] = {'current_regime':
                'trending_bullish', 'regime_confidence': 0.75,
                'volatility_level': 'moderate'}
        if 'sentiment' in include_components:
            result['components']['sentiment'] = {'overall_sentiment':
                'positive', 'sentiment_score': 0.6, 'news_impact': 'moderate'}
        if 'ml' in include_components:
            result['components']['ml'] = {'prediction': 'bullish',
                'probability': 0.7, 'horizon': '24h', 'model_confidence': 0.65}
        return result


class MarketRegimeAnalyzerAdapter(IMarketRegimeAnalyzer):
    """
    Adapter for market regime analyzer that implements the common interface.
    
    This adapter can either wrap an actual analyzer instance or provide
    standalone functionality to avoid circular dependencies.
    """

    def __init__(self, analyzer_instance=None):
        """
        Initialize the adapter.
        
        Args:
            analyzer_instance: Optional actual analyzer instance to wrap
        """
        self.analyzer = analyzer_instance
        self.regime_history = {}

    @async_with_exception_handling
    async def detect_regime(self, symbol: str, timeframe: Union[str,
        AnalysisTimeframe], lookback_periods: int=100) ->Dict[str, Any]:
        """
        Detect the current market regime for a symbol.
        
        Args:
            symbol: The trading symbol
            timeframe: The timeframe to analyze
            lookback_periods: Number of periods to analyze
            
        Returns:
            Dictionary with regime information
        """
        if self.analyzer:
            try:
                return await self.analyzer.detect_regime(symbol=symbol,
                    timeframe=timeframe, lookback_periods=lookback_periods)
            except Exception as e:
                logger.warning(f'Error detecting market regime: {str(e)}')
        logger.info(
            f'Using fallback market regime detection for {symbol} {timeframe}')
        regime_type = MarketRegimeType.RANGING_NARROW
        if symbol.startswith('EUR'):
            regime_type = MarketRegimeType.TRENDING_BULLISH
        elif symbol.startswith('USD'):
            regime_type = MarketRegimeType.TRENDING_BEARISH
        if symbol not in self.regime_history:
            self.regime_history[symbol] = []
        self.regime_history[symbol].append({'timestamp': datetime.now().
            isoformat(), 'regime': regime_type, 'timeframe': str(timeframe)})
        if len(self.regime_history[symbol]) > 10:
            self.regime_history[symbol] = self.regime_history[symbol][-10:]
        return {'regime_type': regime_type, 'confidence': 0.75,
            'regime_metrics': {'trend_strength': 0.6, 'volatility': 0.4,
            'momentum': 0.5}, 'regime_history': self.regime_history[symbol]}

    @async_with_exception_handling
    async def get_regime_probabilities(self, symbol: str, timeframe: Union[
        str, AnalysisTimeframe]) ->Dict[MarketRegimeType, float]:
        """
        Get probability distribution across different regime types.
        
        Args:
            symbol: The trading symbol
            timeframe: The timeframe to analyze
            
        Returns:
            Dictionary mapping regime types to their probabilities
        """
        if self.analyzer:
            try:
                return await self.analyzer.get_regime_probabilities(symbol=
                    symbol, timeframe=timeframe)
            except Exception as e:
                logger.warning(f'Error getting regime probabilities: {str(e)}')
        logger.info(
            f'Using fallback regime probabilities for {symbol} {timeframe}')
        base_probs = {MarketRegimeType.TRENDING_BULLISH: 0.1,
            MarketRegimeType.TRENDING_BEARISH: 0.1, MarketRegimeType.
            RANGING_NARROW: 0.3, MarketRegimeType.RANGING_WIDE: 0.2,
            MarketRegimeType.VOLATILE: 0.1, MarketRegimeType.CHOPPY: 0.1,
            MarketRegimeType.BREAKOUT: 0.05, MarketRegimeType.REVERSAL: 0.05}
        if symbol.startswith('EUR'):
            base_probs[MarketRegimeType.TRENDING_BULLISH] = 0.3
            base_probs[MarketRegimeType.RANGING_NARROW] = 0.2
        elif symbol.startswith('USD'):
            base_probs[MarketRegimeType.TRENDING_BEARISH] = 0.3
            base_probs[MarketRegimeType.RANGING_NARROW] = 0.2
        return base_probs

    @async_with_exception_handling
    async def get_regime_transition_probability(self, symbol: str,
        from_regime: MarketRegimeType, to_regime: MarketRegimeType,
        timeframe: Union[str, AnalysisTimeframe]) ->float:
        """
        Get the probability of transitioning between regimes.
        
        Args:
            symbol: The trading symbol
            from_regime: Starting regime type
            to_regime: Target regime type
            timeframe: The timeframe to analyze
            
        Returns:
            Probability of transition (0.0 to 1.0)
        """
        if self.analyzer:
            try:
                return await self.analyzer.get_regime_transition_probability(
                    symbol=symbol, from_regime=from_regime, to_regime=
                    to_regime, timeframe=timeframe)
            except Exception as e:
                logger.warning(
                    f'Error getting regime transition probability: {str(e)}')
        logger.info(
            f'Using fallback regime transition probability for {symbol} {timeframe}'
            )
        if from_regime == to_regime:
            return 0.7
        if (from_regime == MarketRegimeType.RANGING_NARROW and to_regime ==
            MarketRegimeType.BREAKOUT):
            return 0.3
        if (from_regime == MarketRegimeType.VOLATILE and to_regime ==
            MarketRegimeType.RANGING_WIDE):
            return 0.25
        if (from_regime == MarketRegimeType.TRENDING_BULLISH and to_regime ==
            MarketRegimeType.REVERSAL):
            return 0.15
        if (from_regime == MarketRegimeType.TRENDING_BEARISH and to_regime ==
            MarketRegimeType.REVERSAL):
            return 0.15
        return 0.1


class MultiAssetAnalyzerAdapter(IMultiAssetAnalyzer):
    """
    Adapter for multi-asset analyzer that implements the common interface.
    
    This adapter can either wrap an actual analyzer instance or provide
    standalone functionality to avoid circular dependencies.
    """

    def __init__(self, analyzer_instance=None):
        """
        Initialize the adapter.
        
        Args:
            analyzer_instance: Optional actual analyzer instance to wrap
        """
        self.analyzer = analyzer_instance
        self.correlation_cache = {}

    @async_with_exception_handling
    async def get_correlated_assets(self, symbol: str, min_correlation:
        float=0.7, lookback_periods: int=100) ->Dict[str, float]:
        """
        Get assets correlated with the specified symbol.
        
        Args:
            symbol: The trading symbol
            min_correlation: Minimum correlation coefficient
            lookback_periods: Number of periods to analyze
            
        Returns:
            Dictionary mapping correlated symbols to their correlation coefficients
        """
        if self.analyzer:
            try:
                return await self.analyzer.get_correlated_assets(symbol=
                    symbol, min_correlation=min_correlation,
                    lookback_periods=lookback_periods)
            except Exception as e:
                logger.warning(f'Error getting correlated assets: {str(e)}')
        logger.info(f'Using fallback correlated assets for {symbol}')
        cache_key = f'{symbol}_{min_correlation}_{lookback_periods}'
        if cache_key in self.correlation_cache:
            return self.correlation_cache[cache_key]
        correlations = {}
        if symbol.startswith('EUR'):
            correlations = {'EURGBP': 0.85, 'EURJPY': 0.75, 'EURCHF': 0.82,
                'EURCAD': 0.72, 'GBPUSD': 0.65}
        elif symbol.startswith('USD'):
            correlations = {'USDJPY': 0.78, 'USDCAD': 0.88, 'USDCHF': 0.76,
                'EURUSD': -0.92, 'GBPUSD': -0.85}
        elif symbol.startswith('GBP'):
            correlations = {'GBPUSD': 0.95, 'GBPJPY': 0.82, 'GBPCHF': 0.79,
                'GBPCAD': 0.76, 'EURGBP': -0.88}
        else:
            correlations = {'EURUSD': 0.6, 'GBPUSD': 0.55, 'USDJPY': 0.5,
                'AUDUSD': 0.45, 'USDCAD': 0.4}
        filtered_correlations = {s: c for s, c in correlations.items() if 
            abs(c) >= min_correlation and s != symbol}
        self.correlation_cache[cache_key] = filtered_correlations
        return filtered_correlations

    @async_with_exception_handling
    async def get_currency_strength(self, currencies: Optional[List[str]]=
        None, timeframe: Union[str, AnalysisTimeframe]=AnalysisTimeframe.H1
        ) ->Dict[str, float]:
        """
        Get relative strength of currencies.
        
        Args:
            currencies: Optional list of currencies to analyze
            timeframe: The timeframe to analyze
            
        Returns:
            Dictionary mapping currencies to their strength scores
        """
        if self.analyzer:
            try:
                return await self.analyzer.get_currency_strength(currencies
                    =currencies, timeframe=timeframe)
            except Exception as e:
                logger.warning(f'Error getting currency strength: {str(e)}')
        logger.info(
            f'Using fallback currency strength for timeframe {timeframe}')
        if not currencies:
            currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF',
                'NZD']
        strength = {'USD': 0.75, 'EUR': 0.65, 'GBP': 0.6, 'JPY': 0.55,
            'AUD': 0.7, 'CAD': 0.62, 'CHF': 0.58, 'NZD': 0.63}
        return {c: strength.get(c, 0.5) for c in currencies}

    @async_with_exception_handling
    async def get_cross_pair_opportunities(self, base_currency: str,
        quote_currency: str, related_pairs: Optional[List[str]]=None) ->Dict[
        str, Any]:
        """
        Analyze opportunities across related currency pairs.
        
        Args:
            base_currency: Base currency
            quote_currency: Quote currency
            related_pairs: Optional list of related pairs to analyze
            
        Returns:
            Dictionary with cross-pair analysis results
        """
        if self.analyzer:
            try:
                return await self.analyzer.get_cross_pair_opportunities(
                    base_currency=base_currency, quote_currency=
                    quote_currency, related_pairs=related_pairs)
            except Exception as e:
                logger.warning(
                    f'Error getting cross-pair opportunities: {str(e)}')
        logger.info(
            f'Using fallback cross-pair opportunities for {base_currency}{quote_currency}'
            )
        if not related_pairs:
            related_pairs = [f'{base_currency}USD', f'USD{quote_currency}',
                f'{base_currency}JPY', f'{quote_currency}JPY']
        return {'main_pair': f'{base_currency}{quote_currency}',
            'strength_differential': {'base': 0.65, 'quote': 0.55, 'net': 
            0.1}, 'triangular_arbitrage': {'opportunities': [{'path': [
            f'{base_currency}USD', f'USD{quote_currency}',
            f'{quote_currency}{base_currency}'], 'potential': 0.05,
            'confidence': 0.6}]}, 'divergence_opportunities': [{'pairs': [
            f'{base_currency}{quote_currency}', f'{base_currency}USD'],
            'correlation': 0.85, 'divergence_score': 0.2,
            'reversion_potential': 'medium'}], 'is_fallback': True}
