"""
Cryptocurrency-Specific Trading Strategies

This module implements trading strategies specifically optimized for 
cryptocurrency markets with their unique characteristics.
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
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

class CryptoMomentumStrategy(BaseAssetStrategy):
    """
    Momentum strategy for cryptocurrency markets
    
    This strategy identifies and trades strong momentum moves in crypto assets,
    accounting for crypto-specific characteristics like 24/7 trading, high volatility,
    and unique market influences like Bitcoin dominance.
    """

    def __init__(self, analysis_service: Optional[
        AnalysisIntegrationService]=None, config: Optional[Dict[str, Any]]=None
        ):
        """Initialize the crypto momentum strategy"""
        super().__init__(strategy_type=AssetStrategyType.CRYPTO_MOMENTUM,
            asset_class=AssetClass.CRYPTO, analysis_service=
            analysis_service, config=config or {})
        self.config.setdefault('timeframes', ['5m', '15m', '1h', '4h', '1d'])
        self.config.setdefault('primary_timeframe', '1h')
        self.config.setdefault('confirmation_timeframes', ['4h', '1d'])
        self.config.setdefault('momentum_threshold', 0.25)
        self.config.setdefault('volume_threshold', 1.5)
        self.config.setdefault('max_position_size', 0.1)
        self.config.setdefault('consider_btc_correlation', True)
        self.config.setdefault('consider_market_cap', True)

    async def analyze(self, symbol: str, market_data: Dict[str, MarketData]
        ) ->Dict[str, Any]:
        """
        Analyze market data and generate trading signals
        
        Args:
            symbol: Crypto symbol
            market_data: Dictionary of market data by timeframe
            
        Returns:
            Dictionary with analysis results and signals
        """
        if not self.validate_asset(symbol):
            return {'valid': False, 'error':
                f'Symbol {symbol} is not a cryptocurrency'}
        required_tfs = self.config['timeframes']
        missing_tfs = [tf for tf in required_tfs if tf not in market_data]
        if missing_tfs:
            return {'valid': False, 'error':
                f'Missing timeframes: {missing_tfs}'}
        market_context = await self.analysis_service.get_market_context(symbol)
        market_regime = market_context.get('regime', 'unknown')
        params = self.get_strategy_parameters(market_regime)
        params = self.adjust_parameters(params, market_context)
        analysis_result = await self.analysis_service.analyze_all(symbol=
            symbol, market_data=market_data, components=self.
            get_required_components(), parameters=params)
        momentum_signals = self._extract_momentum_signals(analysis_result,
            params)
        if self.config['consider_btc_correlation'] and symbol != 'BTCUSD':
            btc_influence = await self._analyze_btc_influence(symbol,
                market_data)
            momentum_signals['btc_influence'] = btc_influence
        signal_strength, confidence = self._calculate_signal_metrics(
            momentum_signals)
        position_size = self.get_position_sizing(signal_strength, confidence)
        return {'valid': True, 'strategy': 'crypto_momentum', 'symbol':
            symbol, 'timestamp': datetime.utcnow().isoformat(),
            'market_regime': market_regime, 'signal': {'direction':
            momentum_signals['direction'], 'strength': signal_strength,
            'confidence': confidence, 'position_size': position_size,
            'entry_price': market_data[self.config['primary_timeframe']].
            close[-1], 'stop_loss': self._calculate_stop_loss(market_data,
            momentum_signals, params), 'take_profit': self.
            _calculate_take_profit(market_data, momentum_signals, params)},
            'analysis': momentum_signals, 'parameters_used': params}

    @with_resilience('get_strategy_parameters')
    def get_strategy_parameters(self, market_regime: str) ->Dict[str, Any]:
        """
        Get strategy parameters based on market regime
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Dictionary with strategy parameters
        """
        base_params = {'rsi_period': 14, 'rsi_overbought': 70,
            'rsi_oversold': 30, 'roc_period': 10, 'macd_fast': 12,
            'macd_slow': 26, 'macd_signal': 9, 'volume_ma_period': 20,
            'stop_atr_multiple': 1.5, 'profit_risk_ratio': 2.0}
        if market_regime == 'trending_strong':
            return {**base_params, 'rsi_overbought': 80, 'rsi_oversold': 40,
                'stop_atr_multiple': 2.0, 'profit_risk_ratio': 2.5}
        elif market_regime == 'trending_weak':
            return {**base_params, 'macd_fast': 8, 'macd_slow': 21,
                'stop_atr_multiple': 1.8, 'profit_risk_ratio': 2.2}
        elif market_regime == 'ranging':
            return {**base_params, 'rsi_overbought': 65, 'rsi_oversold': 35,
                'stop_atr_multiple': 1.2, 'profit_risk_ratio': 1.5}
        elif market_regime == 'volatile':
            return {**base_params, 'rsi_overbought': 75, 'rsi_oversold': 25,
                'stop_atr_multiple': 2.5, 'profit_risk_ratio': 3.0}
        else:
            return base_params

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
        adjusted = params.copy()
        volatility = market_context.get('volatility', {}).get('value', 1.0)
        if volatility > 1.5:
            adjusted['stop_atr_multiple'] *= 1.2
            adjusted['profit_risk_ratio'] *= 1.1
        elif volatility < 0.7:
            adjusted['stop_atr_multiple'] *= 0.9
            adjusted['profit_risk_ratio'] *= 0.9
        if self.config['consider_market_cap']:
            market_cap_tier = market_context.get('fundamentals', {}).get(
                'market_cap_tier', 'mid')
            if market_cap_tier == 'large':
                adjusted['stop_atr_multiple'] *= 0.9
            elif market_cap_tier == 'small':
                adjusted['stop_atr_multiple'] *= 1.3
                adjusted['position_size_factor'] = 0.7
        sentiment = market_context.get('sentiment', {}).get('overall',
            'neutral')
        if sentiment == 'strongly_bullish':
            adjusted['rsi_overbought'] += 5
        elif sentiment == 'strongly_bearish':
            adjusted['rsi_oversold'] -= 5
        return adjusted

    @with_resilience('get_position_sizing')
    def get_position_sizing(self, signal_strength: float, confidence: float
        ) ->float:
        """
        Calculate position sizing based on signal strength and confidence
        
        Args:
            signal_strength: Strength of the trading signal (0.0-1.0)
            confidence: Confidence in the signal (0.0-1.0)
            
        Returns:
            Position size as a percentage of available capital
        """
        base_size = self.config_manager.get('max_position_size', 0.1)
        position_size = base_size * signal_strength * confidence
        position_size_factor = self.config_manager.get('position_size_factor', 1.0)
        position_size *= position_size_factor
        max_size = self.config_manager.get('max_position_size', 0.1)
        min_size = self.config_manager.get('min_position_size', 0.01)
        return max(min(position_size, max_size), min_size)

    @with_resilience('get_required_components')
    def get_required_components(self) ->List[str]:
        """
        Get list of required analysis components for this strategy
        
        Returns:
            List of component names
        """
        return ['technical', 'pattern', 'multi_timeframe', 'ml_prediction',
            'sentiment', 'market_regime', 'crypto_specific']

    async def _analyze_btc_influence(self, symbol: str, market_data: Dict[
        str, MarketData]) ->Dict[str, Any]:
        """
        Analyze Bitcoin's influence on the given altcoin
        
        Args:
            symbol: Crypto symbol
            market_data: Market data dictionary
            
        Returns:
            Dictionary with BTC influence analysis
        """
        return {'btc_correlation': 0.85, 'btc_dominance_impact': 'high',
            'btc_trend': 'bullish'}

    def _extract_momentum_signals(self, analysis_result: Dict[str, Any],
        params: Dict[str, Any]) ->Dict[str, Any]:
        """
        Extract momentum-specific signals from the analysis result
        
        Args:
            analysis_result: Comprehensive analysis result
            params: Strategy parameters
            
        Returns:
            Dictionary with momentum signals
        """
        technical = analysis_result.get('technical', {})
        mtf = analysis_result.get('multi_timeframe', {})
        momentum_direction = 'neutral'
        momentum_strength = 0.0
        rsi_bullish = technical.get('rsi', 50) < params['rsi_oversold']
        rsi_bearish = technical.get('rsi', 50) > params['rsi_overbought']
        macd_bullish = technical.get('macd', {}).get('histogram_direction', ''
            ) == 'increasing'
        macd_bearish = technical.get('macd', {}).get('histogram_direction', ''
            ) == 'decreasing'
        roc_bullish = technical.get('roc', 0) > 0
        roc_bearish = technical.get('roc', 0) < 0
        bullish_signals = sum([rsi_bullish, macd_bullish, roc_bullish])
        bearish_signals = sum([rsi_bearish, macd_bearish, roc_bearish])
        if bullish_signals > bearish_signals:
            momentum_direction = 'bullish'
            momentum_strength = bullish_signals / 3.0
        elif bearish_signals > bullish_signals:
            momentum_direction = 'bearish'
            momentum_strength = bearish_signals / 3.0
        if 'trend_alignment' in mtf:
            if mtf['trend_alignment'
                ] == 'strongly_bullish' and momentum_direction == 'bullish':
                momentum_strength *= 1.2
            elif mtf['trend_alignment'
                ] == 'strongly_bearish' and momentum_direction == 'bearish':
                momentum_strength *= 1.2
            elif mtf['trend_alignment'] == 'mixed':
                momentum_strength *= 0.8
        volume_trend = analysis_result.get('technical', {}).get('volume_trend',
            'normal')
        if volume_trend == 'increasing' and momentum_direction != 'neutral':
            momentum_strength *= 1.1
        elif volume_trend == 'decreasing' and momentum_direction != 'neutral':
            momentum_strength *= 0.9
        return {'direction': momentum_direction, 'strength': min(
            momentum_strength, 1.0), 'rsi': technical.get('rsi', 50),
            'macd': technical.get('macd', {}), 'roc': technical.get('roc', 
            0), 'volume_trend': volume_trend, 'mtf_alignment': mtf.get(
            'trend_alignment', 'unknown')}

    def _calculate_stop_loss(self, market_data: Dict[str, MarketData],
        signals: Dict[str, Any], params: Dict[str, Any]) ->float:
        """Calculate appropriate stop loss level"""
        tf = self.config['primary_timeframe']
        close = market_data[tf].close[-1]
        atr = market_data[tf].indicators.get('atr', 0)
        if atr == 0:
            return close * 0.95 if signals['direction'
                ] == 'bullish' else close * 1.05
        stop_distance = atr * params['stop_atr_multiple']
        if signals['direction'] == 'bullish':
            return close - stop_distance
        else:
            return close + stop_distance

    def _calculate_take_profit(self, market_data: Dict[str, MarketData],
        signals: Dict[str, Any], params: Dict[str, Any]) ->float:
        """Calculate appropriate take profit level"""
        tf = self.config['primary_timeframe']
        close = market_data[tf].close[-1]
        stop_loss = self._calculate_stop_loss(market_data, signals, params)
        risk = abs(close - stop_loss)
        reward = risk * params['profit_risk_ratio']
        if signals['direction'] == 'bullish':
            return close + reward
        elif signals['direction'] == 'bearish':
            return close - reward
        else:
            return close * 1.1

    def _calculate_signal_metrics(self, signals: Dict[str, Any]) ->tuple:
        """Calculate signal strength and confidence from signals"""
        signal_strength = signals.get('strength', 0.0)
        confidence_factors = []
        mtf_alignment = signals.get('mtf_alignment', 'unknown')
        if mtf_alignment == 'strongly_bullish' and signals['direction'
            ] == 'bullish':
            confidence_factors.append(1.0)
        elif mtf_alignment == 'strongly_bearish' and signals['direction'
            ] == 'bearish':
            confidence_factors.append(1.0)
        elif mtf_alignment == 'weakly_bullish' and signals['direction'
            ] == 'bullish':
            confidence_factors.append(0.8)
        elif mtf_alignment == 'weakly_bearish' and signals['direction'
            ] == 'bearish':
            confidence_factors.append(0.8)
        elif mtf_alignment == 'mixed':
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.5)
        volume_trend = signals.get('volume_trend', 'normal')
        if volume_trend == 'increasing' and signals['direction'] != 'neutral':
            confidence_factors.append(0.9)
        elif volume_trend == 'decreasing' and signals['direction'
            ] != 'neutral':
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.8)
        btc_influence = signals.get('btc_influence', {})
        if btc_influence:
            btc_trend = btc_influence.get('btc_trend', 'neutral')
            if btc_trend == 'bullish' and signals['direction'] == 'bullish':
                confidence_factors.append(0.9)
            elif btc_trend == 'bearish' and signals['direction'] == 'bearish':
                confidence_factors.append(0.9)
            elif btc_trend != 'neutral' and btc_trend != signals['direction']:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.8)
        if confidence_factors:
            confidence = sum(confidence_factors) / len(confidence_factors)
        else:
            confidence = 0.7
        return signal_strength, confidence


class CryptoMeanReversionStrategy(BaseAssetStrategy):
    """
    Mean reversion strategy for cryptocurrency markets
    
    This strategy identifies overbought/oversold conditions in crypto assets
    and trades on the expectation of price returning to a mean value.
    """

    def __init__(self, analysis_service: Optional[
        AnalysisIntegrationService]=None, config: Optional[Dict[str, Any]]=None
        ):
        """Initialize the crypto mean reversion strategy"""
        super().__init__(strategy_type=AssetStrategyType.
            CRYPTO_MEAN_REVERSION, asset_class=AssetClass.CRYPTO,
            analysis_service=analysis_service, config=config or {})
        self.config.setdefault('timeframes', ['15m', '1h', '4h', '1d'])
        self.config.setdefault('primary_timeframe', '1h')
        self.config.setdefault('mean_period', 20)
        self.config.setdefault('bollinger_std_devs', 2.5)
        self.config.setdefault('rsi_period', 14)
        self.config.setdefault('rsi_overbought', 75)
        self.config.setdefault('rsi_oversold', 25)
        self.config.setdefault('max_position_size', 0.08)

    async def analyze(self, symbol: str, market_data: Dict[str, MarketData]
        ) ->Dict[str, Any]:
        """
        Analyze market data and generate trading signals
        
        Args:
            symbol: Crypto symbol
            market_data: Dictionary of market data by timeframe
            
        Returns:
            Dictionary with analysis results and signals
        """
        if not self.validate_asset(symbol):
            return {'valid': False, 'error':
                f'Symbol {symbol} is not a cryptocurrency'}
        required_tfs = self.config['timeframes']
        missing_tfs = [tf for tf in required_tfs if tf not in market_data]
        if missing_tfs:
            return {'valid': False, 'error':
                f'Missing timeframes: {missing_tfs}'}
        market_context = await self.analysis_service.get_market_context(symbol)
        market_regime = market_context.get('regime', 'unknown')
        params = self.get_strategy_parameters(market_regime)
        params = self.adjust_parameters(params, market_context)
        analysis_result = await self.analysis_service.analyze_all(symbol=
            symbol, market_data=market_data, components=self.
            get_required_components(), parameters=params)
        reversion_signals = self._extract_reversion_signals(analysis_result,
            params)
        signal_strength, confidence = self._calculate_signal_metrics(
            reversion_signals)
        position_size = self.get_position_sizing(signal_strength, confidence)
        return {'valid': True, 'strategy': 'crypto_mean_reversion',
            'symbol': symbol, 'timestamp': datetime.utcnow().isoformat(),
            'market_regime': market_regime, 'signal': {'direction':
            reversion_signals['direction'], 'strength': signal_strength,
            'confidence': confidence, 'position_size': position_size,
            'entry_price': market_data[self.config['primary_timeframe']].
            close[-1], 'stop_loss': self._calculate_stop_loss(market_data,
            reversion_signals, params), 'take_profit': self.
            _calculate_take_profit(market_data, reversion_signals, params)},
            'analysis': reversion_signals, 'parameters_used': params}

    @with_resilience('get_strategy_parameters')
    def get_strategy_parameters(self, market_regime: str) ->Dict[str, Any]:
        """Get strategy parameters based on market regime"""
        base_params = {'bollinger_period': self.config['mean_period'],
            'bollinger_std_dev': self.config['bollinger_std_devs'],
            'rsi_period': self.config['rsi_period'], 'rsi_overbought': self
            .config['rsi_overbought'], 'rsi_oversold': self.config[
            'rsi_oversold'], 'stop_atr_multiple': 1.2, 'profit_risk_ratio': 1.5
            }
        if market_regime == 'ranging':
            return base_params
        elif market_regime == 'trending_strong':
            return {**base_params, 'bollinger_std_dev': base_params[
                'bollinger_std_dev'] + 0.5, 'rsi_overbought': 80,
                'rsi_oversold': 20, 'profit_risk_ratio': 1.2}
        elif market_regime == 'volatile':
            return {**base_params, 'bollinger_std_dev': base_params[
                'bollinger_std_dev'] + 1.0, 'stop_atr_multiple': 1.5,
                'profit_risk_ratio': 1.8}
        else:
            return base_params

    def adjust_parameters(self, params: Dict[str, Any], market_context:
        Dict[str, Any]) ->Dict[str, Any]:
        """Adjust parameters based on market context"""
        adjusted = params.copy()
        volatility = market_context.get('volatility', {}).get('value', 1.0)
        if volatility > 1.5:
            adjusted['bollinger_std_dev'] *= 1.2
            adjusted['stop_atr_multiple'] *= 1.2
        elif volatility < 0.7:
            adjusted['bollinger_std_dev'] *= 0.9
            adjusted['stop_atr_multiple'] *= 0.9
        volume = market_context.get('volume', {}).get('relative_to_average',
            1.0)
        if volume > 1.5:
            adjusted['profit_risk_ratio'] *= 1.1
        return adjusted

    @with_resilience('get_position_sizing')
    def get_position_sizing(self, signal_strength: float, confidence: float
        ) ->float:
        """Calculate position sizing based on signal strength and confidence"""
        base_size = self.config_manager.get('max_position_size', 0.08)
        position_size = base_size * signal_strength * confidence * 0.9
        max_size = self.config_manager.get('max_position_size', 0.08)
        min_size = self.config_manager.get('min_position_size', 0.01)
        return max(min(position_size, max_size), min_size)

    @with_resilience('get_required_components')
    def get_required_components(self) ->List[str]:
        """Get required analysis components"""
        return ['technical', 'pattern', 'multi_timeframe', 'sentiment',
            'market_regime', 'crypto_specific']

    def _extract_reversion_signals(self, analysis_result: Dict[str, Any],
        params: Dict[str, Any]) ->Dict[str, Any]:
        """Extract mean reversion signals from analysis results"""
        technical = analysis_result.get('technical', {})
        bb = technical.get('bollinger_bands', {})
        bb_width = bb.get('width', 1.0)
        bb_position = bb.get('percent_b', 0.5)
        rsi = technical.get('rsi', 50)
        direction = 'neutral'
        if bb_position <= 0.05 and rsi < params['rsi_oversold']:
            direction = 'bullish'
        elif bb_position >= 0.95 and rsi > params['rsi_overbought']:
            direction = 'bearish'
        strength = 0.0
        if direction == 'bullish':
            rsi_factor = max(0, (params['rsi_oversold'] - rsi) / 10)
            bb_factor = max(0, 0.05 - bb_position) * 10
            strength = min((rsi_factor + bb_factor) / 3.5, 1.0)
        elif direction == 'bearish':
            rsi_factor = max(0, (rsi - params['rsi_overbought']) / 10)
            bb_factor = max(0, bb_position - 0.95) * 10
            strength = min((rsi_factor + bb_factor) / 3.5, 1.0)
        return {'direction': direction, 'strength': strength, 'rsi': rsi,
            'bollinger_bands': bb, 'bb_width': bb_width, 'bb_position':
            bb_position}

    def _calculate_signal_metrics(self, signals: Dict[str, Any]) ->tuple:
        """Calculate signal strength and confidence"""
        strength = signals.get('strength', 0.0)
        bb_width = signals.get('bb_width', 1.0)
        normalized_width = min(bb_width / 2.0, 1.0)
        confidence = normalized_width * 0.7 + 0.3
        return strength, confidence

    def _calculate_stop_loss(self, market_data: Dict[str, MarketData],
        signals: Dict[str, Any], params: Dict[str, Any]) ->float:
        """Calculate appropriate stop loss level"""
        tf = self.config['primary_timeframe']
        close = market_data[tf].close[-1]
        if signals['direction'] == 'bullish':
            recent_low = min(market_data[tf].low[-5:])
            return max(recent_low * 0.99, close * 0.97)
        else:
            recent_high = max(market_data[tf].high[-5:])
            return min(recent_high * 1.01, close * 1.03)

    def _calculate_take_profit(self, market_data: Dict[str, MarketData],
        signals: Dict[str, Any], params: Dict[str, Any]) ->float:
        """Calculate take profit level"""
        tf = self.config['primary_timeframe']
        close = market_data[tf].close[-1]
        bb = signals.get('bollinger_bands', {})
        bb_mid = bb.get('middle', close)
        if signals['direction'] == 'bullish':
            return bb_mid
        else:
            return bb_mid


class CryptoVolatilityStrategy(BaseAssetStrategy):
    """
    Volatility-based strategy for cryptocurrency markets
    
    This strategy capitalizes on the high volatility of crypto markets,
    using options-like strategies to profit from large price movements
    regardless of direction.
    """

    def __init__(self, analysis_service: Optional[
        AnalysisIntegrationService]=None, config: Optional[Dict[str, Any]]=None
        ):
        """Initialize the crypto volatility strategy"""
        super().__init__(strategy_type=AssetStrategyType.CRYPTO_VOLATILITY,
            asset_class=AssetClass.CRYPTO, analysis_service=
            analysis_service, config=config or {})
        self.config.setdefault('timeframes', ['5m', '15m', '1h', '4h'])
        self.config.setdefault('primary_timeframe', '1h')
        self.config.setdefault('atr_period', 14)
        self.config.setdefault('atr_threshold', 1.5)
        self.config.setdefault('max_position_size', 0.05)
        self.config.setdefault('detection_period', 48)

    async def analyze(self, symbol: str, market_data: Dict[str, MarketData]
        ) ->Dict[str, Any]:
        """Analyze market data and generate trading signals"""
        if not self.validate_asset(symbol):
            return {'valid': False, 'error':
                f'Symbol {symbol} is not a cryptocurrency'}
        required_tfs = self.config['timeframes']
        missing_tfs = [tf for tf in required_tfs if tf not in market_data]
        if missing_tfs:
            return {'valid': False, 'error':
                f'Missing timeframes: {missing_tfs}'}
        market_context = await self.analysis_service.get_market_context(symbol)
        market_regime = market_context.get('regime', 'unknown')
        params = self.get_strategy_parameters(market_regime)
        params = self.adjust_parameters(params, market_context)
        analysis_result = await self.analysis_service.analyze_all(symbol=
            symbol, market_data=market_data, components=self.
            get_required_components(), parameters=params)
        volatility_signals = self._extract_volatility_signals(analysis_result,
            params)
        signal_strength, confidence = self._calculate_signal_metrics(
            volatility_signals)
        position_size = self.get_position_sizing(signal_strength, confidence)
        return {'valid': True, 'strategy': 'crypto_volatility', 'symbol':
            symbol, 'timestamp': datetime.utcnow().isoformat(),
            'market_regime': market_regime, 'signal': {'direction':
            volatility_signals['direction'], 'strength': signal_strength,
            'confidence': confidence, 'position_size': position_size,
            'entry_price': market_data[self.config['primary_timeframe']].
            close[-1], 'stop_loss': self._calculate_stop_loss(market_data,
            volatility_signals, params), 'take_profit': self.
            _calculate_take_profit(market_data, volatility_signals, params)
            }, 'analysis': volatility_signals, 'parameters_used': params}

    @with_resilience('get_strategy_parameters')
    def get_strategy_parameters(self, market_regime: str) ->Dict[str, Any]:
        """Get strategy parameters based on market regime"""
        base_params = {'atr_period': self.config['atr_period'],
            'atr_threshold': self.config['atr_threshold'],
            'volatility_lookback': 14, 'bollinger_period': 20,
            'bollinger_std_dev': 2.0, 'stop_atr_multiple': 2.0,
            'profit_risk_ratio': 1.5}
        if market_regime == 'volatile':
            return {**base_params, 'profit_risk_ratio': 2.0}
        elif market_regime == 'trending_strong':
            return {**base_params, 'atr_threshold': base_params[
                'atr_threshold'] * 1.2}
        elif market_regime == 'ranging':
            return {**base_params, 'atr_threshold': base_params[
                'atr_threshold'] * 1.5}
        else:
            return base_params

    def adjust_parameters(self, params: Dict[str, Any], market_context:
        Dict[str, Any]) ->Dict[str, Any]:
        """Adjust parameters based on market context"""
        adjusted = params.copy()
        volatility = market_context.get('volatility', {}).get('value', 1.0)
        if volatility > 2.0:
            adjusted['stop_atr_multiple'] *= 1.3
            adjusted['profit_risk_ratio'] *= 1.2
        volume = market_context.get('volume', {}).get('relative_to_average',
            1.0)
        if volume > 2.0:
            adjusted['atr_threshold'] *= 0.9
        has_major_news = market_context.get('news', {}).get('has_major_events',
            False)
        if has_major_news:
            adjusted['atr_threshold'] *= 0.9
        return adjusted

    @with_resilience('get_position_sizing')
    def get_position_sizing(self, signal_strength: float, confidence: float
        ) ->float:
        """Calculate position sizing based on signal strength and confidence"""
        base_size = self.config_manager.get('max_position_size', 0.05)
        position_size = base_size * signal_strength * confidence * 0.7
        max_size = self.config_manager.get('max_position_size', 0.05)
        min_size = self.config_manager.get('min_position_size', 0.01)
        return max(min(position_size, max_size), min_size)

    @with_resilience('get_required_components')
    def get_required_components(self) ->List[str]:
        """Get required analysis components"""
        return ['technical', 'pattern', 'volatility', 'sentiment',
            'market_regime', 'news_events']

    def _extract_volatility_signals(self, analysis_result: Dict[str, Any],
        params: Dict[str, Any]) ->Dict[str, Any]:
        """Extract volatility-focused signals from analysis results"""
        technical = analysis_result.get('technical', {})
        volatility = analysis_result.get('volatility', {})
        atr = volatility.get('atr', 0)
        atr_percent = volatility.get('atr_percent', 0)
        historical_volatility = volatility.get('historical_volatility', 0)
        bb = technical.get('bollinger_bands', {})
        bb_width = bb.get('width', 1.0)
        volatility_increasing = volatility.get('trend', '') == 'increasing'
        volatility_decreasing = volatility.get('trend', '') == 'decreasing'
        volatility_events = volatility.get('events', [])
        recent_events = [e for e in volatility_events if e.get('hours_ago',
            999) <= self.config['detection_period']]
        direction = 'neutral'
        if volatility_increasing and atr_percent > params['atr_threshold']:
            rsi = technical.get('rsi', 50)
            macd_hist = technical.get('macd', {}).get('histogram', 0)
            if rsi > 60 and macd_hist > 0:
                direction = 'bullish'
            elif rsi < 40 and macd_hist < 0:
                direction = 'bearish'
        strength_factors = []
        if atr_percent > 0:
            strength_factors.append(min(atr_percent / params[
                'atr_threshold'], 2.0) / 2.0)
        if bb_width > 1.3:
            strength_factors.append(min(bb_width / 1.3, 2.0) / 2.0)
        if recent_events:
            event_count = len(recent_events)
            strength_factors.append(min(event_count / 3.0, 1.0))
        strength = sum(strength_factors) / max(len(strength_factors), 1)
        return {'direction': direction, 'strength': min(strength, 1.0),
            'atr': atr, 'atr_percent': atr_percent, 'historical_volatility':
            historical_volatility, 'bb_width': bb_width, 'volatility_trend':
            volatility.get('trend', 'stable'), 'recent_volatility_events':
            recent_events}

    def _calculate_signal_metrics(self, signals: Dict[str, Any]) ->tuple:
        """Calculate signal strength and confidence"""
        strength = signals.get('strength', 0.0)
        if signals.get('volatility_trend') == 'strongly_increasing':
            base_confidence = 0.9
        elif signals.get('volatility_trend') == 'increasing':
            base_confidence = 0.8
        elif signals.get('volatility_trend') == 'decreasing':
            base_confidence = 0.6
        else:
            base_confidence = 0.7
        recent_events = signals.get('recent_volatility_events', [])
        if recent_events:
            event_factor = min(len(recent_events) / 5.0, 1.0) * 0.2
            confidence = base_confidence + event_factor
        else:
            confidence = base_confidence
        return strength, min(confidence, 1.0)

    def _calculate_stop_loss(self, market_data: Dict[str, MarketData],
        signals: Dict[str, Any], params: Dict[str, Any]) ->float:
        """Calculate stop loss level for volatility strategy"""
        tf = self.config['primary_timeframe']
        close = market_data[tf].close[-1]
        atr = signals.get('atr', close * 0.02)
        stop_distance = atr * params['stop_atr_multiple']
        if signals['direction'] == 'bullish':
            return close - stop_distance
        elif signals['direction'] == 'bearish':
            return close + stop_distance
        elif len(market_data[tf].close) >= 20:
            recent_high = max(market_data[tf].high[-20:])
            recent_low = min(market_data[tf].low[-20:])
            return recent_low - stop_distance * 0.5
        else:
            return close * 0.92

    def _calculate_take_profit(self, market_data: Dict[str, MarketData],
        signals: Dict[str, Any], params: Dict[str, Any]) ->float:
        """Calculate take profit level for volatility strategy"""
        tf = self.config['primary_timeframe']
        close = market_data[tf].close[-1]
        stop_loss = self._calculate_stop_loss(market_data, signals, params)
        risk = abs(close - stop_loss)
        reward = risk * params['profit_risk_ratio']
        if signals['direction'] == 'bullish':
            return close + reward
        elif signals['direction'] == 'bearish':
            return close - reward
        else:
            return close * 1.1
