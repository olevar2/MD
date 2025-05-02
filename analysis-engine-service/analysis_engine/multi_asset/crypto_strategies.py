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


class CryptoMomentumStrategy(BaseAssetStrategy):
    """
    Momentum strategy for cryptocurrency markets
    
    This strategy identifies and trades strong momentum moves in crypto assets,
    accounting for crypto-specific characteristics like 24/7 trading, high volatility,
    and unique market influences like Bitcoin dominance.
    """
    
    def __init__(
        self,
        analysis_service: Optional[AnalysisIntegrationService] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the crypto momentum strategy"""
        super().__init__(
            strategy_type=AssetStrategyType.CRYPTO_MOMENTUM,
            asset_class=AssetClass.CRYPTO,
            analysis_service=analysis_service,
            config=config or {}
        )
        
        # Set default configuration parameters
        self.config.setdefault("timeframes", ["5m", "15m", "1h", "4h", "1d"])
        self.config.setdefault("primary_timeframe", "1h")
        self.config.setdefault("confirmation_timeframes", ["4h", "1d"])
        self.config.setdefault("momentum_threshold", 0.25)  # Higher for crypto
        self.config.setdefault("volume_threshold", 1.5)  # Higher volume requirement
        self.config.setdefault("max_position_size", 0.10)  # Lower max position size
        self.config.setdefault("consider_btc_correlation", True)
        self.config.setdefault("consider_market_cap", True)

    async def analyze(self, symbol: str, market_data: Dict[str, MarketData]) -> Dict[str, Any]:
        """
        Analyze market data and generate trading signals
        
        Args:
            symbol: Crypto symbol
            market_data: Dictionary of market data by timeframe
            
        Returns:
            Dictionary with analysis results and signals
        """
        if not self.validate_asset(symbol):
            return {"valid": False, "error": f"Symbol {symbol} is not a cryptocurrency"}
            
        # Ensure we have the required timeframes
        required_tfs = self.config["timeframes"]
        missing_tfs = [tf for tf in required_tfs if tf not in market_data]
        if missing_tfs:
            return {"valid": False, "error": f"Missing timeframes: {missing_tfs}"}
            
        # Get market regime to adjust parameters
        market_context = await self.analysis_service.get_market_context(symbol)
        market_regime = market_context.get("regime", "unknown")
        
        # Get strategy parameters adjusted for current market regime
        params = self.get_strategy_parameters(market_regime)
        params = self.adjust_parameters(params, market_context)
        
        # Run comprehensive analysis using the analysis integration service
        analysis_result = await self.analysis_service.analyze_all(
            symbol=symbol,
            market_data=market_data,
            components=self.get_required_components(),
            parameters=params
        )
        
        # Extract momentum signals
        momentum_signals = self._extract_momentum_signals(analysis_result, params)
        
        # Consider BTC dominance and correlation if enabled
        if self.config["consider_btc_correlation"] and symbol != "BTCUSD":
            btc_influence = await self._analyze_btc_influence(symbol, market_data)
            momentum_signals["btc_influence"] = btc_influence
        
        # Calculate signal strength and confidence
        signal_strength, confidence = self._calculate_signal_metrics(momentum_signals)
        
        # Calculate position sizing
        position_size = self.get_position_sizing(signal_strength, confidence)
        
        # Compile final result
        return {
            "valid": True,
            "strategy": "crypto_momentum",
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "market_regime": market_regime,
            "signal": {
                "direction": momentum_signals["direction"],
                "strength": signal_strength,
                "confidence": confidence,
                "position_size": position_size,
                "entry_price": market_data[self.config["primary_timeframe"]].close[-1],
                "stop_loss": self._calculate_stop_loss(market_data, momentum_signals, params),
                "take_profit": self._calculate_take_profit(market_data, momentum_signals, params)
            },
            "analysis": momentum_signals,
            "parameters_used": params
        }
    
    def get_strategy_parameters(self, market_regime: str) -> Dict[str, Any]:
        """
        Get strategy parameters based on market regime
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Dictionary with strategy parameters
        """
        # Base parameters for all regimes
        base_params = {
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "roc_period": 10,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "volume_ma_period": 20,
            "stop_atr_multiple": 1.5,  # Wider stops for crypto due to volatility
            "profit_risk_ratio": 2.0
        }
        
        # Adjust parameters based on market regime
        if market_regime == "trending_strong":
            return {
                **base_params,
                "rsi_overbought": 80,  # Allow for higher RSI in strong trends
                "rsi_oversold": 40,
                "stop_atr_multiple": 2.0,  # Wider stops in strong trends
                "profit_risk_ratio": 2.5
            }
        elif market_regime == "trending_weak":
            return {
                **base_params,
                "macd_fast": 8,  # More sensitive MACD
                "macd_slow": 21,
                "stop_atr_multiple": 1.8,
                "profit_risk_ratio": 2.2
            }
        elif market_regime == "ranging":
            return {
                **base_params,
                "rsi_overbought": 65,  # Tighter overbought/oversold
                "rsi_oversold": 35,
                "stop_atr_multiple": 1.2,  # Tighter stops in ranges
                "profit_risk_ratio": 1.5
            }
        elif market_regime == "volatile":
            return {
                **base_params,
                "rsi_overbought": 75,
                "rsi_oversold": 25,
                "stop_atr_multiple": 2.5,  # Much wider stops in volatile markets
                "profit_risk_ratio": 3.0
            }
        else:
            # Default parameters
            return base_params
    
    def adjust_parameters(self, params: Dict[str, Any], market_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust strategy parameters based on market context
        
        Args:
            params: Current strategy parameters
            market_context: Market context information
            
        Returns:
            Adjusted parameters
        """
        adjusted = params.copy()
        
        # Adjust based on volatility
        volatility = market_context.get("volatility", {}).get("value", 1.0)
        if volatility > 1.5:  # High volatility
            adjusted["stop_atr_multiple"] *= 1.2
            adjusted["profit_risk_ratio"] *= 1.1
        elif volatility < 0.7:  # Low volatility
            adjusted["stop_atr_multiple"] *= 0.9
            adjusted["profit_risk_ratio"] *= 0.9
        
        # Adjust based on market cap (if available and enabled)
        if self.config["consider_market_cap"]:
            market_cap_tier = market_context.get("fundamentals", {}).get("market_cap_tier", "mid")
            if market_cap_tier == "large":  # BTC, ETH, etc.
                # Larger caps tend to be less volatile
                adjusted["stop_atr_multiple"] *= 0.9
            elif market_cap_tier == "small":  # Small cap alts
                # Smaller caps need wider stops
                adjusted["stop_atr_multiple"] *= 1.3
                # And should have smaller position sizing
                adjusted["position_size_factor"] = 0.7
        
        # Adjust based on market sentiment
        sentiment = market_context.get("sentiment", {}).get("overall", "neutral")
        if sentiment == "strongly_bullish":
            adjusted["rsi_overbought"] += 5  # Allow higher RSI values in bullish sentiment
        elif sentiment == "strongly_bearish":
            adjusted["rsi_oversold"] -= 5  # Allow lower RSI values in bearish sentiment
            
        return adjusted
    
    def get_position_sizing(self, signal_strength: float, confidence: float) -> float:
        """
        Calculate position sizing based on signal strength and confidence
        
        Args:
            signal_strength: Strength of the trading signal (0.0-1.0)
            confidence: Confidence in the signal (0.0-1.0)
            
        Returns:
            Position size as a percentage of available capital
        """
        base_size = self.config.get("max_position_size", 0.10)
        
        # Calculate size based on signal strength and confidence
        position_size = base_size * signal_strength * confidence
        
        # Apply market cap factor if enabled
        position_size_factor = self.config.get("position_size_factor", 1.0)
        position_size *= position_size_factor
        
        # Ensure position size is within limits
        max_size = self.config.get("max_position_size", 0.10)
        min_size = self.config.get("min_position_size", 0.01)
        
        return max(min(position_size, max_size), min_size)
    
    def get_required_components(self) -> List[str]:
        """
        Get list of required analysis components for this strategy
        
        Returns:
            List of component names
        """
        return [
            "technical", "pattern", "multi_timeframe", 
            "ml_prediction", "sentiment", "market_regime",
            "crypto_specific"  # Crypto-specific components like BTC dominance
        ]
        
    async def _analyze_btc_influence(self, symbol: str, market_data: Dict[str, MarketData]) -> Dict[str, Any]:
        """
        Analyze Bitcoin's influence on the given altcoin
        
        Args:
            symbol: Crypto symbol
            market_data: Market data dictionary
            
        Returns:
            Dictionary with BTC influence analysis
        """
        # This would fetch BTC correlation data from the analysis service
        # For now we'll return a placeholder
        return {
            "btc_correlation": 0.85,  # High correlation with BTC
            "btc_dominance_impact": "high",
            "btc_trend": "bullish"
        }
    
    def _extract_momentum_signals(self, analysis_result: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract momentum-specific signals from the analysis result
        
        Args:
            analysis_result: Comprehensive analysis result
            params: Strategy parameters
            
        Returns:
            Dictionary with momentum signals
        """
        # Extract technical indicators
        technical = analysis_result.get("technical", {})
        
        # Extract multi-timeframe analysis
        mtf = analysis_result.get("multi_timeframe", {})
        
        # Determine momentum direction
        momentum_direction = "neutral"
        momentum_strength = 0.0
        
        # Check RSI
        rsi_bullish = technical.get("rsi", 50) < params["rsi_oversold"]
        rsi_bearish = technical.get("rsi", 50) > params["rsi_overbought"]
        
        # Check MACD
        macd_bullish = technical.get("macd", {}).get("histogram_direction", "") == "increasing"
        macd_bearish = technical.get("macd", {}).get("histogram_direction", "") == "decreasing"
        
        # Check rate of change
        roc_bullish = technical.get("roc", 0) > 0
        roc_bearish = technical.get("roc", 0) < 0
        
        # Determine primary momentum direction
        bullish_signals = sum([rsi_bullish, macd_bullish, roc_bullish])
        bearish_signals = sum([rsi_bearish, macd_bearish, roc_bearish])
        
        if bullish_signals > bearish_signals:
            momentum_direction = "bullish"
            momentum_strength = bullish_signals / 3.0
        elif bearish_signals > bullish_signals:
            momentum_direction = "bearish"
            momentum_strength = bearish_signals / 3.0
            
        # Check multi-timeframe confirmation
        if "trend_alignment" in mtf:
            if mtf["trend_alignment"] == "strongly_bullish" and momentum_direction == "bullish":
                momentum_strength *= 1.2
            elif mtf["trend_alignment"] == "strongly_bearish" and momentum_direction == "bearish":
                momentum_strength *= 1.2
            elif mtf["trend_alignment"] == "mixed":
                momentum_strength *= 0.8
        
        # Check volume confirmation
        volume_trend = analysis_result.get("technical", {}).get("volume_trend", "normal")
        if volume_trend == "increasing" and momentum_direction != "neutral":
            momentum_strength *= 1.1
        elif volume_trend == "decreasing" and momentum_direction != "neutral":
            momentum_strength *= 0.9
        
        return {
            "direction": momentum_direction,
            "strength": min(momentum_strength, 1.0),
            "rsi": technical.get("rsi", 50),
            "macd": technical.get("macd", {}),
            "roc": technical.get("roc", 0),
            "volume_trend": volume_trend,
            "mtf_alignment": mtf.get("trend_alignment", "unknown")
        }
        
    def _calculate_stop_loss(self, market_data: Dict[str, MarketData], signals: Dict[str, Any], params: Dict[str, Any]) -> float:
        """Calculate appropriate stop loss level"""
        tf = self.config["primary_timeframe"]
        close = market_data[tf].close[-1]
        
        # Get ATR for volatility-based stop loss
        atr = market_data[tf].indicators.get("atr", 0)
        if atr == 0:
            # Default to percentage-based stop if ATR not available
            return close * 0.95 if signals["direction"] == "bullish" else close * 1.05
        
        # Calculate stop based on ATR and direction
        stop_distance = atr * params["stop_atr_multiple"]
        
        if signals["direction"] == "bullish":
            return close - stop_distance
        else:
            return close + stop_distance
    
    def _calculate_take_profit(self, market_data: Dict[str, MarketData], signals: Dict[str, Any], params: Dict[str, Any]) -> float:
        """Calculate appropriate take profit level"""
        tf = self.config["primary_timeframe"]
        close = market_data[tf].close[-1]
        stop_loss = self._calculate_stop_loss(market_data, signals, params)
        
        # Calculate risk in price terms
        risk = abs(close - stop_loss)
        
        # Calculate reward based on risk-reward ratio
        reward = risk * params["profit_risk_ratio"]
        
        if signals["direction"] == "bullish":
            return close + reward
        elif signals["direction"] == "bearish":
            return close - reward
        else:
            # For neutral strategies, could use a percentage target
            return close * 1.1  # 10% move
    
    def _calculate_signal_metrics(self, signals: Dict[str, Any]) -> tuple:
        """Calculate signal strength and confidence from signals"""
        # Base strength on momentum strength
        signal_strength = signals.get("strength", 0.0)
        
        # Base confidence on alignment and confirmations
        confidence_factors = []
        
        # Multi-timeframe alignment boosts confidence
        mtf_alignment = signals.get("mtf_alignment", "unknown")
        if mtf_alignment == "strongly_bullish" and signals["direction"] == "bullish":
            confidence_factors.append(1.0)
        elif mtf_alignment == "strongly_bearish" and signals["direction"] == "bearish":
            confidence_factors.append(1.0)
        elif mtf_alignment == "weakly_bullish" and signals["direction"] == "bullish":
            confidence_factors.append(0.8)
        elif mtf_alignment == "weakly_bearish" and signals["direction"] == "bearish":
            confidence_factors.append(0.8)
        elif mtf_alignment == "mixed":
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.5)
            
        # Volume confirmation improves confidence
        volume_trend = signals.get("volume_trend", "normal")
        if volume_trend == "increasing" and signals["direction"] != "neutral":
            confidence_factors.append(0.9)
        elif volume_trend == "decreasing" and signals["direction"] != "neutral":
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.8)
            
        # BTC influence affects confidence for altcoins
        btc_influence = signals.get("btc_influence", {})
        if btc_influence:
            btc_trend = btc_influence.get("btc_trend", "neutral")
            if btc_trend == "bullish" and signals["direction"] == "bullish":
                confidence_factors.append(0.9)
            elif btc_trend == "bearish" and signals["direction"] == "bearish":
                confidence_factors.append(0.9)
            elif btc_trend != "neutral" and btc_trend != signals["direction"]:
                confidence_factors.append(0.6)  # Lower confidence when going against BTC
            else:
                confidence_factors.append(0.8)
        
        # Calculate average confidence
        if confidence_factors:
            confidence = sum(confidence_factors) / len(confidence_factors)
        else:
            confidence = 0.7  # Default confidence
            
        return signal_strength, confidence


class CryptoMeanReversionStrategy(BaseAssetStrategy):
    """
    Mean reversion strategy for cryptocurrency markets
    
    This strategy identifies overbought/oversold conditions in crypto assets
    and trades on the expectation of price returning to a mean value.
    """
    
    def __init__(
        self,
        analysis_service: Optional[AnalysisIntegrationService] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the crypto mean reversion strategy"""
        super().__init__(
            strategy_type=AssetStrategyType.CRYPTO_MEAN_REVERSION,
            asset_class=AssetClass.CRYPTO,
            analysis_service=analysis_service,
            config=config or {}
        )
        
        # Set default configuration parameters
        self.config.setdefault("timeframes", ["15m", "1h", "4h", "1d"])
        self.config.setdefault("primary_timeframe", "1h")
        self.config.setdefault("mean_period", 20)
        self.config.setdefault("bollinger_std_devs", 2.5)  # Wider bands for crypto
        self.config.setdefault("rsi_period", 14)
        self.config.setdefault("rsi_overbought", 75)
        self.config.setdefault("rsi_oversold", 25)
        self.config.setdefault("max_position_size", 0.08)

    async def analyze(self, symbol: str, market_data: Dict[str, MarketData]) -> Dict[str, Any]:
        """
        Analyze market data and generate trading signals
        
        Args:
            symbol: Crypto symbol
            market_data: Dictionary of market data by timeframe
            
        Returns:
            Dictionary with analysis results and signals
        """
        if not self.validate_asset(symbol):
            return {"valid": False, "error": f"Symbol {symbol} is not a cryptocurrency"}
            
        # Ensure we have the required timeframes
        required_tfs = self.config["timeframes"]
        missing_tfs = [tf for tf in required_tfs if tf not in market_data]
        if missing_tfs:
            return {"valid": False, "error": f"Missing timeframes: {missing_tfs}"}
            
        # Get market regime to adjust parameters
        market_context = await self.analysis_service.get_market_context(symbol)
        market_regime = market_context.get("regime", "unknown")
        
        # Get strategy parameters adjusted for current market regime
        params = self.get_strategy_parameters(market_regime)
        params = self.adjust_parameters(params, market_context)
        
        # Run comprehensive analysis
        analysis_result = await self.analysis_service.analyze_all(
            symbol=symbol,
            market_data=market_data,
            components=self.get_required_components(),
            parameters=params
        )
        
        # Extract mean reversion signals
        reversion_signals = self._extract_reversion_signals(analysis_result, params)
        
        # Calculate signal strength and confidence
        signal_strength, confidence = self._calculate_signal_metrics(reversion_signals)
        
        # Calculate position sizing
        position_size = self.get_position_sizing(signal_strength, confidence)
        
        # Compile final result
        return {
            "valid": True,
            "strategy": "crypto_mean_reversion",
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "market_regime": market_regime,
            "signal": {
                "direction": reversion_signals["direction"],
                "strength": signal_strength,
                "confidence": confidence,
                "position_size": position_size,
                "entry_price": market_data[self.config["primary_timeframe"]].close[-1],
                "stop_loss": self._calculate_stop_loss(market_data, reversion_signals, params),
                "take_profit": self._calculate_take_profit(market_data, reversion_signals, params)
            },
            "analysis": reversion_signals,
            "parameters_used": params
        }
    
    def get_strategy_parameters(self, market_regime: str) -> Dict[str, Any]:
        """Get strategy parameters based on market regime"""
        # Base parameters
        base_params = {
            "bollinger_period": self.config["mean_period"],
            "bollinger_std_dev": self.config["bollinger_std_devs"],
            "rsi_period": self.config["rsi_period"],
            "rsi_overbought": self.config["rsi_overbought"],
            "rsi_oversold": self.config["rsi_oversold"],
            "stop_atr_multiple": 1.2,
            "profit_risk_ratio": 1.5
        }
        
        # Adjust based on market regime
        if market_regime == "ranging":
            # Ideal for mean reversion
            return base_params
        elif market_regime == "trending_strong":
            # Be more conservative in strong trends
            return {
                **base_params,
                "bollinger_std_dev": base_params["bollinger_std_dev"] + 0.5,
                "rsi_overbought": 80,
                "rsi_oversold": 20,
                "profit_risk_ratio": 1.2  # Lower targets in trending markets
            }
        elif market_regime == "volatile":
            # Wider bands in volatile markets
            return {
                **base_params,
                "bollinger_std_dev": base_params["bollinger_std_dev"] + 1.0,
                "stop_atr_multiple": 1.5,  # Wider stops
                "profit_risk_ratio": 1.8  # Higher targets in volatile markets
            }
        else:
            return base_params
    
    def adjust_parameters(self, params: Dict[str, Any], market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust parameters based on market context"""
        adjusted = params.copy()
        
        # Adjust based on volatility
        volatility = market_context.get("volatility", {}).get("value", 1.0)
        if volatility > 1.5:
            adjusted["bollinger_std_dev"] *= 1.2
            adjusted["stop_atr_multiple"] *= 1.2
        elif volatility < 0.7:
            adjusted["bollinger_std_dev"] *= 0.9
            adjusted["stop_atr_multiple"] *= 0.9
            
        # Adjust based on trading volume
        volume = market_context.get("volume", {}).get("relative_to_average", 1.0)
        if volume > 1.5:
            # Higher volume might indicate stronger moves
            adjusted["profit_risk_ratio"] *= 1.1
            
        return adjusted
    
    def get_position_sizing(self, signal_strength: float, confidence: float) -> float:
        """Calculate position sizing based on signal strength and confidence"""
        base_size = self.config.get("max_position_size", 0.08)
        
        # Mean reversion strategies often use smaller sizes due to higher uncertainty
        position_size = base_size * signal_strength * confidence * 0.9
        
        # Ensure position size is within limits
        max_size = self.config.get("max_position_size", 0.08)
        min_size = self.config.get("min_position_size", 0.01)
        
        return max(min(position_size, max_size), min_size)
    
    def get_required_components(self) -> List[str]:
        """Get required analysis components"""
        return [
            "technical", "pattern", "multi_timeframe", 
            "sentiment", "market_regime", "crypto_specific"
        ]
        
    def _extract_reversion_signals(self, analysis_result: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract mean reversion signals from analysis results"""
        # Extract relevant data
        technical = analysis_result.get("technical", {})
        
        # Get Bollinger Band information
        bb = technical.get("bollinger_bands", {})
        bb_width = bb.get("width", 1.0)
        bb_position = bb.get("percent_b", 0.5)
        
        # Get RSI information
        rsi = technical.get("rsi", 50)
        
        # Determine signal direction
        direction = "neutral"
        if bb_position <= 0.05 and rsi < params["rsi_oversold"]:
            direction = "bullish"  # Oversold condition
        elif bb_position >= 0.95 and rsi > params["rsi_overbought"]:
            direction = "bearish"  # Overbought condition
            
        # Calculate signal strength based on extremity of readings
        strength = 0.0
        if direction == "bullish":
            # How far below oversold and outside lower band
            rsi_factor = max(0, (params["rsi_oversold"] - rsi) / 10)  # 0-3 typically
            bb_factor = max(0, 0.05 - bb_position) * 10  # 0-0.5 typically
            strength = min((rsi_factor + bb_factor) / 3.5, 1.0)
        elif direction == "bearish":
            # How far above overbought and outside upper band
            rsi_factor = max(0, (rsi - params["rsi_overbought"]) / 10)  # 0-3 typically
            bb_factor = max(0, bb_position - 0.95) * 10  # 0-0.5 typically
            strength = min((rsi_factor + bb_factor) / 3.5, 1.0)
            
        return {
            "direction": direction,
            "strength": strength,
            "rsi": rsi,
            "bollinger_bands": bb,
            "bb_width": bb_width,
            "bb_position": bb_position
        }
    
    def _calculate_signal_metrics(self, signals: Dict[str, Any]) -> tuple:
        """Calculate signal strength and confidence"""
        strength = signals.get("strength", 0.0)
        
        # For mean reversion, confidence is related to how extreme the readings are
        # and the width of the Bollinger Bands
        bb_width = signals.get("bb_width", 1.0)
        
        # Normalize BB width - wider bands are better for mean reversion
        # as they indicate more extreme moves
        normalized_width = min(bb_width / 2.0, 1.0)
        
        # Higher confidence when BB width is larger and position is more extreme
        confidence = normalized_width * 0.7 + 0.3
        
        return strength, confidence
    
    def _calculate_stop_loss(self, market_data: Dict[str, MarketData], signals: Dict[str, Any], params: Dict[str, Any]) -> float:
        """Calculate appropriate stop loss level"""
        tf = self.config["primary_timeframe"]
        close = market_data[tf].close[-1]
        
        # For mean reversion, often use the most recent extreme as a stop
        if signals["direction"] == "bullish":
            recent_low = min(market_data[tf].low[-5:])
            return max(recent_low * 0.99, close * 0.97)  # 1-3% below recent low
        else:
            recent_high = max(market_data[tf].high[-5:])
            return min(recent_high * 1.01, close * 1.03)  # 1-3% above recent high
    
    def _calculate_take_profit(self, market_data: Dict[str, MarketData], signals: Dict[str, Any], params: Dict[str, Any]) -> float:
        """Calculate take profit level"""
        tf = self.config["primary_timeframe"]
        close = market_data[tf].close[-1]
        
        # For mean reversion, target the mean or mid-point of the Bollinger Bands
        bb = signals.get("bollinger_bands", {})
        bb_mid = bb.get("middle", close)
        
        if signals["direction"] == "bullish":
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
    
    def __init__(
        self,
        analysis_service: Optional[AnalysisIntegrationService] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the crypto volatility strategy"""
        super().__init__(
            strategy_type=AssetStrategyType.CRYPTO_VOLATILITY,
            asset_class=AssetClass.CRYPTO,
            analysis_service=analysis_service,
            config=config or {}
        )
        
        # Set default configuration parameters
        self.config.setdefault("timeframes", ["5m", "15m", "1h", "4h"])
        self.config.setdefault("primary_timeframe", "1h")
        self.config.setdefault("atr_period", 14)
        self.config.setdefault("atr_threshold", 1.5)  # Significantly higher volatility
        self.config.setdefault("max_position_size", 0.05)  # Smaller positions due to higher risk
        self.config.setdefault("detection_period", 48)  # Hours to look for volatility patterns
        
    async def analyze(self, symbol: str, market_data: Dict[str, MarketData]) -> Dict[str, Any]:
        """Analyze market data and generate trading signals"""
        if not self.validate_asset(symbol):
            return {"valid": False, "error": f"Symbol {symbol} is not a cryptocurrency"}
            
        # Ensure we have the required timeframes
        required_tfs = self.config["timeframes"]
        missing_tfs = [tf for tf in required_tfs if tf not in market_data]
        if missing_tfs:
            return {"valid": False, "error": f"Missing timeframes: {missing_tfs}"}
            
        # Get market context
        market_context = await self.analysis_service.get_market_context(symbol)
        market_regime = market_context.get("regime", "unknown")
        
        # Get adjusted parameters
        params = self.get_strategy_parameters(market_regime)
        params = self.adjust_parameters(params, market_context)
        
        # Run full analysis
        analysis_result = await self.analysis_service.analyze_all(
            symbol=symbol,
            market_data=market_data,
            components=self.get_required_components(),
            parameters=params
        )
        
        # Extract volatility signals
        volatility_signals = self._extract_volatility_signals(analysis_result, params)
        
        # Calculate signal metrics
        signal_strength, confidence = self._calculate_signal_metrics(volatility_signals)
        
        # Calculate position sizing
        position_size = self.get_position_sizing(signal_strength, confidence)
        
        return {
            "valid": True,
            "strategy": "crypto_volatility",
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "market_regime": market_regime,
            "signal": {
                "direction": volatility_signals["direction"],
                "strength": signal_strength,
                "confidence": confidence,
                "position_size": position_size,
                "entry_price": market_data[self.config["primary_timeframe"]].close[-1],
                "stop_loss": self._calculate_stop_loss(market_data, volatility_signals, params),
                "take_profit": self._calculate_take_profit(market_data, volatility_signals, params)
            },
            "analysis": volatility_signals,
            "parameters_used": params
        }
        
    def get_strategy_parameters(self, market_regime: str) -> Dict[str, Any]:
        """Get strategy parameters based on market regime"""
        # Base parameters
        base_params = {
            "atr_period": self.config["atr_period"],
            "atr_threshold": self.config["atr_threshold"],
            "volatility_lookback": 14,
            "bollinger_period": 20,
            "bollinger_std_dev": 2.0,
            "stop_atr_multiple": 2.0,  # Wide stops for volatile markets
            "profit_risk_ratio": 1.5
        }
        
        # Adjust for market regime
        if market_regime == "volatile":
            # Ideal for volatility strategy
            return {
                **base_params,
                "profit_risk_ratio": 2.0  # Higher targets in volatile markets
            }
        elif market_regime == "trending_strong":
            return {
                **base_params,
                "atr_threshold": base_params["atr_threshold"] * 1.2  # Need higher volatility
            }
        elif market_regime == "ranging":
            return {
                **base_params,
                "atr_threshold": base_params["atr_threshold"] * 1.5  # Need much higher volatility
            }
        else:
            return base_params
        
    def adjust_parameters(self, params: Dict[str, Any], market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust parameters based on market context"""
        adjusted = params.copy()
        
        # Volatility strategy is already volatility-focused, but may need
        # adjustments based on other factors
        
        # Consider current volatility level
        volatility = market_context.get("volatility", {}).get("value", 1.0)
        if volatility > 2.0:  # Extremely volatile
            adjusted["stop_atr_multiple"] *= 1.3  # Even wider stops
            adjusted["profit_risk_ratio"] *= 1.2  # Higher targets
        
        # Consider trading volume
        volume = market_context.get("volume", {}).get("relative_to_average", 1.0)
        if volume > 2.0:
            # Higher volume supports volatility plays
            adjusted["atr_threshold"] *= 0.9  # Lower threshold
        
        # Consider news events
        has_major_news = market_context.get("news", {}).get("has_major_events", False)
        if has_major_news:
            # News can drive volatility
            adjusted["atr_threshold"] *= 0.9
            
        return adjusted
        
    def get_position_sizing(self, signal_strength: float, confidence: float) -> float:
        """Calculate position sizing based on signal strength and confidence"""
        # Volatility strategies use smaller position sizes due to higher risk
        base_size = self.config.get("max_position_size", 0.05)
        
        # Calculate position size with an additional reduction factor
        position_size = base_size * signal_strength * confidence * 0.7
        
        # Ensure within limits
        max_size = self.config.get("max_position_size", 0.05)
        min_size = self.config.get("min_position_size", 0.01)
        
        return max(min(position_size, max_size), min_size)
        
    def get_required_components(self) -> List[str]:
        """Get required analysis components"""
        return [
            "technical", "pattern", "volatility", 
            "sentiment", "market_regime", "news_events"
        ]
        
    def _extract_volatility_signals(self, analysis_result: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract volatility-focused signals from analysis results"""
        technical = analysis_result.get("technical", {})
        volatility = analysis_result.get("volatility", {})
        
        # Get ATR and other volatility metrics
        atr = volatility.get("atr", 0)
        atr_percent = volatility.get("atr_percent", 0)
        historical_volatility = volatility.get("historical_volatility", 0)
        
        # Get Bollinger Band data
        bb = technical.get("bollinger_bands", {})
        bb_width = bb.get("width", 1.0)
        
        # Check for volatility expansion/contraction
        volatility_increasing = volatility.get("trend", "") == "increasing"
        volatility_decreasing = volatility.get("trend", "") == "decreasing"
        
        # Check for volatility events
        volatility_events = volatility.get("events", [])
        recent_events = [e for e in volatility_events if e.get("hours_ago", 999) <= self.config["detection_period"]]
        
        # Determine direction - volatility strategies can be directional or non-directional
        direction = "neutral"
        
        # If volatility is increasing significantly, we can attempt to determine
        # the likely breakout direction
        if volatility_increasing and atr_percent > params["atr_threshold"]:
            # Check technical indicators for direction
            rsi = technical.get("rsi", 50)
            macd_hist = technical.get("macd", {}).get("histogram", 0)
            
            # Combine directional indicators
            if rsi > 60 and macd_hist > 0:
                direction = "bullish"
            elif rsi < 40 and macd_hist < 0:
                direction = "bearish"
        
        # Calculate strength based on volatility metrics
        # Higher values indicate stronger volatility signals
        strength_factors = []
        
        # ATR percent relative to threshold
        if atr_percent > 0:
            strength_factors.append(min(atr_percent / params["atr_threshold"], 2.0) / 2.0)
        
        # BB width relative to historical
        if bb_width > 1.3:  # Significantly wider bands
            strength_factors.append(min(bb_width / 1.3, 2.0) / 2.0)
            
        # Recent volatility events
        if recent_events:
            event_count = len(recent_events)
            strength_factors.append(min(event_count / 3.0, 1.0))
            
        # Calculate average strength
        strength = sum(strength_factors) / max(len(strength_factors), 1)
        
        return {
            "direction": direction,
            "strength": min(strength, 1.0),
            "atr": atr,
            "atr_percent": atr_percent,
            "historical_volatility": historical_volatility,
            "bb_width": bb_width,
            "volatility_trend": volatility.get("trend", "stable"),
            "recent_volatility_events": recent_events
        }
        
    def _calculate_signal_metrics(self, signals: Dict[str, Any]) -> tuple:
        """Calculate signal strength and confidence"""
        strength = signals.get("strength", 0.0)
        
        # For volatility strategy, confidence depends on:
        # 1. How clear the volatility signals are
        # 2. Recent volatility events
        
        # Base confidence on volatility trend
        if signals.get("volatility_trend") == "strongly_increasing":
            base_confidence = 0.9
        elif signals.get("volatility_trend") == "increasing":
            base_confidence = 0.8
        elif signals.get("volatility_trend") == "decreasing":
            base_confidence = 0.6  # Less confident when volatility is decreasing
        else:
            base_confidence = 0.7
            
        # Adjust confidence based on recent events
        recent_events = signals.get("recent_volatility_events", [])
        if recent_events:
            # More events = higher confidence
            event_factor = min(len(recent_events) / 5.0, 1.0) * 0.2
            confidence = base_confidence + event_factor
        else:
            confidence = base_confidence
            
        return strength, min(confidence, 1.0)
        
    def _calculate_stop_loss(self, market_data: Dict[str, MarketData], signals: Dict[str, Any], params: Dict[str, Any]) -> float:
        """Calculate stop loss level for volatility strategy"""
        tf = self.config["primary_timeframe"]
        close = market_data[tf].close[-1]
        
        # For volatility strategies, wide stops are needed
        atr = signals.get("atr", close * 0.02)  # Default to 2% if ATR not available
        stop_distance = atr * params["stop_atr_multiple"]
        
        if signals["direction"] == "bullish":
            return close - stop_distance
        elif signals["direction"] == "bearish":
            return close + stop_distance
        else:
            # For neutral volatility plays, choose based on recent support/resistance
            if len(market_data[tf].close) >= 20:
                # Use recent highs/lows for support/resistance
                recent_high = max(market_data[tf].high[-20:])
                recent_low = min(market_data[tf].low[-20:])
                
                # Place stop outside recent range
                return recent_low - stop_distance * 0.5
            else:
                # Default to percentage-based
                return close * 0.92
    
    def _calculate_take_profit(self, market_data: Dict[str, MarketData], signals: Dict[str, Any], params: Dict[str, Any]) -> float:
        """Calculate take profit level for volatility strategy"""
        tf = self.config["primary_timeframe"]
        close = market_data[tf].close[-1]
        stop_loss = self._calculate_stop_loss(market_data, signals, params)
        
        risk = abs(close - stop_loss)
        reward = risk * params["profit_risk_ratio"]
        
        if signals["direction"] == "bullish":
            return close + reward
        elif signals["direction"] == "bearish":
            return close - reward
        else:
            # For neutral strategies, could use a percentage target
            return close * 1.1  # 10% move
