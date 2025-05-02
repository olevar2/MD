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


class ForexTrendStrategy(BaseAssetStrategy):
    """
    Forex-specific trend-following strategy
    
    This strategy focuses on capturing medium to long-term trends in forex markets.
    It integrates multiple timeframe analysis, sentiment analysis, and pattern recognition.
    """
    
    def __init__(
        self,
        analysis_service: AnalysisIntegrationService = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize a forex trend strategy
        
        Args:
            analysis_service: Analysis integration service
            config: Strategy configuration
        """
        super().__init__(
            strategy_type=AssetStrategyType.FOREX_TREND,
            asset_class=AssetClass.FOREX,
            analysis_service=analysis_service,
            config=config or {}
        )
        
        # Default configuration
        self.default_config = {
            "timeframes": ["15m", "1h", "4h", "1d"],  # Multi-timeframe analysis
            "primary_timeframe": "4h",                # Primary decision timeframe
            "trend_detection": {
                "ema_fast": 8,                        # Fast EMA period
                "ema_slow": 21,                       # Slow EMA period
                "macd_fast": 12,                      # MACD fast period
                "macd_slow": 26,                      # MACD slow period
                "macd_signal": 9,                     # MACD signal period
                "atr_period": 14,                     # ATR period for volatility
            },
            "entry_filters": {
                "min_trend_strength": 0.6,            # Minimum trend strength
                "min_signal_confidence": 0.7,         # Minimum signal confidence
                "min_pullback": 0.3,                  # Minimum pullback percentage
                "rsi_oversold": 30,                   # RSI oversold threshold for long
                "rsi_overbought": 70,                 # RSI overbought threshold for short
            },
            "exit_rules": {
                "trailing_stop_atr_mult": 2.0,        # Trailing stop ATR multiplier
                "take_profit_atr_mult": 3.0,          # Take profit ATR multiplier
                "max_holding_periods": 20,            # Maximum holding periods
            },
            "risk_management": {
                "max_risk_per_trade": 0.01,           # Maximum risk per trade (1%)
                "max_correlated_exposure": 0.05,      # Maximum correlated exposure (5%)
                "session_filters_enabled": True,      # Enable session-based filters
            },
            "currency_correlations": {
                "check_correlations": True,           # Check for currency correlations
                "correlation_threshold": 0.7,         # Correlation threshold
            },
            "session_preferences": {
                "london_weight": 1.0,                 # London session weight
                "ny_weight": 1.0,                     # New York session weight
                "asia_weight": 0.7,                   # Asia session weight
                "sydney_weight": 0.7,                 # Sydney session weight
            },
        }
        
        # Merge provided config with defaults
        self.config = {**self.default_config, **self.config}
        
        # Market regime specific parameters
        self.regime_params = {
            "trending": {
                "entry_filters": {
                    "min_trend_strength": 0.5,        # Lower threshold in trending market
                    "min_signal_confidence": 0.6,
                    "min_pullback": 0.2,
                },
                "exit_rules": {
                    "trailing_stop_atr_mult": 2.5,    # Wider trailing stop
                    "take_profit_atr_mult": 4.0,      # Higher take profit
                }
            },
            "ranging": {
                "entry_filters": {
                    "min_trend_strength": 0.7,        # Higher threshold in ranging market
                    "min_signal_confidence": 0.8,
                    "min_pullback": 0.4,
                },
                "exit_rules": {
                    "trailing_stop_atr_mult": 1.5,    # Tighter trailing stop
                    "take_profit_atr_mult": 2.0,      # Lower take profit
                }
            },
            "volatile": {
                "entry_filters": {
                    "min_trend_strength": 0.8,        # Higher threshold in volatile market
                    "min_signal_confidence": 0.85,
                    "min_pullback": 0.5,
                },
                "exit_rules": {
                    "trailing_stop_atr_mult": 3.0,    # Wider trailing stop for volatility
                    "take_profit_atr_mult": 3.0,
                }
            }
        }
    
    async def analyze(self, symbol: str, market_data: Dict[str, MarketData]) -> Dict[str, Any]:
        """
        Analyze market data and generate trading signals
        
        Args:
            symbol: Asset symbol
            market_data: Dictionary of market data by timeframe
            
        Returns:
            Dictionary with analysis results and signals
        """
        # Validate this is a forex asset
        if not self.validate_asset(symbol):
            return {"error": f"Symbol {symbol} is not a valid forex pair for this strategy"}
        
        # Get components we need for analysis
        components = self.get_required_components()
        
        # Run the comprehensive analysis using the integration service
        analysis_results = await self.analysis_service.analyze_asset(
            symbol=symbol,
            market_data=market_data,
            include_components=components
        )
        
        # Check for errors in analysis
        if "error" in analysis_results:
            return {"error": analysis_results["error"]}
            
        # Get market regime from analysis results
        market_regime = self._detect_market_regime(analysis_results)
        
        # Get parameters based on market regime
        params = self.get_strategy_parameters(market_regime)
        
        # Apply asset-specific adjustments
        adjusted_params = self.adjust_parameters(params, analysis_results)
        
        # Generate trading signals
        signals = self._generate_signals(symbol, analysis_results, adjusted_params)
        
        # Calculate position sizing if a signal is generated
        if signals["signal"] != "neutral":
            signals["position_size"] = self.get_position_sizing(
                signals["strength"], 
                signals["confidence"]
            )
        
        # Include parameters and market regime in results
        signals["market_regime"] = market_regime
        signals["parameters"] = adjusted_params
        signals["forex_specific"] = self._get_forex_specific_insights(symbol, analysis_results)
        
        return signals
    
    def get_strategy_parameters(self, market_regime: str) -> Dict[str, Any]:
        """
        Get strategy parameters based on market regime
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Dictionary with strategy parameters
        """
        # Start with base parameters
        params = {**self.config}
        
        # Apply regime-specific overrides if available
        if market_regime in self.regime_params:
            regime_specific = self.regime_params[market_regime]
            
            # Deep merge of nested dictionaries
            for key, value in regime_specific.items():
                if isinstance(value, dict) and key in params and isinstance(params[key], dict):
                    params[key] = {**params[key], **value}
                else:
                    params[key] = value
        
        return params
    
    def adjust_parameters(self, params: Dict[str, Any], market_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust strategy parameters based on market context
        
        Args:
            params: Current strategy parameters
            market_context: Market context information
            
        Returns:
            Adjusted parameters
        """
        adjusted = {**params}  # Create a copy to avoid modifying the original
        
        # Check for high volatility and adjust parameters
        if "asset_specific" in market_context and "session_activity" in market_context["asset_specific"]:
            session_activity = market_context["asset_specific"]["session_activity"]
            
            # Adjust based on current forex session
            active_session = session_activity.get("active_session", "none")
            if active_session == "london_open" or active_session == "ny_open":
                # Reduce filter thresholds during high-activity sessions
                if "entry_filters" in adjusted:
                    adjusted["entry_filters"]["min_signal_confidence"] *= 0.9
                    
            elif active_session == "asia" or active_session == "sydney":
                # Increase filter thresholds during low-activity sessions
                if "entry_filters" in adjusted:
                    adjusted["entry_filters"]["min_signal_confidence"] *= 1.1
        
        # Adjust for spread considerations
        if "asset_specific" in market_context and "spread_viability" in market_context["asset_specific"]:
            if not market_context["asset_specific"]["spread_viability"]:
                # If spread is too high relative to signal, increase thresholds
                if "entry_filters" in adjusted:
                    adjusted["entry_filters"]["min_signal_confidence"] *= 1.2
                    adjusted["entry_filters"]["min_trend_strength"] *= 1.2
        
        return adjusted
    
    def get_position_sizing(self, signal_strength: float, confidence: float) -> float:
        """
        Calculate position sizing based on signal strength and confidence
        
        Args:
            signal_strength: Strength of the trading signal
            confidence: Confidence in the signal
            
        Returns:
            Position size as a percentage of available capital
        """
        max_risk = self.config["risk_management"]["max_risk_per_trade"]
        
        # Scale position size by signal strength and confidence
        position_scale = signal_strength * confidence
        
        # Apply scaling but ensure we're within risk limits
        position_size = max_risk * position_scale
        
        # Limit to maximum risk
        return min(position_size, max_risk)
    
    def get_required_components(self) -> List[str]:
        """
        Get list of required analysis components for this strategy
        
        Returns:
            List of component names
        """
        return [
            "technical", "pattern", "multi_timeframe", 
            "sentiment", "market_regime"
        ]
    
    def _detect_market_regime(self, analysis_results: Dict[str, Any]) -> str:
        """Detect market regime from analysis results"""
        if "components" in analysis_results and "market_regime" in analysis_results["components"]:
            regime = analysis_results["components"]["market_regime"].get("regime")
            if regime:
                return regime
        
        # Default to trending if no regime detected
        return "trending"
    
    def _generate_signals(
        self,
        symbol: str,
        analysis_results: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate trading signals from analysis results"""
        # Initialize signal object
        signal = {
            "symbol": symbol,
            "signal": "neutral",
            "strength": 0.0,
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Extract directional signals from analysis
        if "overall_signal" in analysis_results:
            signal["signal"] = analysis_results["overall_signal"]
            
        if "overall_confidence" in analysis_results:
            signal["confidence"] = analysis_results["overall_confidence"]
        
        # Get signal strength from confidence scores
        bullish_scores = []
        bearish_scores = []
        
        if "confidence_scores" in analysis_results:
            for key, score in analysis_results["confidence_scores"].items():
                if "bullish" in key:
                    bullish_scores.append(score)
                elif "bearish" in key:
                    bearish_scores.append(score)
        
        # Calculate weighted strength based on directional scores
        if bullish_scores:
            bullish_strength = sum(bullish_scores) / len(bullish_scores)
        else:
            bullish_strength = 0.0
            
        if bearish_scores:
            bearish_strength = sum(bearish_scores) / len(bearish_scores)
        else:
            bearish_strength = 0.0
            
        # Set strength based on signal direction
        if signal["signal"] == "bullish":
            signal["strength"] = bullish_strength
        elif signal["signal"] == "bearish":
            signal["strength"] = bearish_strength
        else:
            signal["strength"] = 0.0
        
        # Apply minimum thresholds from parameters
        min_confidence = params["entry_filters"]["min_signal_confidence"]
        min_strength = params["entry_filters"]["min_trend_strength"]
        
        # Only generate a non-neutral signal if it meets minimum thresholds
        if signal["confidence"] < min_confidence or signal["strength"] < min_strength:
            signal["signal"] = "neutral"
            signal["explanation"] = f"Signal below minimum thresholds (confidence: {signal['confidence']:.2f} < {min_confidence:.2f} or strength: {signal['strength']:.2f} < {min_strength:.2f})"
        else:
            signal["explanation"] = f"Signal meets criteria with confidence {signal['confidence']:.2f} and strength {signal['strength']:.2f}"
            
        return signal
    
    def _get_forex_specific_insights(self, symbol: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get forex-specific insights from analysis results"""
        insights = {}
        
        # Extract session information if available
        if "asset_specific" in analysis_results and "session_activity" in analysis_results["asset_specific"]:
            insights["session"] = analysis_results["asset_specific"]["session_activity"]
            
        # Extract currency correlations if available
        if "components" in analysis_results and "correlation" in analysis_results["components"]:
            insights["correlations"] = analysis_results["components"]["correlation"].get("correlations", {})
            
            # Check for potential correlation-based exposure risks
            if self.config["currency_correlations"]["check_correlations"]:
                threshold = self.config["currency_correlations"]["correlation_threshold"]
                insights["high_correlations"] = self._check_high_correlations(symbol, insights["correlations"], threshold)
                
        return insights
    
    def _check_high_correlations(
        self,
        symbol: str,
        correlations: Dict[str, float],
        threshold: float
    ) -> Dict[str, Any]:
        """Check for high correlations that might impact risk"""
        result = {
            "positive_correlated": [],
            "negative_correlated": [],
            "has_risk_concentration": False
        }
        
        base_currency = symbol[:3]  # Assumes format like EURUSD
        quote_currency = symbol[3:6]  # Assumes format like EURUSD
        
        # Check all correlations against threshold
        for pair, corr_value in correlations.items():
            if pair == symbol:
                continue
                
            if abs(corr_value) >= threshold:
                if corr_value > 0:
                    result["positive_correlated"].append({"pair": pair, "correlation": corr_value})
                else:
                    result["negative_correlated"].append({"pair": pair, "correlation": corr_value})
                    
                # Check if this correlation involves the same currencies
                # This would indicate a risk concentration
                if base_currency in pair or quote_currency in pair:
                    result["has_risk_concentration"] = True
        
        return result


class ForexRangeStrategy(BaseAssetStrategy):
    """
    Forex-specific range trading strategy
    
    This strategy is designed for range-bound markets, focusing on 
    support/resistance levels and mean reversion techniques.
    """
    
    def __init__(
        self,
        analysis_service: AnalysisIntegrationService = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize a forex range strategy
        
        Args:
            analysis_service: Analysis integration service
            config: Strategy configuration
        """
        super().__init__(
            strategy_type=AssetStrategyType.FOREX_RANGE,
            asset_class=AssetClass.FOREX,
            analysis_service=analysis_service,
            config=config or {}
        )
        
        # Default configuration
        self.default_config = {
            "timeframes": ["5m", "15m", "1h", "4h"],  # Multi-timeframe analysis
            "primary_timeframe": "1h",                # Primary decision timeframe
            "range_detection": {
                "min_range_periods": 20,              # Minimum periods to confirm range
                "max_trend_strength": 30,             # Maximum ADX for range market
                "bollinger_periods": 20,              # Bollinger Band periods
                "bollinger_std_dev": 2.0,             # Bollinger Band std dev
                "rsi_period": 14,                     # RSI period
            },
            "entry_rules": {
                "rsi_oversold": 30,                   # RSI oversold threshold for long
                "rsi_overbought": 70,                 # RSI overbought threshold for short
                "min_band_distance": 0.5,             # Minimum distance to band (%)
                "min_distance_to_level": 0.1,         # Min distance to S/R level (%)
                "min_level_touches": 2,               # Minimum touches to confirm level
            },
            "exit_rules": {
                "target_opposite_band": True,         # Target opposite Bollinger Band
                "max_holding_periods": 10,            # Maximum holding periods
                "trailing_stop_enabled": True,        # Enable trailing stop
                "trailing_stop_activation": 0.5,      # Activation threshold (% to target)
            },
            "risk_management": {
                "max_risk_per_trade": 0.005,          # Maximum risk per trade (0.5%)
                "max_correlated_exposure": 0.03,      # Maximum correlated exposure (3%)
            }
        }
        
        # Merge provided config with defaults
        self.config = {**self.default_config, **self.config}
        
        # Market regime specific parameters
        self.regime_params = {
            "ranging": {
                "entry_rules": {
                    "min_band_distance": 0.4,         # Less distance needed in confirmed range
                },
                "exit_rules": {
                    "target_opposite_band": True,
                }
            },
            "trending": {
                "entry_rules": {
                    "min_band_distance": 0.7,         # More distance needed in trend
                },
                "exit_rules": {
                    "target_opposite_band": False,    # Don't target opposite band in trend
                    "max_holding_periods": 5,         # Shorter holding in trend
                }
            },
            "volatile": {
                "entry_rules": {
                    "min_band_distance": 0.8,         # More distance needed in volatility
                    "min_level_touches": 3,           # Need more touches in volatility
                },
                "risk_management": {
                    "max_risk_per_trade": 0.003,      # Lower risk in volatility
                }
            }
        }
    
    async def analyze(self, symbol: str, market_data: Dict[str, MarketData]) -> Dict[str, Any]:
        """Implementation for analyze method - similar to ForexTrendStrategy"""
        # Similar implementation to ForexTrendStrategy.analyze()
        # Forex-specific range strategy implementation
        pass
    
    def get_strategy_parameters(self, market_regime: str) -> Dict[str, Any]:
        """Implementation for get_strategy_parameters - similar to ForexTrendStrategy"""
        # Similar implementation to ForexTrendStrategy.get_strategy_parameters()
        pass
    
    def adjust_parameters(self, params: Dict[str, Any], market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation for adjust_parameters - similar to ForexTrendStrategy"""
        # Similar implementation to ForexTrendStrategy.adjust_parameters()
        pass
    
    def get_position_sizing(self, signal_strength: float, confidence: float) -> float:
        """Implementation for get_position_sizing - similar to ForexTrendStrategy"""
        # Similar implementation to ForexTrendStrategy.get_position_sizing()
        pass
        
    # Other forex range-specific methods would be implemented here


class ForexBreakoutStrategy(BaseAssetStrategy):
    """
    Forex-specific breakout strategy
    
    This strategy focuses on identifying and trading breakouts from consolidation
    periods, key support/resistance levels, and chart patterns.
    """
    
    def __init__(
        self,
        analysis_service: AnalysisIntegrationService = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize a forex breakout strategy
        
        Args:
            analysis_service: Analysis integration service
            config: Strategy configuration
        """
        super().__init__(
            strategy_type=AssetStrategyType.FOREX_BREAKOUT,
            asset_class=AssetClass.FOREX,
            analysis_service=analysis_service,
            config=config or {}
        )
        
        # Default configuration
        self.default_config = {
            "timeframes": ["15m", "1h", "4h", "1d"],  # Multi-timeframe analysis
            "primary_timeframe": "1h",                # Primary decision timeframe
            "breakout_detection": {
                "consolidation_periods": 20,          # Periods to detect consolidation
                "volatility_contraction": 0.5,        # Required volatility contraction
                "volume_increase_req": 1.5,           # Required volume increase
                "min_level_touches": 2,               # Minimum touches to confirm level
                "max_false_breakout": 0.3,            # Max allowed price penetration
            },
            "entry_rules": {
                "min_breakout_size": 0.5,             # Minimum breakout size as ATR multiple
                "confirmation_candles": 1,            # Candles to confirm breakout
                "use_momentum": True,                 # Consider momentum indicators
            },
            "exit_rules": {
                "target_projection": 2.0,             # Target as multiple of breakout size
                "trailing_stop_atr_mult": 1.5,        # Trailing stop ATR multiplier
                "max_holding_periods": 15,            # Maximum holding periods
            },
            "risk_management": {
                "max_risk_per_trade": 0.01,           # Maximum risk per trade (1%)
                "reduce_overnight_exposure": True,    # Reduce exposure overnight
            },
            "pattern_preferences": {
                "favor_chart_patterns": True,         # Give priority to chart patterns
                "min_pattern_quality": 0.7,           # Minimum pattern quality score
            },
        }
        
        # Merge provided config with defaults
        self.config = {**self.default_config, **self.config}
        
        # Market regime specific parameters
        self.regime_params = {
            "trending": {
                "entry_rules": {
                    "min_breakout_size": 0.4,         # Smaller breakout needed in trend
                    "confirmation_candles": 1,
                },
                "exit_rules": {
                    "target_projection": 2.5,         # Higher target in trending market
                }
            },
            "ranging": {
                "entry_rules": {
                    "min_breakout_size": 0.7,         # Larger breakout needed in range
                    "confirmation_candles": 2,        # More confirmation needed
                },
                "exit_rules": {
                    "target_projection": 1.5,         # Lower target in ranging market
                }
            },
            "volatile": {
                "entry_rules": {
                    "min_breakout_size": 0.9,         # Larger breakout needed in volatility
                    "confirmation_candles": 2,        # More confirmation needed
                },
                "risk_management": {
                    "max_risk_per_trade": 0.007,      # Lower risk in volatility
                }
            }
        }
    
    async def analyze(self, symbol: str, market_data: Dict[str, MarketData]) -> Dict[str, Any]:
        """Implementation for analyze method - similar to ForexTrendStrategy"""
        # Similar implementation to ForexTrendStrategy.analyze()
        # Forex-specific breakout strategy implementation
        pass
    
    def get_strategy_parameters(self, market_regime: str) -> Dict[str, Any]:
        """Implementation for get_strategy_parameters - similar to ForexTrendStrategy"""
        # Similar implementation to ForexTrendStrategy.get_strategy_parameters()
        pass
    
    def adjust_parameters(self, params: Dict[str, Any], market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation for adjust_parameters - similar to ForexTrendStrategy"""
        # Similar implementation to ForexTrendStrategy.adjust_parameters()
        pass
    
    def get_position_sizing(self, signal_strength: float, confidence: float) -> float:
        """Implementation for get_position_sizing - similar to ForexTrendStrategy"""
        # Similar implementation to ForexTrendStrategy.get_position_sizing()
        pass
        
    # Other forex breakout-specific methods would be implemented here
