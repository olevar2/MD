"""
MultiTimeframeConfluenceStrategy

This module implements a trading strategy that looks for alignments across multiple
timeframes, combining signals from different time horizons for higher probability trades.

Part of Phase 4 implementation to enhance the adaptive trading capabilities.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from strategy_execution_engine.strategies.advanced_ta_strategy import AdvancedTAStrategy
from analysis_engine.services.tool_effectiveness import MarketRegime
from analysis_engine.analysis.technical_indicators import TechnicalIndicators


class MultiTimeframeConfluenceStrategy(AdvancedTAStrategy):
    """
    Trading strategy that looks for alignments across multiple timeframes,
    combining signals from different time horizons for higher probability trades.
    """
    
    def _init_strategy_config(self) -> None:
        """Initialize strategy-specific configuration parameters."""
        # Define timeframe importance weights (higher weight = more important)
        self.timeframe_weights = {
            "1m": 0.3,
            "5m": 0.4,
            "15m": 0.5,
            "30m": 0.6,
            "1h": 0.8,
            "4h": 1.0,
            "daily": 1.2,
            "weekly": 1.5,
            "monthly": 1.8
        }
        
        # Strategy-specific parameters
        self.adaptive_params = {
            "min_timeframe_agreement": 3,      # Minimum number of timeframes that must agree
            "directional_agreement_threshold": 0.7,  # Percentage of timeframes that must show same direction
            "primary_tf_multiplier": 1.5,      # Importance multiplier for primary timeframe
            "secondary_tf_multiplier": 1.0,    # Importance multiplier for secondary timeframes
            "confirmation_bars": 2,            # Number of bars needed for confirmation
            "entry_filter_strength": 0.7,      # Strength of entry filters (0.0-1.0)
            "stop_loss_buffer_pct": 0.05,      # Additional buffer for stop loss
            "atr_multiple_sl": 1.0,            # ATR multiple for stop loss
            "atr_multiple_tp": 2.0             # ATR multiple for take profit
        }
        
        # Configuration specific to multi-timeframe strategies
        self.config.update({
            "entry_timeframe": "1h",           # Timeframe used for entry signals
            "major_timeframes": ["4h", "daily", "weekly"],  # Major trend timeframes
            "minor_timeframes": ["5m", "15m", "30m"],      # Minor trend timeframes
            "signal_types": ["price_action", "indicator", "trend", "support_resistance"],
            "use_confirmation_divergence": True,   # Use divergence as confirmation
            "enable_partial_position_sizing": True, # Allow partial positions based on strength
            "minimum_confluence_score": 0.7        # Minimum score to generate signal
        })
        
        self.logger.info(f"Initialized {self.name} with multi-timeframe parameters")
    
    def _adapt_parameters_to_regime(self, regime: MarketRegime) -> None:
        """Adjust strategy parameters based on the current market regime."""
        self.logger.info(f"Adapting parameters to {regime} regime")
        
        if regime == MarketRegime.TRENDING:
            # In trending markets, emphasize trend continuations across timeframes
            self.adaptive_params["min_timeframe_agreement"] = 3
            self.adaptive_params["directional_agreement_threshold"] = 0.7
            self.adaptive_params["entry_filter_strength"] = 0.6  # Less strict filters
            self.adaptive_params["atr_multiple_tp"] = 2.5  # Larger targets
            self.config["major_timeframes"] = ["4h", "daily", "weekly"]  # Longer timeframes
            
        elif regime == MarketRegime.RANGING:
            # In ranging markets, require more timeframe agreement
            self.adaptive_params["min_timeframe_agreement"] = 4
            self.adaptive_params["directional_agreement_threshold"] = 0.8
            self.adaptive_params["entry_filter_strength"] = 0.8  # Stricter filters
            self.adaptive_params["atr_multiple_tp"] = 1.5  # Smaller targets
            self.config["major_timeframes"] = ["1h", "4h", "daily"]  # Medium timeframes
            
        elif regime == MarketRegime.VOLATILE:
            # In volatile markets, be very selective
            self.adaptive_params["min_timeframe_agreement"] = 5
            self.adaptive_params["directional_agreement_threshold"] = 0.85
            self.adaptive_params["entry_filter_strength"] = 0.9  # Very strict filters
            self.adaptive_params["atr_multiple_sl"] = 1.5  # Wider stops
            self.adaptive_params["stop_loss_buffer_pct"] = 0.1  # Larger buffer
            self.config["major_timeframes"] = ["4h", "daily", "weekly"]  # Higher timeframes only
            
        elif regime == MarketRegime.BREAKOUT:
            # In breakout regimes, faster reaction
            self.adaptive_params["min_timeframe_agreement"] = 2  # Need fewer alignments
            self.adaptive_params["confirmation_bars"] = 1  # Faster entry
            self.adaptive_params["entry_filter_strength"] = 0.6  # Less strict filters
            self.config["major_timeframes"] = ["30m", "1h", "4h"]  # Medium timeframes
    
    def _perform_strategy_analysis(
        self,
        symbol: str,
        price_data: Dict[str, pd.DataFrame],
        confluence_results: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform analysis for the multi-timeframe confluence strategy.
        
        Args:
            symbol: The trading symbol
            price_data: Dict of price data frames indexed by timeframe
            confluence_results: Results from confluence analysis
            additional_data: Optional additional data
            
        Returns:
            Dictionary with strategy-specific analysis results
        """
        self.logger.info(f"Performing multi-timeframe analysis for {symbol}")
        
        analysis_result = {
            "timeframe_signals": {},
            "confluence_zones": [],
            "directional_bias": "",
            "bias_strength": 0.0,
            "price_action_alignment": {},
            "indicator_alignment": {},
            "momentum_alignment": {},
            "elliott_wave_alignment": {},
            "combined_score": 0.0
        }
        
        try:
            # Get multi-timeframe confluence from the confluence analyzer results
            mtf_confluences = confluence_results.get("multi_timeframe_confluences", [])
            sr_confluences = confluence_results.get("support_resistance_confluence", [])
            
            # Use existing MTF confluence analysis if available
            if mtf_confluences:
                self.logger.info(f"Using {len(mtf_confluences)} multi-timeframe confluences from analyzer")
                
                # Process price level confluences
                price_confluences = [
                    conf for conf in mtf_confluences
                    if conf.get("type") == "support_resistance"
                ]
                
                # Process indicator confluences
                indicator_confluences = [
                    conf for conf in mtf_confluences
                    if conf.get("type") == "indicator"
                ]
                
                # Add existing confluences to results
                analysis_result["confluence_zones"] = price_confluences
                
                # Determine directional bias from indicator confluences
                if indicator_confluences:
                    bullish_count = sum(1 for conf in indicator_confluences if conf.get("bias") == "bullish")
                    bearish_count = sum(1 for conf in indicator_confluences if conf.get("bias") == "bearish")
                    
                    if bullish_count > bearish_count:
                        analysis_result["directional_bias"] = "bullish"
                        analysis_result["bias_strength"] = bullish_count / len(indicator_confluences)
                    elif bearish_count > bullish_count:
                        analysis_result["directional_bias"] = "bearish"
                        analysis_result["bias_strength"] = bearish_count / len(indicator_confluences)
            
            # If no existing MTF analysis or we want to do our own, analyze across timeframes
            if not mtf_confluences or True:  # Always do our own analysis
                # Perform analysis on each timeframe
                timeframe_signals = {}
                
                all_timeframes = set(self.config["major_timeframes"] + 
                                    self.config["minor_timeframes"] + 
                                    [self.config["entry_timeframe"]])
                
                for timeframe in all_timeframes:
                    if timeframe not in price_data:
                        continue
                        
                    df = price_data[timeframe]
                    if df is None or df.empty:
                        continue
                    
                    # Analyze individual timeframe
                    tf_analysis = self._analyze_timeframe(df, timeframe)
                    timeframe_signals[timeframe] = tf_analysis
                
                analysis_result["timeframe_signals"] = timeframe_signals
                
                # Determine cross-timeframe alignments for different aspects
                price_action_alignment = self._analyze_price_action_alignment(timeframe_signals)
                indicator_alignment = self._analyze_indicator_alignment(timeframe_signals)
                momentum_alignment = self._analyze_momentum_alignment(timeframe_signals)
                
                analysis_result["price_action_alignment"] = price_action_alignment
                analysis_result["indicator_alignment"] = indicator_alignment
                analysis_result["momentum_alignment"] = momentum_alignment
                
                # Determine overall directional bias
                directional_bias, bias_strength = self._calculate_directional_bias(
                    price_action_alignment,
                    indicator_alignment,
                    momentum_alignment
                )
                
                analysis_result["directional_bias"] = directional_bias
                analysis_result["bias_strength"] = bias_strength
            
            # Calculate combined score
            combined_score = self._calculate_combined_score(analysis_result)
            analysis_result["combined_score"] = combined_score
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error in multi-timeframe analysis: {str(e)}", exc_info=True)
            return analysis_result
    
    def _analyze_timeframe(
        self,
        price_data: pd.DataFrame,
        timeframe: str
    ) -> Dict[str, Any]:
        """Analyze an individual timeframe for signals and conditions."""
        result = {
            "trend": "neutral",
            "trend_strength": 0.0,
            "support_levels": [],
            "resistance_levels": [],
            "overbought": False,
            "oversold": False,
            "momentum": "neutral",
            "price_action": {
                "candle_patterns": [],
                "recent_pivot": None
            },
            "indicators": {},
            "signals": []
        }
        
        try:
            # Calculate key indicators
            ma_50 = self.technical_indicators.calculate_ma(price_data, period=50)
            ma_200 = self.technical_indicators.calculate_ma(price_data, period=200)
            rsi = self.technical_indicators.calculate_rsi(price_data, period=14)
            macd = self.technical_indicators.calculate_macd(price_data)
            atr = self.technical_indicators.calculate_atr(price_data, period=14)
            stoch = self.technical_indicators.calculate_stochastic(price_data)
            
            # Store indicator values
            result["indicators"] = {
                "ma_50": ma_50.iloc[-1],
                "ma_200": ma_200.iloc[-1],
                "rsi": rsi.iloc[-1],
                "macd_histogram": macd["histogram"].iloc[-1],
                "macd_signal": macd["signal"].iloc[-1],
                "macd_macd": macd["macd"].iloc[-1],
                "atr": atr.iloc[-1],
                "stoch_k": stoch["k"].iloc[-1],
                "stoch_d": stoch["d"].iloc[-1]
            }
            
            # Determine trend
            close = price_data["close"].iloc[-1]
            if close > ma_50.iloc[-1] and ma_50.iloc[-1] > ma_200.iloc[-1]:
                result["trend"] = "bullish"
                # Calculate trend strength
                trend_strength = min(1.0, (close - ma_50.iloc[-1]) / (ma_50.iloc[-1] * 0.01))
                result["trend_strength"] = trend_strength
            elif close < ma_50.iloc[-1] and ma_50.iloc[-1] < ma_200.iloc[-1]:
                result["trend"] = "bearish"
                # Calculate trend strength
                trend_strength = min(1.0, (ma_50.iloc[-1] - close) / (ma_50.iloc[-1] * 0.01))
                result["trend_strength"] = trend_strength
            
            # Check overbought/oversold conditions
            result["overbought"] = rsi.iloc[-1] > 70 or stoch["k"].iloc[-1] > 80
            result["oversold"] = rsi.iloc[-1] < 30 or stoch["k"].iloc[-1] < 20
            
            # Determine momentum
            if macd["histogram"].iloc[-1] > 0 and macd["histogram"].iloc[-1] > macd["histogram"].iloc[-2]:
                result["momentum"] = "bullish"
            elif macd["histogram"].iloc[-1] < 0 and macd["histogram"].iloc[-1] < macd["histogram"].iloc[-2]:
                result["momentum"] = "bearish"
            
            # Generate signals for this timeframe
            signals = []
            
            # Trend signal
            if result["trend"] == "bullish" and result["trend_strength"] > 0.3:
                signals.append({
                    "type": "trend",
                    "direction": "buy",
                    "strength": result["trend_strength"],
                    "desc": f"Bullish trend on {timeframe}"
                })
            elif result["trend"] == "bearish" and result["trend_strength"] > 0.3:
                signals.append({
                    "type": "trend",
                    "direction": "sell",
                    "strength": result["trend_strength"],
                    "desc": f"Bearish trend on {timeframe}"
                })
            
            # Momentum signal
            if result["momentum"] == "bullish":
                signals.append({
                    "type": "momentum",
                    "direction": "buy",
                    "strength": min(1.0, abs(macd["histogram"].iloc[-1] / 0.001)),
                    "desc": f"Bullish momentum on {timeframe}"
                })
            elif result["momentum"] == "bearish":
                signals.append({
                    "type": "momentum",
                    "direction": "sell",
                    "strength": min(1.0, abs(macd["histogram"].iloc[-1] / 0.001)),
                    "desc": f"Bearish momentum on {timeframe}"
                })
            
            # Oversold/Overbought reversal signals
            if result["oversold"] and result["momentum"] == "bullish":
                signals.append({
                    "type": "reversal",
                    "direction": "buy",
                    "strength": 0.7,
                    "desc": f"Oversold reversal on {timeframe}"
                })
            elif result["overbought"] and result["momentum"] == "bearish":
                signals.append({
                    "type": "reversal",
                    "direction": "sell",
                    "strength": 0.7,
                    "desc": f"Overbought reversal on {timeframe}"
                })
            
            # Store signals
            result["signals"] = signals
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing timeframe {timeframe}: {str(e)}")
            return result
    
    def _analyze_price_action_alignment(
        self,
        timeframe_signals: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze how price action aligns across timeframes."""
        result = {
            "aligned_direction": "neutral",
            "agreement_strength": 0.0,
            "supporting_timeframes": [],
            "details": {}
        }
        
        try:
            bullish_signals = []
            bearish_signals = []
            
            # Collect trend signals from each timeframe
            for tf, signals in timeframe_signals.items():
                if signals["trend"] == "bullish":
                    weight = self.timeframe_weights.get(tf, 0.5) * signals["trend_strength"]
                    bullish_signals.append({
                        "timeframe": tf,
                        "weight": weight,
                        "strength": signals["trend_strength"]
                    })
                elif signals["trend"] == "bearish":
                    weight = self.timeframe_weights.get(tf, 0.5) * signals["trend_strength"]
                    bearish_signals.append({
                        "timeframe": tf,
                        "weight": weight,
                        "strength": signals["trend_strength"]
                    })
            
            # Calculate total weight for each direction
            bullish_weight = sum(signal["weight"] for signal in bullish_signals)
            bearish_weight = sum(signal["weight"] for signal in bearish_signals)
            
            # Determine aligned direction
            if bullish_weight > bearish_weight and bullish_weight > 1.0:
                result["aligned_direction"] = "bullish"
                result["agreement_strength"] = min(1.0, bullish_weight / max(bullish_weight + bearish_weight, 1.0))
                result["supporting_timeframes"] = [signal["timeframe"] for signal in bullish_signals]
                
            elif bearish_weight > bullish_weight and bearish_weight > 1.0:
                result["aligned_direction"] = "bearish"
                result["agreement_strength"] = min(1.0, bearish_weight / max(bullish_weight + bearish_weight, 1.0))
                result["supporting_timeframes"] = [signal["timeframe"] for signal in bearish_signals]
            
            # Store details
            result["details"] = {
                "bullish_signals": bullish_signals,
                "bearish_signals": bearish_signals,
                "bullish_weight": bullish_weight,
                "bearish_weight": bearish_weight
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing price action alignment: {str(e)}")
            return result
    
    def _analyze_indicator_alignment(
        self,
        timeframe_signals: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze how technical indicators align across timeframes."""
        result = {
            "aligned_direction": "neutral",
            "agreement_strength": 0.0,
            "supporting_timeframes": [],
            "details": {}
        }
        
        try:
            bullish_indicators = []
            bearish_indicators = []
            
            # Check indicators across timeframes
            for tf, signals in timeframe_signals.items():
                weight = self.timeframe_weights.get(tf, 0.5)
                
                # RSI
                rsi = signals.get("indicators", {}).get("rsi", 50)
                if rsi < 30:
                    bullish_indicators.append({
                        "timeframe": tf,
                        "indicator": "rsi",
                        "value": rsi,
                        "weight": weight * 0.6
                    })
                elif rsi > 70:
                    bearish_indicators.append({
                        "timeframe": tf,
                        "indicator": "rsi",
                        "value": rsi,
                        "weight": weight * 0.6
                    })
                
                # MACD
                macd_hist = signals.get("indicators", {}).get("macd_histogram", 0)
                if macd_hist > 0:
                    bullish_indicators.append({
                        "timeframe": tf,
                        "indicator": "macd",
                        "value": macd_hist,
                        "weight": weight * 0.7
                    })
                elif macd_hist < 0:
                    bearish_indicators.append({
                        "timeframe": tf,
                        "indicator": "macd",
                        "value": macd_hist,
                        "weight": weight * 0.7
                    })
                
                # Stochastic
                stoch_k = signals.get("indicators", {}).get("stoch_k", 50)
                stoch_d = signals.get("indicators", {}).get("stoch_d", 50)
                
                if stoch_k < 20 and stoch_k > stoch_d:
                    bullish_indicators.append({
                        "timeframe": tf,
                        "indicator": "stochastic",
                        "value": stoch_k,
                        "weight": weight * 0.5
                    })
                elif stoch_k > 80 and stoch_k < stoch_d:
                    bearish_indicators.append({
                        "timeframe": tf,
                        "indicator": "stochastic",
                        "value": stoch_k,
                        "weight": weight * 0.5
                    })
            
            # Calculate total weight for each direction
            bullish_weight = sum(ind["weight"] for ind in bullish_indicators)
            bearish_weight = sum(ind["weight"] for ind in bearish_indicators)
            
            # Determine aligned direction
            if bullish_weight > bearish_weight and bullish_weight > 1.0:
                result["aligned_direction"] = "bullish"
                result["agreement_strength"] = min(1.0, bullish_weight / max(bullish_weight + bearish_weight, 1.0))
                result["supporting_timeframes"] = list(set(ind["timeframe"] for ind in bullish_indicators))
                
            elif bearish_weight > bullish_weight and bearish_weight > 1.0:
                result["aligned_direction"] = "bearish"
                result["agreement_strength"] = min(1.0, bearish_weight / max(bullish_weight + bearish_weight, 1.0))
                result["supporting_timeframes"] = list(set(ind["timeframe"] for ind in bearish_indicators))
            
            # Store details
            result["details"] = {
                "bullish_indicators": bullish_indicators,
                "bearish_indicators": bearish_indicators,
                "bullish_weight": bullish_weight,
                "bearish_weight": bearish_weight
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing indicator alignment: {str(e)}")
            return result
    
    def _analyze_momentum_alignment(
        self,
        timeframe_signals: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze how momentum aligns across timeframes."""
        result = {
            "aligned_direction": "neutral",
            "agreement_strength": 0.0,
            "supporting_timeframes": [],
            "details": {}
        }
        
        try:
            bullish_momentum = []
            bearish_momentum = []
            
            # Check momentum across timeframes
            for tf, signals in timeframe_signals.items():
                momentum = signals.get("momentum", "neutral")
                weight = self.timeframe_weights.get(tf, 0.5)
                
                if momentum == "bullish":
                    bullish_momentum.append({
                        "timeframe": tf,
                        "weight": weight
                    })
                elif momentum == "bearish":
                    bearish_momentum.append({
                        "timeframe": tf,
                        "weight": weight
                    })
            
            # Calculate total weight for each direction
            bullish_weight = sum(m["weight"] for m in bullish_momentum)
            bearish_weight = sum(m["weight"] for m in bearish_momentum)
            
            # Determine aligned direction
            if bullish_weight > bearish_weight and bullish_weight > 1.0:
                result["aligned_direction"] = "bullish"
                result["agreement_strength"] = min(1.0, bullish_weight / max(bullish_weight + bearish_weight, 1.0))
                result["supporting_timeframes"] = [m["timeframe"] for m in bullish_momentum]
                
            elif bearish_weight > bullish_weight and bearish_weight > 1.0:
                result["aligned_direction"] = "bearish"
                result["agreement_strength"] = min(1.0, bearish_weight / max(bullish_weight + bearish_weight, 1.0))
                result["supporting_timeframes"] = [m["timeframe"] for m in bearish_momentum]
            
            # Store details
            result["details"] = {
                "bullish_momentum": bullish_momentum,
                "bearish_momentum": bearish_momentum,
                "bullish_weight": bullish_weight,
                "bearish_weight": bearish_weight
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing momentum alignment: {str(e)}")
            return result
    
    def _calculate_directional_bias(
        self,
        price_action_alignment: Dict[str, Any],
        indicator_alignment: Dict[str, Any],
        momentum_alignment: Dict[str, Any]
    ) -> Tuple[str, float]:
        """Calculate overall directional bias from different alignment types."""
        # Get directions
        pa_direction = price_action_alignment.get("aligned_direction", "neutral")
        ind_direction = indicator_alignment.get("aligned_direction", "neutral")
        mom_direction = momentum_alignment.get("aligned_direction", "neutral")
        
        # Count agreement
        directions = [pa_direction, ind_direction, mom_direction]
        bullish_count = directions.count("bullish")
        bearish_count = directions.count("bearish")
        
        # Calculate weights
        pa_weight = price_action_alignment.get("agreement_strength", 0) * 0.4
        ind_weight = indicator_alignment.get("agreement_strength", 0) * 0.35
        mom_weight = momentum_alignment.get("agreement_strength", 0) * 0.25
        
        bullish_weight = (pa_weight if pa_direction == "bullish" else 0) + \
                        (ind_weight if ind_direction == "bullish" else 0) + \
                        (mom_weight if mom_direction == "bullish" else 0)
        
        bearish_weight = (pa_weight if pa_direction == "bearish" else 0) + \
                        (ind_weight if ind_direction == "bearish" else 0) + \
                        (mom_weight if mom_direction == "bearish" else 0)
        
        # Determine overall bias
        if bullish_count > bearish_count and bullish_weight > 0.3:
            bias = "bullish"
            strength = bullish_weight
        elif bearish_count > bullish_count and bearish_weight > 0.3:
            bias = "bearish"
            strength = bearish_weight
        else:
            bias = "neutral"
            strength = 0.0
            
        return bias, strength
    
    def _calculate_combined_score(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate a combined score for the multi-timeframe analysis."""
        score = 0.0
        
        # Directional bias contribution
        bias_strength = analysis_result.get("bias_strength", 0)
        score += bias_strength * 0.4  # 40% weight
        
        # Price action alignment contribution
        pa_strength = analysis_result.get("price_action_alignment", {}).get("agreement_strength", 0)
        pa_tf_count = len(analysis_result.get("price_action_alignment", {}).get("supporting_timeframes", []))
        
        if pa_tf_count >= self.adaptive_params["min_timeframe_agreement"]:
            score += pa_strength * 0.2  # 20% weight
        
        # Indicator alignment contribution
        ind_strength = analysis_result.get("indicator_alignment", {}).get("agreement_strength", 0)
        ind_tf_count = len(analysis_result.get("indicator_alignment", {}).get("supporting_timeframes", []))
        
        if ind_tf_count >= self.adaptive_params["min_timeframe_agreement"]:
            score += ind_strength * 0.2  # 20% weight
        
        # Momentum alignment contribution
        mom_strength = analysis_result.get("momentum_alignment", {}).get("agreement_strength", 0)
        mom_tf_count = len(analysis_result.get("momentum_alignment", {}).get("supporting_timeframes", []))
        
        if mom_tf_count >= self.adaptive_params["min_timeframe_agreement"]:
            score += mom_strength * 0.2  # 20% weight
        
        # Confluence zones contribution
        confluence_count = len(analysis_result.get("confluence_zones", []))
        if confluence_count > 0:
            zone_score = min(1.0, confluence_count / 3)  # Cap at 3 zones
            score += zone_score * 0.2  # 20% weight
        
        return min(1.0, score)  # Cap at 1.0
    
    def _generate_signals(
        self,
        symbol: str,
        strategy_analysis: Dict[str, Any],
        confluence_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate trading signals based on strategy analysis."""
        signals = []
        
        try:
            # Get directional bias
            bias = strategy_analysis.get("directional_bias", "neutral")
            bias_strength = strategy_analysis.get("bias_strength", 0)
            combined_score = strategy_analysis.get("combined_score", 0)
            
            # Skip if no clear bias or insufficient score
            if bias == "neutral" or bias_strength < 0.5:
                return []
                
            if combined_score < self.config["minimum_confluence_score"]:
                return []
            
            # Get entry timeframe data
            entry_tf = self.config["entry_timeframe"]
            entry_signals = strategy_analysis.get("timeframe_signals", {}).get(entry_tf, {})
            
            # Get the last price for entry calculation
            if not entry_signals:
                return []
            
            # Create the signal
            if bias == "bullish":
                # Get potential entry point
                entry_tf_data = strategy_analysis.get("timeframe_signals", {}).get(entry_tf, {})
                atr = entry_tf_data.get("indicators", {}).get("atr", 0)
                
                # Use entry price from entry timeframe
                entry_price = entry_tf_data.get("indicators", {}).get("ma_50", None)
                if entry_price is None:
                    return []
                
                # Calculate stop loss and take profit
                stop_loss = entry_price - (atr * self.adaptive_params["atr_multiple_sl"])
                take_profit = entry_price + (atr * self.adaptive_params["atr_multiple_tp"])
                
                # Create buy signal
                signals.append({
                    "symbol": symbol,
                    "strategy": self.name,
                    "direction": "buy",
                    "type": "multi_timeframe_confluence",
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "timeframe": entry_tf,
                    "confidence": combined_score,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {
                        "bias_strength": bias_strength,
                        "supporting_timeframes": {
                            "price_action": strategy_analysis.get("price_action_alignment", {}).get("supporting_timeframes", []),
                            "indicators": strategy_analysis.get("indicator_alignment", {}).get("supporting_timeframes", []),
                            "momentum": strategy_analysis.get("momentum_alignment", {}).get("supporting_timeframes", [])
                        },
                        "confluence_zones": len(strategy_analysis.get("confluence_zones", [])),
                    }
                })
                
            elif bias == "bearish":
                # Get potential entry point
                entry_tf_data = strategy_analysis.get("timeframe_signals", {}).get(entry_tf, {})
                atr = entry_tf_data.get("indicators", {}).get("atr", 0)
                
                # Use entry price from entry timeframe
                entry_price = entry_tf_data.get("indicators", {}).get("ma_50", None)
                if entry_price is None:
                    return []
                
                # Calculate stop loss and take profit
                stop_loss = entry_price + (atr * self.adaptive_params["atr_multiple_sl"])
                take_profit = entry_price - (atr * self.adaptive_params["atr_multiple_tp"])
                
                # Create sell signal
                signals.append({
                    "symbol": symbol,
                    "strategy": self.name,
                    "direction": "sell",
                    "type": "multi_timeframe_confluence",
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "timeframe": entry_tf,
                    "confidence": combined_score,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {
                        "bias_strength": bias_strength,
                        "supporting_timeframes": {
                            "price_action": strategy_analysis.get("price_action_alignment", {}).get("supporting_timeframes", []),
                            "indicators": strategy_analysis.get("indicator_alignment", {}).get("supporting_timeframes", []),
                            "momentum": strategy_analysis.get("momentum_alignment", {}).get("supporting_timeframes", [])
                        },
                        "confluence_zones": len(strategy_analysis.get("confluence_zones", [])),
                    }
                })
            
            # For each signal, calculate reward-risk ratio
            for signal in signals:
                if signal["direction"] == "buy":
                    risk = signal["entry_price"] - signal["stop_loss"]
                    reward = signal["take_profit"] - signal["entry_price"]
                else:
                    risk = signal["stop_loss"] - signal["entry_price"]
                    reward = signal["entry_price"] - signal["take_profit"]
                
                if risk > 0:
                    signal["reward_risk_ratio"] = reward / risk
                else:
                    signal["reward_risk_ratio"] = 0
                    
                # Apply position sizing based on confidence if enabled
                if self.config["enable_partial_position_sizing"]:
                    signal["position_size_factor"] = combined_score
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}", exc_info=True)
            
        return signals
