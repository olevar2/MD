"""
AdvancedBreakoutStrategy

This module implements a breakout trading strategy using Fibonacci levels and pivot points,
with confluence-based confirmation and adaptation to market regimes.

Part of Phase 4 implementation to enhance the adaptive trading capabilities.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

from strategy_execution_engine.strategies.advanced_ta_strategy import AdvancedTAStrategy
from analysis_engine.services.tool_effectiveness import MarketRegime
from analysis_engine.analysis.technical_indicators import TechnicalIndicators
from analysis_engine.analysis.fibonacci_tools import FibonacciTools
from analysis_engine.analysis.pivot_points import PivotPointCalculator
from strategy_execution_engine.clients.feature_store_client import FeatureStoreClient


class AdvancedBreakoutStrategy(AdvancedTAStrategy):
    """
    Advanced breakout trading strategy using Fibonacci levels and pivot points
    with confluence-based confirmation and adaptation to market regimes.
    """

    def _init_strategy_config(self) -> None:
        """Initialize strategy-specific configuration parameters."""
        # Initialize required services
        self.fibonacci_tools = FibonacciTools()
        self.pivot_calculator = PivotPointCalculator()

        # Initialize feature store client
        self.feature_store_client = FeatureStoreClient(
            use_cache=True,
            cache_ttl=300  # 5 minutes cache TTL
        )

        # Strategy-specific parameters
        self.adaptive_params = {
            "breakout_confirmation_bars": 2,  # Number of bars to confirm breakout
            "min_breakout_size_pips": 20,     # Minimum breakout size in pips
            "breakout_volume_factor": 1.5,    # Volume increase required for breakout
            "fib_levels": [0.382, 0.5, 0.618, 0.786, 1.0, 1.27, 1.618],  # Fib levels to use
            "atr_multiple_entry": 1.0,        # ATR multiple for entry
            "atr_multiple_sl": 1.5,           # ATR multiple for stop loss
            "atr_multiple_tp": 2.5            # ATR multiple for take profit
        }

        # Configuration specific to breakout strategies
        self.config.update({
            "pivot_point_method": "traditional",  # traditional, woodie, camarilla, etc.
            "breakout_types": ["support_resistance", "range", "channel", "fib_level"],
            "use_volume_confirmation": True,
            "filter_by_volatility": True,
            "min_reward_risk_ratio": 1.5,
            "max_daily_signals": 2,
            "prefer_aligned_with_trend": True,
            "feature_store_indicators": [
                "sma_50", "sma_200", "atr_14", "adx_14"  # Indicators to fetch from feature store
            ]
        })

        self.logger.info(f"Initialized {self.name} with breakout parameters")

    def _adapt_parameters_to_regime(self, regime: MarketRegime) -> None:
        """Adjust strategy parameters based on the current market regime."""
        self.logger.info(f"Adapting parameters to {regime} regime")

        if regime == MarketRegime.TRENDING:
            # In trending markets, focus on trend continuation breakouts
            self.adaptive_params["breakout_confirmation_bars"] = 1
            self.adaptive_params["breakout_volume_factor"] = 1.3
            self.adaptive_params["atr_multiple_tp"] = 3.0  # More aggressive take profit
            self.config["prefer_aligned_with_trend"] = True

        elif regime == MarketRegime.RANGING:
            # In ranging markets, more conservative on breakouts
            self.adaptive_params["breakout_confirmation_bars"] = 3  # Need more confirmation
            self.adaptive_params["min_breakout_size_pips"] = 30  # Larger breakout required
            self.adaptive_params["breakout_volume_factor"] = 2.0  # Need more volume
            self.adaptive_params["atr_multiple_tp"] = 1.5  # Less aggressive take profit
            self.config["prefer_aligned_with_trend"] = False

        elif regime == MarketRegime.VOLATILE:
            # In volatile markets, very selective on breakouts
            self.adaptive_params["breakout_confirmation_bars"] = 2
            self.adaptive_params["min_breakout_size_pips"] = 40
            self.adaptive_params["breakout_volume_factor"] = 2.5
            self.adaptive_params["atr_multiple_sl"] = 2.0  # Wider stop loss
            self.config["min_reward_risk_ratio"] = 2.0  # Require better R:R
            self.config["max_daily_signals"] = 1  # Limit signals in volatile markets

    async def _perform_strategy_analysis(
        self,
        symbol: str,
        price_data: Dict[str, pd.DataFrame],
        confluence_results: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform analysis for the breakout strategy.

        Args:
            symbol: The trading symbol
            price_data: Dict of price data frames indexed by timeframe
            confluence_results: Results from confluence analysis
            additional_data: Optional additional data

        Returns:
            Dictionary with strategy-specific analysis results
        """
        self.logger.info(f"Performing breakout analysis for {symbol}")

        analysis_result = {
            "breakout_opportunities": [],
            "pivot_levels": {},
            "fibonacci_levels": {},
            "detected_ranges": [],
            "support_resistance_breakouts": []
        }

        try:
            # Get price data for all timeframes
            for timeframe, df in price_data.items():
                if df is None or df.empty:
                    continue

                # Get symbol from price data
                symbol = df.get('symbol', symbol)
                if isinstance(symbol, pd.Series):
                    symbol = symbol.iloc[0]

                # Get start and end dates
                start_date = df.index[0] if not df.empty else datetime.now() - timedelta(days=30)
                end_date = df.index[-1] if not df.empty else datetime.now()

                # Try to get ATR from feature store
                try:
                    atr_data = await self.feature_store_client.get_indicators(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        indicators=["atr_14"]
                    )

                    if not atr_data.empty and "atr_14" in atr_data.columns:
                        atr = atr_data["atr_14"].iloc[-1]
                    else:
                        # Fall back to direct calculation
                        atr = self.technical_indicators.calculate_atr(df, period=14).iloc[-1]
                except Exception as e:
                    self.logger.warning(f"Error getting ATR from feature store: {str(e)}, falling back to direct calculation")
                    # Fall back to direct calculation
                    atr = self.technical_indicators.calculate_atr(df, period=14).iloc[-1]

                # Calculate pivot points
                pivot_points = self.pivot_calculator.calculate_pivot_points(
                    df, method=self.config["pivot_point_method"]
                )
                analysis_result["pivot_levels"][timeframe] = pivot_points

                # Calculate Fibonacci levels from recent swings
                fib_levels = self._calculate_fibonacci_levels(df)
                analysis_result["fibonacci_levels"][timeframe] = fib_levels

                # Detect potential breakout candidates
                range_breakouts = self._detect_range_breakouts(df, atr)
                sr_breakouts = self._detect_sr_breakouts(
                    df, confluence_results.get("support_resistance_confluence", []), atr
                )
                fib_breakouts = self._detect_fibonacci_breakouts(df, fib_levels, atr)

                # Compile all breakout opportunities
                timeframe_breakouts = range_breakouts + sr_breakouts + fib_breakouts

                # Add additional metadata
                for breakout in timeframe_breakouts:
                    breakout["timeframe"] = timeframe
                    breakout["atr"] = atr

                    # Check if breakout aligns with trend
                    trend = self._determine_trend(df)
                    breakout["aligned_with_trend"] = (
                        (trend == "bullish" and breakout["direction"] == "buy") or
                        (trend == "bearish" and breakout["direction"] == "sell")
                    )

                    # Check for volume confirmation if available
                    if "volume" in df.columns and self.config["use_volume_confirmation"]:
                        avg_volume = df["volume"].rolling(10).mean().iloc[-1]
                        recent_volume = df["volume"].iloc[-1]
                        breakout["volume_confirmed"] = (
                            recent_volume >= avg_volume * self.adaptive_params["breakout_volume_factor"]
                        )
                    else:
                        breakout["volume_confirmed"] = True

                analysis_result["breakout_opportunities"].extend(timeframe_breakouts)

            # Sort breakouts by score
            analysis_result["breakout_opportunities"] = sorted(
                analysis_result["breakout_opportunities"],
                key=lambda x: x.get("score", 0),
                reverse=True
            )

            return analysis_result

        except Exception as e:
            self.logger.error(f"Error in breakout analysis: {str(e)}", exc_info=True)
            return analysis_result

    def _calculate_fibonacci_levels(self, price_data: pd.DataFrame) -> Dict[str, List[float]]:
        """Calculate Fibonacci levels from recent swing highs and lows."""
        try:
            # Find recent significant swing high and low
            high = price_data["high"].rolling(10).max().iloc[-20]
            low = price_data["low"].rolling(10).min().iloc[-20]

            # Calculate retracement levels (uptrend)
            up_levels = self.fibonacci_tools.calculate_retracement_levels(
                low, high, levels=self.adaptive_params["fib_levels"]
            )

            # Calculate extension levels (uptrend)
            up_extensions = self.fibonacci_tools.calculate_extension_levels(
                low, high, low, levels=self.adaptive_params["fib_levels"][4:]  # Use only extension levels
            )

            # Calculate retracement levels (downtrend)
            down_levels = self.fibonacci_tools.calculate_retracement_levels(
                high, low, levels=self.adaptive_params["fib_levels"]
            )

            # Calculate extension levels (downtrend)
            down_extensions = self.fibonacci_tools.calculate_extension_levels(
                high, low, high, levels=self.adaptive_params["fib_levels"][4:]  # Use only extension levels
            )

            return {
                "uptrend_retracement": up_levels,
                "uptrend_extension": up_extensions,
                "downtrend_retracement": down_levels,
                "downtrend_extension": down_extensions
            }

        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci levels: {str(e)}")
            return {"uptrend_retracement": [], "uptrend_extension": [],
                   "downtrend_retracement": [], "downtrend_extension": []}

    def _detect_range_breakouts(
        self,
        price_data: pd.DataFrame,
        atr: float
    ) -> List[Dict[str, Any]]:
        """Detect potential range breakouts."""
        breakouts = []

        try:
            # Calculate recent price range
            lookback = 20  # bars to look back
            if len(price_data) < lookback:
                return breakouts

            recent_data = price_data.iloc[-lookback:]
            range_high = recent_data["high"].max()
            range_low = recent_data["low"].min()
            range_size = range_high - range_low

            # Check if we have sufficient price action to determine a range
            if range_size < atr * 2:
                return breakouts  # Range too small to be meaningful

            # Get current close and previous close
            current_close = price_data["close"].iloc[-1]
            previous_close = price_data["close"].iloc[-2]

            # Check for bullish breakout
            if (previous_close < range_high and
                current_close > range_high and
                (current_close - range_high) > self.adaptive_params["min_breakout_size_pips"] * 0.0001):

                # Calculate target based on range size projection
                target = current_close + range_size
                stop_loss = range_high - (atr * self.adaptive_params["atr_multiple_sl"])

                breakouts.append({
                    "type": "range_breakout",
                    "direction": "buy",
                    "entry_price": current_close,
                    "stop_loss": stop_loss,
                    "target": target,
                    "score": self._calculate_breakout_score(
                        direction="bullish",
                        price_data=price_data,
                        breakout_size=(current_close - range_high) / atr
                    ),
                    "range_details": {
                        "high": range_high,
                        "low": range_low,
                        "size": range_size,
                        "duration_bars": lookback
                    }
                })

            # Check for bearish breakout
            if (previous_close > range_low and
                current_close < range_low and
                (range_low - current_close) > self.adaptive_params["min_breakout_size_pips"] * 0.0001):

                # Calculate target based on range size projection
                target = current_close - range_size
                stop_loss = range_low + (atr * self.adaptive_params["atr_multiple_sl"])

                breakouts.append({
                    "type": "range_breakout",
                    "direction": "sell",
                    "entry_price": current_close,
                    "stop_loss": stop_loss,
                    "target": target,
                    "score": self._calculate_breakout_score(
                        direction="bearish",
                        price_data=price_data,
                        breakout_size=(range_low - current_close) / atr
                    ),
                    "range_details": {
                        "high": range_high,
                        "low": range_low,
                        "size": range_size,
                        "duration_bars": lookback
                    }
                })

            return breakouts

        except Exception as e:
            self.logger.error(f"Error detecting range breakouts: {str(e)}")
            return []

    def _detect_sr_breakouts(
        self,
        price_data: pd.DataFrame,
        sr_zones: List[Dict[str, Any]],
        atr: float
    ) -> List[Dict[str, Any]]:
        """Detect potential breakouts of support/resistance zones."""
        breakouts = []

        try:
            if not sr_zones:
                return breakouts

            current_close = price_data["close"].iloc[-1]
            previous_close = price_data["close"].iloc[-2]

            for zone in sr_zones:
                zone_price = zone.get("price", 0)
                zone_strength = zone.get("strength", 0)

                # Skip weak zones
                if zone_strength < 0.5:
                    continue

                # Check for bullish breakout of resistance
                if (previous_close < zone_price and
                    current_close > zone_price and
                    (current_close - zone_price) > self.adaptive_params["min_breakout_size_pips"] * 0.0001):

                    stop_loss = zone_price - (atr * self.adaptive_params["atr_multiple_sl"])
                    target = current_close + (atr * self.adaptive_params["atr_multiple_tp"])

                    breakouts.append({
                        "type": "support_resistance_breakout",
                        "direction": "buy",
                        "entry_price": current_close,
                        "stop_loss": stop_loss,
                        "target": target,
                        "zone_price": zone_price,
                        "zone_strength": zone_strength,
                        "score": zone_strength * 0.7 + 0.3
                    })

                # Check for bearish breakout of support
                if (previous_close > zone_price and
                    current_close < zone_price and
                    (zone_price - current_close) > self.adaptive_params["min_breakout_size_pips"] * 0.0001):

                    stop_loss = zone_price + (atr * self.adaptive_params["atr_multiple_sl"])
                    target = current_close - (atr * self.adaptive_params["atr_multiple_tp"])

                    breakouts.append({
                        "type": "support_resistance_breakout",
                        "direction": "sell",
                        "entry_price": current_close,
                        "stop_loss": stop_loss,
                        "target": target,
                        "zone_price": zone_price,
                        "zone_strength": zone_strength,
                        "score": zone_strength * 0.7 + 0.3
                    })

            return breakouts

        except Exception as e:
            self.logger.error(f"Error detecting S/R breakouts: {str(e)}")
            return []

    def _detect_fibonacci_breakouts(
        self,
        price_data: pd.DataFrame,
        fib_levels: Dict[str, List[float]],
        atr: float
    ) -> List[Dict[str, Any]]:
        """Detect potential breakouts of Fibonacci levels."""
        breakouts = []

        try:
            if not fib_levels:
                return breakouts

            current_close = price_data["close"].iloc[-1]
            previous_close = price_data["close"].iloc[-2]

            # Check uptrend retracement levels
            for level, price in fib_levels.get("uptrend_retracement", {}).items():
                # Check for bullish bounce
                if (previous_close <= price and current_close > price and
                    abs(previous_close - price) < atr * 0.5):

                    stop_loss = price - (atr * self.adaptive_params["atr_multiple_sl"])
                    target = current_close + (atr * self.adaptive_params["atr_multiple_tp"])

                    breakouts.append({
                        "type": "fibonacci_bounce",
                        "direction": "buy",
                        "entry_price": current_close,
                        "stop_loss": stop_loss,
                        "target": target,
                        "fib_level": level,
                        "fib_price": price,
                        "score": 0.6  # Base score for Fibonacci bounce
                    })

            # Check downtrend retracement levels
            for level, price in fib_levels.get("downtrend_retracement", {}).items():
                # Check for bearish bounce
                if (previous_close >= price and current_close < price and
                    abs(previous_close - price) < atr * 0.5):

                    stop_loss = price + (atr * self.adaptive_params["atr_multiple_sl"])
                    target = current_close - (atr * self.adaptive_params["atr_multiple_tp"])

                    breakouts.append({
                        "type": "fibonacci_bounce",
                        "direction": "sell",
                        "entry_price": current_close,
                        "stop_loss": stop_loss,
                        "target": target,
                        "fib_level": level,
                        "fib_price": price,
                        "score": 0.6  # Base score for Fibonacci bounce
                    })

            return breakouts

        except Exception as e:
            self.logger.error(f"Error detecting Fibonacci breakouts: {str(e)}")
            return []

    async def _calculate_breakout_score(self, direction: str, price_data: pd.DataFrame, breakout_size: float) -> float:
        """Calculate a score for the detected breakout based on various factors."""
        # Start with a base score
        score = 0.5

        # Adjust based on breakout size relative to ATR
        if breakout_size >= 2.0:
            score += 0.2
        elif breakout_size >= 1.0:
            score += 0.1

        try:
            # Check if breakout is aligned with larger trend
            trend = await self._determine_trend(price_data)
            if (direction == "bullish" and trend == "bullish") or (direction == "bearish" and trend == "bearish"):
                score += 0.15

            # Get symbol and timeframe from price data
            symbol = price_data.get('symbol', 'unknown')
            if isinstance(symbol, pd.Series):
                symbol = symbol.iloc[0]

            # Get start and end dates from price data
            start_date = price_data.index[0] if not price_data.empty else datetime.now() - timedelta(days=30)
            end_date = price_data.index[-1] if not price_data.empty else datetime.now()

            # Get ATR from feature store
            atr_data = await self.feature_store_client.get_indicators(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                indicators=["atr_14"]
            )

            if not atr_data.empty and "atr_14" in atr_data.columns:
                # Calculate recent and historical ATR
                recent_atr = atr_data["atr_14"].iloc[-20:].mean()
                historical_atr = atr_data["atr_14"].iloc[-50:-20].mean() if len(atr_data) >= 50 else recent_atr

                if recent_atr > historical_atr * 1.3:  # Volatility expansion
                    score += 0.1
            else:
                # Fall back to direct calculation
                recent_atr = self.technical_indicators.calculate_atr(price_data.iloc[-20:], period=14).iloc[-1]
                historical_atr = self.technical_indicators.calculate_atr(price_data.iloc[-50:-20], period=14).iloc[-1]

                if recent_atr > historical_atr * 1.3:  # Volatility expansion
                    score += 0.1
        except Exception as e:
            self.logger.error(f"Error calculating breakout score: {str(e)}")

        # Cap score at 1.0
        return min(1.0, score)

    async def _determine_trend(self, price_data: pd.DataFrame) -> str:
        """Determine the current market trend using indicators from the feature store."""
        try:
            # Get symbol and timeframe from price data
            symbol = price_data.get('symbol', 'unknown')
            if isinstance(symbol, pd.Series):
                symbol = symbol.iloc[0]

            # Get start and end dates from price data
            start_date = price_data.index[0] if not price_data.empty else datetime.now() - timedelta(days=30)
            end_date = price_data.index[-1] if not price_data.empty else datetime.now()

            # Get indicators from feature store
            indicators = await self.feature_store_client.get_indicators(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                indicators=["sma_50", "sma_200", "adx_14"]
            )

            # If feature store request failed, fall back to direct calculation
            if indicators.empty:
                self.logger.warning("Failed to get indicators from feature store, falling back to direct calculation")
                ma_50 = self.technical_indicators.calculate_ma(price_data, period=50).iloc[-1]
                ma_200 = self.technical_indicators.calculate_ma(price_data, period=200).iloc[-1]
                adx = self.technical_indicators.calculate_adx(price_data).iloc[-1]
            else:
                # Get the latest values
                ma_50 = indicators["sma_50"].iloc[-1]
                ma_200 = indicators["sma_200"].iloc[-1]
                adx = indicators["adx_14"].iloc[-1]

            current_close = price_data["close"].iloc[-1]

            # Determine trend
            if current_close > ma_50 and ma_50 > ma_200 and adx > 25:
                return "bullish"
            elif current_close < ma_50 and ma_50 < ma_200 and adx > 25:
                return "bearish"
            else:
                return "neutral"

        except Exception as e:
            self.logger.error(f"Error determining trend: {str(e)}")
            return "neutral"  # Default to neutral on error

    async def _generate_signals(
        self,
        symbol: str,
        strategy_analysis: Dict[str, Any],
        confluence_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate trading signals based on strategy analysis."""
        signals = []

        try:
            breakout_opportunities = strategy_analysis.get("breakout_opportunities", [])

            for breakout in breakout_opportunities:
                # Skip opportunities without sufficient score
                if breakout.get("score", 0) < 0.7:
                    continue

                # Skip if not aligned with trend and we prefer aligned signals
                if (self.config["prefer_aligned_with_trend"] and
                    not breakout.get("aligned_with_trend", False)):
                    continue

                # Skip if no volume confirmation and we require it
                if (self.config["use_volume_confirmation"] and
                    not breakout.get("volume_confirmed", True)):
                    continue

                # Calculation is done but let's format and double check
                entry_price = breakout.get("entry_price")
                stop_loss = breakout.get("stop_loss")
                target = breakout.get("target")

                # Validate risk-reward
                if entry_price and stop_loss and target:
                    if breakout["direction"] == "buy":
                        risk = entry_price - stop_loss
                        reward = target - entry_price
                    else:  # sell
                        risk = stop_loss - entry_price
                        reward = entry_price - target

                    reward_risk_ratio = reward / risk if risk else 0

                    # Skip if R:R is insufficient
                    if reward_risk_ratio < self.config["min_reward_risk_ratio"]:
                        continue

                # Create signal
                signal = {
                    "symbol": symbol,
                    "strategy": self.name,
                    "direction": breakout["direction"],
                    "type": breakout["type"],
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": target,
                    "timeframe": breakout.get("timeframe", ""),
                    "confidence": breakout.get("score", 0.5),
                    "reward_risk_ratio": reward_risk_ratio,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {
                        "breakout_details": {k: v for k, v in breakout.items() if k not in ["type", "direction", "entry_price", "stop_loss", "target", "score", "timeframe"]},
                        "confluence_score": confluence_results.get("confluence_score", 0),
                    }
                }

                signals.append(signal)

        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}", exc_info=True)

        return signals

    async def cleanup(self) -> None:
        """Clean up resources used by the strategy."""
        if hasattr(self, 'feature_store_client'):
            await self.feature_store_client.close()
            self.logger.info("Closed feature store client connection")
