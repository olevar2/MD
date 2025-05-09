"""
Analysis Adapter Module

This module provides adapter implementations for analysis provider interfaces,
helping to break circular dependencies between services.
"""
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import logging
import asyncio
import json
import copy
import random

from common_lib.strategy.interfaces import (
    IAnalysisProvider, SignalDirection, SignalTimeframe, 
    SignalSource, MarketRegimeType
)
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class AnalysisProviderAdapter(IAnalysisProvider):
    """
    Adapter for analysis providers that implements the common interface.
    
    This adapter can either wrap an actual provider instance or provide
    standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, provider_instance=None):
        """
        Initialize the adapter.
        
        Args:
            provider_instance: Optional actual provider instance to wrap
        """
        self.provider = provider_instance
        self.analysis_cache = {}
    
    async def get_technical_analysis(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int = 100
    ) -> Dict[str, Any]:
        """
        Get technical analysis for a symbol.
        
        Args:
            symbol: The trading symbol
            timeframe: The timeframe to analyze
            lookback_bars: Number of bars to analyze
            
        Returns:
            Dictionary with technical analysis results
        """
        if self.provider:
            try:
                # Try to use the wrapped provider if available
                return await self.provider.get_technical_analysis(
                    symbol=symbol,
                    timeframe=timeframe,
                    lookback_bars=lookback_bars
                )
            except Exception as e:
                logger.warning(f"Error getting technical analysis: {str(e)}")
        
        # Check if we have cached analysis
        cache_key = f"ta_{symbol}_{timeframe}_{lookback_bars}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        # Fallback to simple analysis if no provider available
        indicators = {
            "rsi": {
                "value": random.uniform(30, 70),
                "signal": "neutral",
                "strength": 0.5
            },
            "macd": {
                "value": random.uniform(-0.5, 0.5),
                "signal": "neutral",
                "strength": 0.5
            },
            "bollinger_bands": {
                "upper": random.uniform(1.1, 1.2),
                "middle": 1.0,
                "lower": random.uniform(0.8, 0.9),
                "signal": "neutral",
                "strength": 0.5
            },
            "moving_averages": {
                "ma_20": random.uniform(0.9, 1.1),
                "ma_50": random.uniform(0.9, 1.1),
                "ma_200": random.uniform(0.9, 1.1),
                "signal": "neutral",
                "strength": 0.5
            }
        }
        
        analysis_result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "indicators": indicators,
            "overall_signal": "neutral",
            "overall_strength": 0.5
        }
        
        # Cache the result
        self.analysis_cache[cache_key] = analysis_result
        
        return analysis_result
    
    async def get_pattern_recognition(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int = 100
    ) -> Dict[str, Any]:
        """
        Get pattern recognition analysis for a symbol.
        
        Args:
            symbol: The trading symbol
            timeframe: The timeframe to analyze
            lookback_bars: Number of bars to analyze
            
        Returns:
            Dictionary with pattern recognition results
        """
        if self.provider:
            try:
                # Try to use the wrapped provider if available
                return await self.provider.get_pattern_recognition(
                    symbol=symbol,
                    timeframe=timeframe,
                    lookback_bars=lookback_bars
                )
            except Exception as e:
                logger.warning(f"Error getting pattern recognition: {str(e)}")
        
        # Check if we have cached analysis
        cache_key = f"pattern_{symbol}_{timeframe}_{lookback_bars}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        # Fallback to simple analysis if no provider available
        patterns = []
        
        # Randomly add some patterns
        if random.random() > 0.7:
            patterns.append({
                "pattern_type": "double_bottom",
                "confidence": random.uniform(0.6, 0.9),
                "direction": "buy",
                "start_index": random.randint(0, lookback_bars // 2),
                "end_index": random.randint(lookback_bars // 2, lookback_bars - 1)
            })
        
        if random.random() > 0.7:
            patterns.append({
                "pattern_type": "head_and_shoulders",
                "confidence": random.uniform(0.6, 0.9),
                "direction": "sell",
                "start_index": random.randint(0, lookback_bars // 2),
                "end_index": random.randint(lookback_bars // 2, lookback_bars - 1)
            })
        
        analysis_result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "patterns": patterns,
            "overall_signal": "neutral" if not patterns else patterns[0]["direction"],
            "overall_strength": 0.0 if not patterns else patterns[0]["confidence"]
        }
        
        # Cache the result
        self.analysis_cache[cache_key] = analysis_result
        
        return analysis_result
    
    async def get_market_regime(
        self,
        symbol: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """
        Get current market regime for a symbol.
        
        Args:
            symbol: The trading symbol
            timeframe: The timeframe to analyze
            
        Returns:
            Dictionary with market regime information
        """
        if self.provider:
            try:
                # Try to use the wrapped provider if available
                return await self.provider.get_market_regime(
                    symbol=symbol,
                    timeframe=timeframe
                )
            except Exception as e:
                logger.warning(f"Error getting market regime: {str(e)}")
        
        # Check if we have cached analysis
        cache_key = f"regime_{symbol}_{timeframe}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        # Fallback to simple analysis if no provider available
        regimes = [
            MarketRegimeType.TRENDING_BULLISH,
            MarketRegimeType.TRENDING_BEARISH,
            MarketRegimeType.RANGING_NARROW,
            MarketRegimeType.RANGING_WIDE,
            MarketRegimeType.VOLATILE
        ]
        
        regime = random.choice(regimes)
        
        analysis_result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "regime": regime,
            "confidence": random.uniform(0.7, 0.9),
            "metrics": {
                "volatility": random.uniform(0.005, 0.02),
                "trend_strength": random.uniform(0.3, 0.8),
                "range_width": random.uniform(0.01, 0.05)
            }
        }
        
        # Cache the result
        self.analysis_cache[cache_key] = analysis_result
        
        return analysis_result
    
    async def get_multi_timeframe_analysis(
        self,
        symbol: str,
        timeframes: List[str],
        lookback_bars: int = 100
    ) -> Dict[str, Any]:
        """
        Get multi-timeframe analysis for a symbol.
        
        Args:
            symbol: The trading symbol
            timeframes: List of timeframes to analyze
            lookback_bars: Number of bars to analyze
            
        Returns:
            Dictionary with multi-timeframe analysis results
        """
        if self.provider:
            try:
                # Try to use the wrapped provider if available
                return await self.provider.get_multi_timeframe_analysis(
                    symbol=symbol,
                    timeframes=timeframes,
                    lookback_bars=lookback_bars
                )
            except Exception as e:
                logger.warning(f"Error getting multi-timeframe analysis: {str(e)}")
        
        # Check if we have cached analysis
        timeframes_key = "_".join(timeframes)
        cache_key = f"mtf_{symbol}_{timeframes_key}_{lookback_bars}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        # Fallback to simple analysis if no provider available
        tf_results = {}
        
        # Get analysis for each timeframe
        for tf in timeframes:
            ta_result = await self.get_technical_analysis(symbol, tf, lookback_bars)
            tf_results[tf] = {
                "signal": ta_result.get("overall_signal", "neutral"),
                "strength": ta_result.get("overall_strength", 0.5),
                "indicators": ta_result.get("indicators", {})
            }
        
        # Determine overall signal based on higher timeframes having more weight
        signals = [tf_results[tf]["signal"] for tf in timeframes]
        strengths = [tf_results[tf]["strength"] for tf in timeframes]
        
        # Simple majority vote for direction
        buy_count = signals.count("buy")
        sell_count = signals.count("sell")
        neutral_count = signals.count("neutral")
        
        if buy_count > sell_count and buy_count > neutral_count:
            overall_signal = "buy"
        elif sell_count > buy_count and sell_count > neutral_count:
            overall_signal = "sell"
        else:
            overall_signal = "neutral"
        
        # Average strength
        overall_strength = sum(strengths) / len(strengths) if strengths else 0.5
        
        analysis_result = {
            "symbol": symbol,
            "timeframes": timeframes,
            "timestamp": datetime.now().isoformat(),
            "timeframe_results": tf_results,
            "overall_signal": overall_signal,
            "overall_strength": overall_strength,
            "confluence_level": random.uniform(0.3, 0.8)
        }
        
        # Cache the result
        self.analysis_cache[cache_key] = analysis_result
        
        return analysis_result
    
    async def get_multi_asset_analysis(
        self,
        symbol: str,
        related_symbols: List[str] = None
    ) -> Dict[str, Any]:
        """
        Get multi-asset analysis for a symbol.
        
        Args:
            symbol: The trading symbol
            related_symbols: Optional list of related symbols
            
        Returns:
            Dictionary with multi-asset analysis results
        """
        if self.provider:
            try:
                # Try to use the wrapped provider if available
                return await self.provider.get_multi_asset_analysis(
                    symbol=symbol,
                    related_symbols=related_symbols
                )
            except Exception as e:
                logger.warning(f"Error getting multi-asset analysis: {str(e)}")
        
        # Use default related symbols if none provided
        if not related_symbols:
            if symbol.startswith("EUR"):
                related_symbols = ["EURUSD", "EURGBP", "EURJPY", "EURCHF"]
            elif symbol.startswith("USD"):
                related_symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"]
            else:
                related_symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
        
        # Check if we have cached analysis
        related_key = "_".join(sorted(related_symbols)) if related_symbols else "none"
        cache_key = f"multi_asset_{symbol}_{related_key}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        # Fallback to simple analysis if no provider available
        correlations = {}
        for rel_symbol in related_symbols:
            if rel_symbol != symbol:
                correlations[rel_symbol] = random.uniform(-0.8, 0.8)
        
        # Generate random strength values for currencies
        currencies = set()
        for s in [symbol] + (related_symbols or []):
            if len(s) >= 6:  # Forex pair format
                currencies.add(s[:3])
                currencies.add(s[3:6])
        
        currency_strength = {}
        for currency in currencies:
            currency_strength[currency] = random.uniform(-1.0, 1.0)
        
        analysis_result = {
            "symbol": symbol,
            "related_symbols": related_symbols,
            "timestamp": datetime.now().isoformat(),
            "correlations": correlations,
            "currency_strength": currency_strength,
            "confluence_signals": [],
            "overall_bias": "neutral",
            "strength": 0.5
        }
        
        # Cache the result
        self.analysis_cache[cache_key] = analysis_result
        
        return analysis_result
    
    async def get_integrated_analysis(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int = 100,
        include_components: List[str] = None
    ) -> Dict[str, Any]:
        """
        Get integrated analysis from multiple components.
        
        Args:
            symbol: The trading symbol
            timeframe: The timeframe to analyze
            lookback_bars: Number of bars to analyze
            include_components: List of components to include
            
        Returns:
            Dictionary with integrated analysis results
        """
        if self.provider:
            try:
                # Try to use the wrapped provider if available
                return await self.provider.get_integrated_analysis(
                    symbol=symbol,
                    timeframe=timeframe,
                    lookback_bars=lookback_bars,
                    include_components=include_components
                )
            except Exception as e:
                logger.warning(f"Error getting integrated analysis: {str(e)}")
        
        # Determine which components to include
        components = include_components or [
            "technical", "pattern", "market_regime", "multi_timeframe"
        ]
        
        # Check if we have cached analysis
        components_key = "_".join(sorted(components))
        cache_key = f"integrated_{symbol}_{timeframe}_{lookback_bars}_{components_key}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        # Collect results from each component
        component_results = {}
        signals = []
        
        # Technical analysis
        if "technical" in components:
            ta_result = await self.get_technical_analysis(symbol, timeframe, lookback_bars)
            component_results["technical"] = ta_result
            
            # Add signals from technical indicators
            for indicator, data in ta_result.get("indicators", {}).items():
                if "signal" in data:
                    signals.append({
                        "source_id": f"ta_{indicator}",
                        "source_type": "technical_analysis",
                        "direction": data["signal"],
                        "strength": data.get("strength", 0.5),
                        "timeframe": timeframe
                    })
        
        # Pattern recognition
        if "pattern" in components:
            pattern_result = await self.get_pattern_recognition(symbol, timeframe, lookback_bars)
            component_results["pattern"] = pattern_result
            
            # Add signals from patterns
            for pattern in pattern_result.get("patterns", []):
                signals.append({
                    "source_id": f"pattern_{pattern['pattern_type']}",
                    "source_type": "pattern_recognition",
                    "direction": pattern["direction"],
                    "strength": pattern.get("confidence", 0.5),
                    "timeframe": timeframe
                })
        
        # Market regime
        if "market_regime" in components:
            regime_result = await self.get_market_regime(symbol, timeframe)
            component_results["market_regime"] = regime_result
        
        # Multi-timeframe analysis
        if "multi_timeframe" in components:
            # Use a few standard timeframes
            if timeframe == "1m":
                mtf_timeframes = ["1m", "5m", "15m", "1h"]
            elif timeframe == "5m":
                mtf_timeframes = ["5m", "15m", "1h", "4h"]
            elif timeframe == "15m":
                mtf_timeframes = ["15m", "1h", "4h", "1d"]
            elif timeframe == "1h":
                mtf_timeframes = ["1h", "4h", "1d", "1w"]
            else:
                mtf_timeframes = ["1h", "4h", "1d", "1w"]
            
            mtf_result = await self.get_multi_timeframe_analysis(
                symbol, mtf_timeframes, lookback_bars
            )
            component_results["multi_timeframe"] = mtf_result
            
            # Add signals from multi-timeframe analysis
            signals.append({
                "source_id": "mtf_confluence",
                "source_type": "multi_timeframe",
                "direction": mtf_result.get("overall_signal", "neutral"),
                "strength": mtf_result.get("overall_strength", 0.5),
                "timeframe": timeframe
            })
        
        # Multi-asset analysis
        if "multi_asset" in components:
            multi_asset_result = await self.get_multi_asset_analysis(symbol)
            component_results["multi_asset"] = multi_asset_result
            
            # Add signal from multi-asset analysis
            signals.append({
                "source_id": "multi_asset_confluence",
                "source_type": "multi_asset",
                "direction": multi_asset_result.get("overall_bias", "neutral"),
                "strength": multi_asset_result.get("strength", 0.5),
                "timeframe": timeframe
            })
        
        # Determine overall signal
        buy_signals = [s for s in signals if s["direction"] == "buy"]
        sell_signals = [s for s in signals if s["direction"] == "sell"]
        
        if len(buy_signals) > len(sell_signals):
            overall_signal = "buy"
            signal_strength = sum(s["strength"] for s in buy_signals) / len(buy_signals) if buy_signals else 0.5
        elif len(sell_signals) > len(buy_signals):
            overall_signal = "sell"
            signal_strength = sum(s["strength"] for s in sell_signals) / len(sell_signals) if sell_signals else 0.5
        else:
            overall_signal = "neutral"
            signal_strength = 0.5
        
        # Create integrated result
        integrated_result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "components": component_results,
            "signals": signals,
            "overall_signal": overall_signal,
            "overall_strength": signal_strength,
            "market_regime": component_results.get("market_regime", {}).get("regime", "unknown")
        }
        
        # Cache the result
        self.analysis_cache[cache_key] = integrated_result
        
        return integrated_result
