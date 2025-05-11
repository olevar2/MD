"""
Degraded Mode Strategies for Analysis Engine Service

This module implements degraded mode operations for the Analysis Engine Service.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from core_foundations.resilience.degraded_mode import (
    DegradedModeManager,
    DegradedModeStrategy,
    with_degraded_mode,
    fallback_for
)
from core_foundations.exceptions.service_exceptions import DependencyUnavailableError

logger = logging.getLogger(__name__)

# Local cache for degraded mode operations
_local_cache = {
    "indicators": {},
    "patterns": {},
    "signals": {},
    "market_regimes": {},
}


class AnalysisEngineDegradedMode:
    """
    Degraded mode operations for Analysis Engine Service.

    This class provides fallback strategies when dependencies of the
    Analysis Engine Service are unavailable.
    """

    @staticmethod
    def initialize_degraded_mode() -> None:
        """Initialize degraded mode components"""
        manager = DegradedModeManager()

        # Register dependencies
        manager.register_dependency("feature-store")
        manager.register_dependency("data-pipeline")
        manager.register_dependency("ml-integration")

        # Set up dependency monitor
        from core_foundations.monitoring.health_check import HealthCheck
        health_check = HealthCheck.get_instance()
        if health_check:
            from core_foundations.resilience.degraded_mode import create_health_check_dependency_monitor
            create_health_check_dependency_monitor(health_check)

        logger.info("Initialized Analysis Engine degraded mode components")

    @staticmethod
    def update_cache(cache_type: str, key: str, data: Any) -> None:
        """
        Update local cache with data.

        Args:
            cache_type: Type of cache (indicators, patterns, signals, market_regimes)
            key: Cache key
            data: Data to cache
        """
        if cache_type not in _local_cache:
            _local_cache[cache_type] = {}

        _local_cache[cache_type][key] = {
            "data": data,
            "timestamp": datetime.utcnow()
        }

        # Log cache update but limit data size in log
        log_data = str(data)
        if len(log_data) > 100:
            log_data = log_data[:97] + "..."
        logger.debug(f"Updated {cache_type} cache for {key}: {log_data}")

    @staticmethod
    def get_from_cache(cache_type: str, key: str, max_age_minutes: int = 60) -> Optional[Any]:
        """
        Get data from local cache if not expired.

        Args:
            cache_type: Type of cache (indicators, patterns, signals, market_regimes)
            key: Cache key
            max_age_minutes: Maximum age of cached data in minutes

        Returns:
            Cached data or None if not found or expired
        """
        if cache_type not in _local_cache:
            return None

        cache_entry = _local_cache[cache_type].get(key)
        if not cache_entry:
            return None

        # Check if expired
        age = datetime.utcnow() - cache_entry["timestamp"]
        if age > timedelta(minutes=max_age_minutes):
            logger.debug(f"Cache expired for {cache_type}:{key}")
            return None

        logger.debug(f"Cache hit for {cache_type}:{key}, age={age.total_seconds():.1f}s")
        return cache_entry["data"]


# Original function
@with_degraded_mode("feature-store", DegradedModeStrategy.USE_FALLBACK)
def get_technical_indicators(symbol: str, timeframe: str, indicators: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Get technical indicators from Feature Store.

    Args:
        symbol: Currency pair symbol
        timeframe: Candlestick timeframe
        indicators: List of indicator names

    Returns:
        Dictionary mapping indicator names to DataFrame with indicator values
    """
    # Use the adapter pattern to get the feature provider
    from analysis_engine.clients.service_client_factory import ServiceClientFactory
    factory = ServiceClientFactory()
    feature_provider = factory.create_feature_provider()

    # Get data from feature store
    indicator_data = {}
    for indicator in indicators:
        # Convert to datetime for the adapter
        from datetime import datetime, timedelta
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=30)  # Get 30 days of data

        # Get the indicator data
        df = feature_provider.get_feature(
            feature_name=indicator,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )

        indicator_data[indicator] = df

    # Update local cache for degraded mode
    cache_key = f"{symbol}_{timeframe}_{'-'.join(sorted(indicators))}"
    AnalysisEngineDegradedMode.update_cache("indicators", cache_key, indicator_data)

    return indicator_data


# Fallback implementation
@fallback_for("feature-store", get_technical_indicators)
def get_technical_indicators_fallback(symbol: str, timeframe: str, indicators: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Fallback implementation for getting technical indicators when Feature Store is unavailable.

    Args:
        symbol: Currency pair symbol
        timeframe: Candlestick timeframe
        indicators: List of indicator names

    Returns:
        Dictionary mapping indicator names to DataFrame with indicator values
    """
    logger.warning(f"Using fallback for technical indicators: {symbol} {timeframe}")

    # Try to get from cache
    cache_key = f"{symbol}_{timeframe}_{'-'.join(sorted(indicators))}"
    cached_data = AnalysisEngineDegradedMode.get_from_cache("indicators", cache_key)

    if cached_data:
        logger.info(f"Using cached indicators for {symbol} {timeframe}")
        return cached_data

    # If not in cache, try to compute basic indicators locally
    # Note: This is simplified and would only work for basic indicators
    try:
        # Try to get raw data from data pipeline directly
        from data_pipeline_client import DataPipelineClient
        data_client = DataPipelineClient()

        # Get OHLCV data
        ohlcv_data = data_client.get_ohlcv(symbol, timeframe, limit=100)
        df = pd.DataFrame(ohlcv_data)

        # Compute basic indicators locally
        result = {}

        for indicator in indicators:
            if indicator == "sma_20":
                result[indicator] = df["close"].rolling(window=20).mean()
            elif indicator == "ema_20":
                result[indicator] = df["close"].ewm(span=20, adjust=False).mean()
            elif indicator == "rsi_14":
                delta = df["close"].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                result[indicator] = 100 - (100 / (1 + rs))
            else:
                # Unknown indicator
                logger.warning(f"Cannot compute indicator {indicator} in fallback mode")
                result[indicator] = pd.Series(float('nan'), index=df.index)

        logger.info(f"Computed fallback indicators for {symbol} {timeframe}")
        return result

    except Exception as e:
        # If we can't compute, return empty data
        logger.error(f"Failed to compute fallback indicators: {str(e)}")

        # Return empty DataFrames
        result = {indicator: pd.DataFrame() for indicator in indicators}
        return result


# Original function
@with_degraded_mode("ml-integration", DegradedModeStrategy.USE_FALLBACK)
def get_market_regime_prediction(symbol: str, timeframe: str) -> Dict[str, Any]:
    """
    Get market regime prediction from ML Integration Service.

    Args:
        symbol: Currency pair symbol
        timeframe: Candlestick timeframe

    Returns:
        Market regime prediction with confidence scores
    """
    # Use the adapter pattern to get the analysis provider
    from analysis_engine.clients.service_client_factory import ServiceClientFactory
    factory = ServiceClientFactory()
    analysis_provider = factory.create_analysis_provider()

    # Convert to datetime for the adapter
    from datetime import datetime, timedelta
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=7)  # Get 7 days of data for analysis

    # Get prediction
    analysis_result = analysis_provider.analyze_market(
        symbol=symbol,
        timeframe=timeframe,
        analysis_type="market_regime",
        start_time=start_time,
        end_time=end_time
    )

    # Extract prediction from analysis result
    prediction = {
        "regime": analysis_result.get("regime", "UNKNOWN"),
        "confidence": analysis_result.get("confidence", 0.5),
        "timestamp": datetime.utcnow().isoformat(),
        "method": "ml_model"
    }

    # Update local cache for degraded mode
    cache_key = f"{symbol}_{timeframe}"
    AnalysisEngineDegradedMode.update_cache("market_regimes", cache_key, prediction)

    return prediction


# Fallback implementation
@fallback_for("ml-integration", get_market_regime_prediction)
def get_market_regime_prediction_fallback(symbol: str, timeframe: str) -> Dict[str, Any]:
    """
    Fallback implementation for getting market regime when ML Integration Service is unavailable.

    Args:
        symbol: Currency pair symbol
        timeframe: Candlestick timeframe

    Returns:
        Market regime prediction with confidence scores
    """
    logger.warning(f"Using fallback for market regime prediction: {symbol} {timeframe}")

    # Try to get from cache
    cache_key = f"{symbol}_{timeframe}"
    cached_prediction = AnalysisEngineDegradedMode.get_from_cache("market_regimes", cache_key)

    if cached_prediction:
        logger.info(f"Using cached market regime for {symbol} {timeframe}")
        # Add flag to indicate this is from cache
        cached_prediction["from_cache"] = True
        cached_prediction["cache_time"] = datetime.utcnow().isoformat()
        return cached_prediction

    # If not in cache, use heuristic to determine market regime
    try:
        # Try to get technical indicators
        indicators = get_technical_indicators(symbol, timeframe, ["atr_14", "sma_20", "sma_50", "rsi_14"])

        # Simple heuristic logic for market regime
        if len(indicators.get("atr_14", pd.DataFrame())) > 0:
            atr = indicators["atr_14"].iloc[-1]
            sma_20 = indicators["sma_20"].iloc[-1]
            sma_50 = indicators["sma_50"].iloc[-1]
            rsi = indicators["rsi_14"].iloc[-1]

            # Simplified logic
            if sma_20 > sma_50 and rsi > 60:
                regime = "TRENDING_UP"
                confidence = 0.7
            elif sma_20 < sma_50 and rsi < 40:
                regime = "TRENDING_DOWN"
                confidence = 0.7
            elif atr < atr.rolling(window=20).mean().iloc[-1] * 0.8:
                regime = "CONSOLIDATION"
                confidence = 0.6
            else:
                regime = "VOLATILE"
                confidence = 0.5
        else:
            # Default when we can't compute
            regime = "UNKNOWN"
            confidence = 0.3

        # Create fallback prediction
        fallback_prediction = {
            "regime": regime,
            "confidence": confidence,
            "method": "fallback_heuristic",
            "timestamp": datetime.utcnow().isoformat(),
            "is_fallback": True
        }

        logger.info(f"Computed fallback market regime for {symbol} {timeframe}: {regime}")
        return fallback_prediction

    except Exception as e:
        # If all else fails, return unknown regime
        logger.error(f"Failed to compute fallback market regime: {str(e)}")
        return {
            "regime": "UNKNOWN",
            "confidence": 0.1,
            "method": "fallback_default",
            "timestamp": datetime.utcnow().isoformat(),
            "is_fallback": True,
            "error": str(e)
        }
