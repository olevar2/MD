"""
Market Regime Classifier

This module provides functionality to classify market regimes based on
extracted features from price data.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from analysis_engine.caching.cache_service import cache_result
from .models import MarketRegimeResult, MarketRegimeType, TrendState, VolatilityState

logger = logging.getLogger(__name__)


class MarketRegimeClassifier:
    """
    Classifier for market regimes based on extracted features.

    This class is responsible for classifying market regimes based on
    features extracted from price data.
    """

    def __init__(self):
        """Initialize the market regime classifier."""
        logger.info("MarketRegimeClassifier initialized")

    @cache_result(ttl=900)  # Cache for 15 minutes
    def classify_regime(
        self,
        instrument: str,
        timeframe: str,
        features: Dict[str, float]
    ) -> MarketRegimeResult:
        """
        Classify market regime based on extracted features.

        Args:
            instrument: The instrument being analyzed
            timeframe: The timeframe being analyzed
            features: Extracted features from price data

        Returns:
            MarketRegimeResult: Classification result
        """
        if not features:
            return MarketRegimeResult(
                instrument=instrument,
                timeframe=timeframe,
                regime=MarketRegimeType.UNKNOWN,
                direction=TrendState.NEUTRAL,
                volatility=VolatilityState.UNKNOWN,
                strength=0.0,
                detected_at=datetime.utcnow()
            )

        # Extract features
        adx = features.get("adx", 0)
        volatility_ratio = features.get("volatility_ratio", 0)
        sma20 = features.get("sma20", 0)
        sma50 = features.get("sma50", 0)
        close = features.get("close", 0)

        # Classify direction
        if close > sma20 and sma20 > sma50:
            direction = TrendState.BULLISH
            direction_strength = min(1.0, (close / sma20 - 1) * 10)
        elif close < sma20 and sma20 < sma50:
            direction = TrendState.BEARISH
            direction_strength = min(1.0, (1 - close / sma20) * 10)
        else:
            direction = TrendState.NEUTRAL
            direction_strength = 0.5

        # Classify volatility
        if volatility_ratio < 0.5:
            volatility = VolatilityState.LOW
        elif volatility_ratio < 1.2:
            volatility = VolatilityState.MEDIUM
        else:
            volatility = VolatilityState.HIGH

        # Classify regime
        if adx > 25:
            # Strong trend
            regime = MarketRegimeType.TRENDING
            strength = min(1.0, adx / 50)
        elif adx < 15 and volatility_ratio < 0.8:
            # Low volatility, no trend
            regime = MarketRegimeType.RANGING
            strength = min(1.0, (25 - adx) / 25)
        else:
            # High volatility without strong trend
            regime = MarketRegimeType.VOLATILE
            strength = min(1.0, volatility_ratio / 2)

        # Create result
        return MarketRegimeResult(
            instrument=instrument,
            timeframe=timeframe,
            regime=regime,
            direction=direction,
            volatility=volatility,
            strength=strength,
            metrics={
                "adx": round(adx, 2),
                "rsi": round(features.get("rsi", 0), 2),
                "volatility_ratio": round(volatility_ratio, 2),
                "ma_diff": round(features.get("ma_diff", 0), 2),
                "price_change": round(features.get("price_change", 0), 2)
            },
            detected_at=datetime.utcnow()
        )

    @cache_result(ttl=900)  # Cache for 15 minutes
    def classify_regime_change(
        self,
        current_regime: MarketRegimeResult,
        previous_regime: MarketRegimeResult
    ) -> Dict[str, Any]:
        """
        Classify regime change based on current and previous regimes.

        Args:
            current_regime: Current regime classification
            previous_regime: Previous regime classification

        Returns:
            Dict[str, Any]: Change classification
        """
        # Determine if there's been a change
        regime_changed = current_regime.regime != previous_regime.regime
        direction_changed = current_regime.direction != previous_regime.direction
        volatility_changed = current_regime.volatility != previous_regime.volatility

        # Calculate significance of change
        significance = 0
        if regime_changed:
            significance += 0.6
        if direction_changed:
            significance += 0.3
        if volatility_changed:
            significance += 0.1

        return {
            "regime_changed": regime_changed,
            "direction_changed": direction_changed,
            "volatility_changed": volatility_changed,
            "change_significance": significance
        }