"""
Market Regime Classifier

This module provides functionality for classifying market regimes based on
extracted features.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from analysis_engine.analysis.market_regime.models import (
    RegimeType, DirectionType, VolatilityLevel, 
    RegimeClassification, FeatureSet
)


class RegimeClassifier:
    """
    Classifies market regimes based on extracted features.
    
    This class is responsible for analyzing feature sets and determining
    the current market regime, direction, and volatility level.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RegimeClassifier.
        
        Args:
            config: Optional configuration dictionary with parameters:
                - trend_threshold: Threshold for trend strength (default: 0.25)
                - volatility_thresholds: Dict with thresholds for volatility levels
                - momentum_threshold: Threshold for momentum (default: 0.2)
                - hysteresis: Dict with hysteresis values for regime changes
        """
        self.config = config or {}
        
        # Thresholds for classification
        self.trend_threshold = self.config.get('trend_threshold', 0.25)
        self.volatility_thresholds = self.config.get('volatility_thresholds', {
            'low': 0.5,
            'medium': 1.0,
            'high': 2.0
        })
        self.momentum_threshold = self.config.get('momentum_threshold', 0.2)
        
        # Hysteresis to prevent rapid regime switching
        self.hysteresis = self.config.get('hysteresis', {
            'trend_to_range': 0.05,
            'range_to_trend': 0.05,
            'volatility_increase': 0.1,
            'volatility_decrease': 0.1
        })
        
        # State for hysteresis
        self.previous_classification = None
    
    def classify(self, features: FeatureSet, timestamp: Optional[datetime] = None) -> RegimeClassification:
        """
        Classify market regime based on extracted features.
        
        Args:
            features: FeatureSet containing extracted features
            timestamp: Optional timestamp for the classification
                
        Returns:
            RegimeClassification: Classification result with regime type,
                direction, volatility level, and confidence
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Extract key features
        trend_strength = features.trend_strength
        volatility = features.volatility
        momentum = features.momentum
        mean_reversion = features.mean_reversion
        range_width = features.range_width
        
        # Determine direction
        direction = self._classify_direction(momentum)
        
        # Determine volatility level
        volatility_level = self._classify_volatility(volatility)
        
        # Apply hysteresis if we have a previous classification
        if self.previous_classification is not None:
            trend_strength = self._apply_trend_hysteresis(
                trend_strength, 
                self.previous_classification.regime
            )
            
            volatility = self._apply_volatility_hysteresis(
                volatility, 
                self.previous_classification.volatility
            )
        
        # Determine regime type
        regime_type, confidence = self._classify_regime_type(
            trend_strength, volatility_level, direction, mean_reversion, range_width
        )
        
        # Create classification result
        classification = RegimeClassification(
            regime=regime_type,
            confidence=confidence,
            direction=direction,
            volatility=volatility_level,
            timestamp=timestamp,
            features=features.to_dict()
        )
        
        # Store for hysteresis
        self.previous_classification = classification
        
        return classification
    
    def _classify_direction(self, momentum: float) -> DirectionType:
        """Classify market direction based on momentum."""
        if momentum > self.momentum_threshold:
            return DirectionType.BULLISH
        elif momentum < -self.momentum_threshold:
            return DirectionType.BEARISH
        else:
            return DirectionType.NEUTRAL
    
    def _classify_volatility(self, volatility: float) -> VolatilityLevel:
        """Classify volatility level."""
        if volatility < self.volatility_thresholds['low']:
            return VolatilityLevel.LOW
        elif volatility < self.volatility_thresholds['medium']:
            return VolatilityLevel.MEDIUM
        elif volatility < self.volatility_thresholds['high']:
            return VolatilityLevel.HIGH
        else:
            return VolatilityLevel.EXTREME
    
    def _classify_regime_type(
        self, 
        trend_strength: float, 
        volatility_level: VolatilityLevel,
        direction: DirectionType,
        mean_reversion: float,
        range_width: float
    ) -> Tuple[RegimeType, float]:
        """
        Classify regime type based on features.
        
        Returns:
            Tuple containing RegimeType and confidence score
        """
        # High volatility overrides other considerations
        if volatility_level == VolatilityLevel.EXTREME:
            if direction == DirectionType.BULLISH:
                return RegimeType.VOLATILE_BULLISH, 0.9
            elif direction == DirectionType.BEARISH:
                return RegimeType.VOLATILE_BEARISH, 0.9
            else:
                return RegimeType.VOLATILE_NEUTRAL, 0.9
        
        # Strong trend indicates trending regime
        if trend_strength > self.trend_threshold:
            confidence = min(trend_strength * 2, 0.95)  # Scale confidence
            
            if direction == DirectionType.BULLISH:
                return RegimeType.TRENDING_BULLISH, confidence
            elif direction == DirectionType.BEARISH:
                return RegimeType.TRENDING_BEARISH, confidence
            else:
                # Unusual case: strong trend but neutral direction
                # Might indicate transition or conflicting signals
                if mean_reversion > 0:
                    return RegimeType.RANGING_BULLISH, 0.6
                else:
                    return RegimeType.RANGING_BEARISH, 0.6
        
        # Not trending, so ranging
        # Use range width and mean reversion to determine confidence
        confidence = min(0.5 + range_width, 0.9)
        
        if direction == DirectionType.BULLISH:
            return RegimeType.RANGING_BULLISH, confidence
        elif direction == DirectionType.BEARISH:
            return RegimeType.RANGING_BEARISH, confidence
        else:
            return RegimeType.RANGING_NEUTRAL, confidence
    
    def _apply_trend_hysteresis(self, trend_strength: float, previous_regime: RegimeType) -> float:
        """Apply hysteresis to trend strength to prevent rapid regime switching."""
        was_trending = previous_regime in (
            RegimeType.TRENDING_BULLISH, 
            RegimeType.TRENDING_BEARISH
        )
        
        if was_trending:
            # Make it harder to switch from trending to ranging
            return trend_strength + self.hysteresis['trend_to_range']
        else:
            # Make it harder to switch from ranging to trending
            return trend_strength - self.hysteresis['range_to_trend']
    
    def _apply_volatility_hysteresis(self, volatility: float, previous_volatility: VolatilityLevel) -> float:
        """Apply hysteresis to volatility to prevent rapid level changes."""
        volatility_levels = {
            VolatilityLevel.LOW: 0,
            VolatilityLevel.MEDIUM: 1,
            VolatilityLevel.HIGH: 2,
            VolatilityLevel.EXTREME: 3
        }
        
        previous_level = volatility_levels[previous_volatility]
        
        # Determine if volatility is increasing or decreasing
        current_level = 0
        for level_name, threshold in sorted(
            self.volatility_thresholds.items(), 
            key=lambda x: x[1]
        ):
            if volatility >= threshold:
                current_level += 1
        
        if current_level > previous_level:
            # Make it harder to increase volatility level
            return volatility - self.hysteresis['volatility_increase']
        elif current_level < previous_level:
            # Make it harder to decrease volatility level
            return volatility + self.hysteresis['volatility_decrease']
        
        return volatility