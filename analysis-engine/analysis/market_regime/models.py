"""
Market Regime Analysis Models

This module defines data models and enums used in market regime analysis.
"""

from enum import Enum, auto
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List


class RegimeType(Enum):
    """Enum representing different market regime types."""
    TRENDING_BULLISH = auto()
    TRENDING_BEARISH = auto()
    RANGING_NEUTRAL = auto()
    RANGING_BULLISH = auto()
    RANGING_BEARISH = auto()
    VOLATILE_BULLISH = auto()
    VOLATILE_BEARISH = auto()
    VOLATILE_NEUTRAL = auto()
    UNDEFINED = auto()


class VolatilityLevel(Enum):
    """Enum representing different volatility levels."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    EXTREME = auto()


class DirectionType(Enum):
    """Enum representing market direction."""
    BULLISH = auto()
    BEARISH = auto()
    NEUTRAL = auto()


@dataclass
class RegimeClassification:
    """Data model for regime classification results."""
    regime: RegimeType
    confidence: float
    direction: DirectionType
    volatility: VolatilityLevel
    timestamp: datetime
    features: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the classification to a dictionary."""
        return {
            'regime': self.regime.name,
            'confidence': self.confidence,
            'direction': self.direction.name,
            'volatility': self.volatility.name,
            'timestamp': self.timestamp.isoformat(),
            'features': self.features,
            'metadata': self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RegimeClassification':
        """Create a classification from a dictionary."""
        return cls(
            regime=RegimeType[data['regime']],
            confidence=data['confidence'],
            direction=DirectionType[data['direction']],
            volatility=VolatilityLevel[data['volatility']],
            timestamp=datetime.fromisoformat(data['timestamp']),
            features=data['features'],
            metadata=data.get('metadata', {})
        )


@dataclass
class FeatureSet:
    """Data model for feature sets used in regime analysis."""
    volatility: float
    trend_strength: float
    momentum: float
    mean_reversion: float
    range_width: float
    additional_features: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert the feature set to a dictionary."""
        result = {
            'volatility': self.volatility,
            'trend_strength': self.trend_strength,
            'momentum': self.momentum,
            'mean_reversion': self.mean_reversion,
            'range_width': self.range_width
        }
        
        if self.additional_features:
            result.update(self.additional_features)
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'FeatureSet':
        """Create a feature set from a dictionary."""
        base_features = {
            'volatility', 'trend_strength', 'momentum', 
            'mean_reversion', 'range_width'
        }
        
        additional_features = {k: v for k, v in data.items() if k not in base_features}
        
        return cls(
            volatility=data['volatility'],
            trend_strength=data['trend_strength'],
            momentum=data['momentum'],
            mean_reversion=data['mean_reversion'],
            range_width=data['range_width'],
            additional_features=additional_features or None
        )