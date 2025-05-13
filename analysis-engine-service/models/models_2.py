"""
Market Regime Models

This module defines data models and enumerations for market regime analysis.
"""
from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class MarketRegimeType(str, Enum):
    """Enumeration of market regime types"""
    TRENDING = 'trending'
    RANGING = 'ranging'
    VOLATILE = 'volatile'
    UNKNOWN = 'unknown'


class TrendState(str, Enum):
    """Enumeration of trend states"""
    BULLISH = 'bullish'
    BEARISH = 'bearish'
    NEUTRAL = 'neutral'


class VolatilityState(str, Enum):
    """Enumeration of volatility states"""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    UNKNOWN = 'unknown'


@dataclass
class MarketRegimeResult:
    """Result of market regime analysis"""
    instrument: str
    timeframe: str
    regime: MarketRegimeType
    direction: TrendState
    volatility: VolatilityState
    strength: float
    metrics: Optional[Dict[str, float]] = None
    detected_at: datetime = None

    def __post_init__(self):
        if self.detected_at is None:
            self.detected_at = datetime.utcnow()

    def to_dict(self) ->Dict[str, Any]:
        """Convert to dictionary representation"""
        return {'instrument': self.instrument, 'timeframe': self.timeframe,
            'regime': self.regime.value, 'direction': self.direction.value,
            'volatility': self.volatility.value, 'strength': round(self.
            strength, 2), 'metrics': self.metrics, 'detected_at': self.
            detected_at.isoformat() if self.detected_at else None}

    @classmethod
    @with_exception_handling
    def from_dict(cls, data: Dict[str, Any]) ->'MarketRegimeResult':
        """Create from dictionary representation"""
        detected_at = None
        if data.get('detected_at'):
            try:
                detected_at = datetime.fromisoformat(data['detected_at'])
            except (ValueError, TypeError):
                detected_at = datetime.utcnow()
        return cls(instrument=data.get('instrument', ''), timeframe=data.
            get('timeframe', ''), regime=MarketRegimeType(data.get('regime',
            MarketRegimeType.UNKNOWN.value)), direction=TrendState(data.get
            ('direction', TrendState.NEUTRAL.value)), volatility=
            VolatilityState(data.get('volatility', VolatilityState.UNKNOWN.
            value)), strength=data.get('strength', 0.0), metrics=data.get(
            'metrics'), detected_at=detected_at)


@dataclass
class RegimeChangeResult:
    """Result of market regime change analysis"""
    instrument: str
    timeframe: str
    period_days: int
    regime_changed: bool
    direction_changed: bool
    volatility_changed: bool
    change_significance: float
    current_regime: MarketRegimeResult
    previous_regime: MarketRegimeResult
    analyzed_at: datetime = None

    def __post_init__(self):
        if self.analyzed_at is None:
            self.analyzed_at = datetime.utcnow()

    def to_dict(self) ->Dict[str, Any]:
        """Convert to dictionary representation"""
        return {'instrument': self.instrument, 'timeframe': self.timeframe,
            'period_days': self.period_days, 'regime_changed': self.
            regime_changed, 'direction_changed': self.direction_changed,
            'volatility_changed': self.volatility_changed,
            'change_significance': round(self.change_significance, 2),
            'current_regime': self.current_regime.to_dict(),
            'previous_regime': self.previous_regime.to_dict(),
            'analyzed_at': self.analyzed_at.isoformat() if self.analyzed_at
             else None}

    @classmethod
    @with_exception_handling
    def from_dict(cls, data: Dict[str, Any]) ->'RegimeChangeResult':
        """Create from dictionary representation"""
        analyzed_at = None
        if data.get('analyzed_at'):
            try:
                analyzed_at = datetime.fromisoformat(data['analyzed_at'])
            except (ValueError, TypeError):
                analyzed_at = datetime.utcnow()
        return cls(instrument=data.get('instrument', ''), timeframe=data.
            get('timeframe', ''), period_days=data.get('period_days', 0),
            regime_changed=data.get('regime_changed', False),
            direction_changed=data.get('direction_changed', False),
            volatility_changed=data.get('volatility_changed', False),
            change_significance=data.get('change_significance', 0.0),
            current_regime=MarketRegimeResult.from_dict(data.get(
            'current_regime', {})), previous_regime=MarketRegimeResult.
            from_dict(data.get('previous_regime', {})), analyzed_at=analyzed_at
            )
