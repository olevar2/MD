"""
News Impact Model

This module provides data models for representing the impact of news on financial markets.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ImpactLevel(Enum):
    """Enumeration of impact levels for news events."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class SentimentLevel(Enum):
    """Enumeration of sentiment levels for news events."""
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1


@dataclass
class NewsImpact:
    """
    Data model for news impact on financial markets.
    
    This class represents the impact of a news event on financial markets,
    including the affected instruments, impact level, and direction.
    """
    
    # News metadata
    news_id: str
    timestamp: datetime
    headline: str
    
    # Impact details
    impact_level: ImpactLevel = ImpactLevel.LOW
    sentiment_level: SentimentLevel = SentimentLevel.NEUTRAL
    impact_score: float = 0.0  # Normalized score from 0 to 1
    direction: float = 0.0  # -1 to 1, where negative means bearish
    
    # Affected instruments
    affected_instruments: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Categories
    categories: Dict[str, float] = field(default_factory=dict)
    
    @property
    def is_high_impact(self) -> bool:
        """Check if this is a high-impact news event."""
        return self.impact_level == ImpactLevel.HIGH or self.impact_score >= 0.7
    
    @property
    def direction_label(self) -> str:
        """Get the direction label based on the direction score."""
        if self.direction >= 0.2:
            return "bullish"
        elif self.direction <= -0.2:
            return "bearish"
        else:
            return "neutral"
    
    @property
    def top_affected_instrument(self) -> str:
        """Get the top affected instrument based on impact score."""
        if not self.affected_instruments:
            return "none"
        
        return max(self.affected_instruments.items(), 
                  key=lambda x: abs(x[1].get("impact_score", 0)))[0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "news_id": self.news_id,
            "timestamp": self.timestamp.isoformat(),
            "headline": self.headline,
            "impact_level": self.impact_level.name,
            "sentiment_level": self.sentiment_level.name,
            "impact_score": self.impact_score,
            "direction": self.direction,
            "direction_label": self.direction_label,
            "affected_instruments": self.affected_instruments,
            "categories": self.categories,
            "is_high_impact": self.is_high_impact,
            "top_affected_instrument": self.top_affected_instrument
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NewsImpact':
        """Create from dictionary representation."""
        # Parse timestamp
        timestamp = datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00')) \
            if isinstance(data.get("timestamp"), str) else datetime.now()
        
        # Parse impact level
        impact_level = ImpactLevel[data.get("impact_level", "LOW")] \
            if isinstance(data.get("impact_level"), str) else ImpactLevel.LOW
        
        # Parse sentiment level
        sentiment_level = SentimentLevel[data.get("sentiment_level", "NEUTRAL")] \
            if isinstance(data.get("sentiment_level"), str) else SentimentLevel.NEUTRAL
        
        return cls(
            news_id=data.get("news_id", ""),
            timestamp=timestamp,
            headline=data.get("headline", ""),
            impact_level=impact_level,
            sentiment_level=sentiment_level,
            impact_score=data.get("impact_score", 0.0),
            direction=data.get("direction", 0.0),
            affected_instruments=data.get("affected_instruments", {}),
            categories=data.get("categories", {})
        )