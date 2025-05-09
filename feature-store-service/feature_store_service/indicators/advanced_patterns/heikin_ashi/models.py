"""
Heikin-Ashi Pattern Models Module.

This module defines the core data models and enums used in Heikin-Ashi pattern analysis.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime


class HeikinAshiPatternType(Enum):
    """Types of Heikin-Ashi patterns."""
    REVERSAL = "heikin_ashi_reversal"
    CONTINUATION = "heikin_ashi_continuation"
    STRONG_TREND = "heikin_ashi_strong_trend"
    WEAK_TREND = "heikin_ashi_weak_trend"


class HeikinAshiTrendType(Enum):
    """Types of Heikin-Ashi trends."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class HeikinAshiCandle:
    """
    Represents a Heikin-Ashi candle.
    
    Attributes:
        open: Open price
        high: High price
        low: Low price
        close: Close price
        timestamp: Candle timestamp
        index: Index in the DataFrame
    """
    open: float
    high: float
    low: float
    close: float
    timestamp: datetime
    index: int


@dataclass
class HeikinAshiPattern:
    """
    Represents a Heikin-Ashi pattern.
    
    Attributes:
        pattern_type: Type of the Heikin-Ashi pattern
        start_index: Starting index of the pattern
        end_index: Ending index of the pattern
        candles: List of Heikin-Ashi candles in the pattern
        trend_type: Type of trend (bullish, bearish, neutral)
        strength: Strength of the pattern (0.0-1.0)
        target_price: Target price projection
        stop_price: Suggested stop loss price
    """
    pattern_type: HeikinAshiPatternType
    start_index: int
    end_index: int
    candles: List[HeikinAshiCandle]
    trend_type: HeikinAshiTrendType
    strength: float
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "pattern_type": self.pattern_type.value,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "trend_type": self.trend_type.value,
            "strength": self.strength,
            "target_price": self.target_price,
            "stop_price": self.stop_price
        }