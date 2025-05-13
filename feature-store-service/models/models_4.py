"""
Volume Spread Analysis (VSA) Pattern Models Module.

This module defines the core data models and enums used in VSA pattern analysis.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime


class VSAPatternType(Enum):
    """Types of VSA patterns."""
    NO_DEMAND = "vsa_no_demand"
    NO_SUPPLY = "vsa_no_supply"
    STOPPING_VOLUME = "vsa_stopping_volume"
    CLIMACTIC_VOLUME = "vsa_climactic_volume"
    EFFORT_VS_RESULT = "vsa_effort_vs_result"
    HIDDEN_EFFORT = "vsa_hidden_effort"
    HIDDEN_WEAKNESS = "vsa_hidden_weakness"
    TRAP_MOVE = "vsa_trap_move"


class VSADirection(Enum):
    """Direction of a VSA pattern."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class VSABar:
    """
    Represents a price bar with VSA characteristics.
    
    Attributes:
        open: Open price
        high: High price
        low: Low price
        close: Close price
        volume: Volume
        spread: Price spread (high - low)
        close_location: Close location within the bar (0-1)
        volume_delta: Volume change from previous bar
        timestamp: Bar timestamp
        index: Index in the DataFrame
    """
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread: float
    close_location: float
    volume_delta: float
    timestamp: datetime
    index: int


@dataclass
class VSAPattern:
    """
    Represents a VSA pattern.
    
    Attributes:
        pattern_type: Type of the VSA pattern
        start_index: Starting index of the pattern
        end_index: Ending index of the pattern
        bars: List of VSA bars in the pattern
        direction: Direction of the pattern (bullish, bearish, neutral)
        strength: Strength of the pattern (0.0-1.0)
        volume_confirms: Whether volume confirms the pattern
        target_price: Target price projection
        stop_price: Suggested stop loss price
    """
    pattern_type: VSAPatternType
    start_index: int
    end_index: int
    bars: List[VSABar]
    direction: VSADirection
    strength: float
    volume_confirms: bool
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "pattern_type": self.pattern_type.value,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "direction": self.direction.value,
            "strength": self.strength,
            "volume_confirms": self.volume_confirms,
            "target_price": self.target_price,
            "stop_price": self.stop_price
        }