"""
Renko Pattern Models Module.

This module defines the core data models and enums used in Renko pattern analysis.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime


class RenkoDirection(Enum):
    """Direction of a Renko brick."""
    UP = "up"
    DOWN = "down"


class RenkoPatternType(Enum):
    """Types of Renko patterns."""
    REVERSAL = "renko_reversal"
    BREAKOUT = "renko_breakout"
    DOUBLE_TOP = "renko_double_top"
    DOUBLE_BOTTOM = "renko_double_bottom"


@dataclass
class RenkoBrick:
    """
    Represents a single Renko brick.
    
    Attributes:
        direction: Direction of the brick (up or down)
        open_price: Opening price of the brick
        close_price: Closing price of the brick
        open_time: Opening time of the brick
        close_time: Closing time of the brick
        index: Index of the brick in the Renko chart
    """
    direction: RenkoDirection
    open_price: float
    close_price: float
    open_time: Optional[datetime] = None
    close_time: Optional[datetime] = None
    index: Optional[int] = None
    
    @property
    def size(self) -> float:
        """Size of the brick."""
        return abs(self.close_price - self.open_price)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "direction": self.direction.value,
            "open_price": self.open_price,
            "close_price": self.close_price,
            "open_time": self.open_time.isoformat() if self.open_time else None,
            "close_time": self.close_time.isoformat() if self.close_time else None,
            "index": self.index,
            "size": self.size
        }


@dataclass
class RenkoPattern:
    """
    Represents a detected Renko pattern.
    
    Attributes:
        pattern_type: Type of the pattern
        start_index: Starting index of the pattern
        end_index: Ending index of the pattern
        bricks: List of bricks forming the pattern
        direction: Direction of the pattern (bullish or bearish)
        strength: Strength of the pattern (0.0-1.0)
        target_price: Target price projection
        stop_price: Suggested stop loss price
    """
    pattern_type: RenkoPatternType
    start_index: int
    end_index: int
    bricks: List[RenkoBrick]
    direction: str
    strength: float
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "pattern_type": self.pattern_type.value,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "bricks": [brick.to_dict() for brick in self.bricks],
            "direction": self.direction,
            "strength": self.strength,
            "target_price": self.target_price,
            "stop_price": self.stop_price
        }