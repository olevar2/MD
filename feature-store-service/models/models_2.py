"""
Market Profile Models Module.

This module defines the core data models and enums used in Market Profile analysis.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass
from datetime import datetime, time


class MarketProfilePatternType(Enum):
    """Types of Market Profile patterns."""
    VALUE_AREA = "market_profile_value_area"
    SINGLE_PRINT = "market_profile_single_print"
    IB_BREAKOUT = "market_profile_ib_breakout"
    POC_REJECTION = "market_profile_poc_rejection"
    BALANCE_IMBALANCE = "market_profile_balance_imbalance"


class MarketProfileDirection(Enum):
    """Direction of a Market Profile pattern."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class PriceLevel:
    """
    Represents a price level in the Market Profile.
    
    Attributes:
        price: The price level
        tpo_count: Time Price Opportunity (TPO) count at this level
        volume: Volume traded at this level
        timestamps: Set of timestamps when price visited this level
    """
    price: float
    tpo_count: int
    volume: float
    timestamps: Set[datetime]


@dataclass
class MarketProfileStructure:
    """
    Represents the Market Profile structure for a session.
    
    Attributes:
        date: Date of the profile
        price_levels: Dictionary mapping price levels to PriceLevel objects
        poc_price: Point of Control price (highest volume/TPO level)
        value_area_high: Value Area High price
        value_area_low: Value Area Low price
        initial_balance_high: Initial Balance High price
        initial_balance_low: Initial Balance Low price
        single_prints: List of single print price levels
    """
    date: datetime
    price_levels: Dict[float, PriceLevel]
    poc_price: float
    value_area_high: float
    value_area_low: float
    initial_balance_high: float
    initial_balance_low: float
    single_prints: List[float]


@dataclass
class MarketProfilePattern:
    """
    Represents a Market Profile pattern.
    
    Attributes:
        pattern_type: Type of the Market Profile pattern
        start_index: Starting index of the pattern
        end_index: Ending index of the pattern
        profile: MarketProfileStructure associated with the pattern
        direction: Direction of the pattern (bullish, bearish, neutral)
        strength: Strength of the pattern (0.0-1.0)
        target_price: Target price projection
        stop_price: Suggested stop loss price
    """
    pattern_type: MarketProfilePatternType
    start_index: int
    end_index: int
    profile: MarketProfileStructure
    direction: MarketProfileDirection
    strength: float
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
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "poc_price": self.profile.poc_price,
            "value_area_high": self.profile.value_area_high,
            "value_area_low": self.profile.value_area_low
        }