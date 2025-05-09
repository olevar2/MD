"""
Ichimoku Pattern Models Module.

This module defines the core data models and enums used in Ichimoku pattern analysis.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime


class IchimokuPatternType(Enum):
    """Types of Ichimoku patterns."""
    TK_CROSS = "ichimoku_tk_cross"
    KUMO_BREAKOUT = "ichimoku_kumo_breakout"
    KUMO_TWIST = "ichimoku_kumo_twist"
    CHIKOU_CROSS = "ichimoku_chikou_cross"


@dataclass
class IchimokuComponents:
    """
    Represents the components of an Ichimoku Cloud.
    
    Attributes:
        tenkan_sen: Conversion line (9-period)
        kijun_sen: Base line (26-period)
        senkou_span_a: Leading span A (midpoint of tenkan and kijun projected 26 periods ahead)
        senkou_span_b: Leading span B (52-period midpoint projected 26 periods ahead)
        chikou_span: Lagging span (current close projected 26 periods back)
    """
    tenkan_sen: float
    kijun_sen: float
    senkou_span_a: float
    senkou_span_b: float
    chikou_span: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation."""
        return {
            "tenkan_sen": self.tenkan_sen,
            "kijun_sen": self.kijun_sen,
            "senkou_span_a": self.senkou_span_a,
            "senkou_span_b": self.senkou_span_b,
            "chikou_span": self.chikou_span
        }


@dataclass
class IchimokuPattern:
    """
    Represents a detected Ichimoku pattern.
    
    Attributes:
        pattern_type: Type of the pattern
        start_index: Starting index of the pattern
        end_index: Ending index of the pattern
        direction: Direction of the pattern (bullish or bearish)
        strength: Strength of the pattern (0.0-1.0)
        components: Ichimoku components at the pattern
        target_price: Target price projection
        stop_price: Suggested stop loss price
    """
    pattern_type: IchimokuPatternType
    start_index: int
    end_index: int
    direction: str
    strength: float
    components: IchimokuComponents
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "pattern_type": self.pattern_type.value,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "direction": self.direction,
            "strength": self.strength,
            "components": self.components.to_dict(),
            "target_price": self.target_price,
            "stop_price": self.stop_price
        }