"""
Wyckoff Pattern Models Module.

This module defines the core data models and enums used in Wyckoff pattern analysis.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime


class WyckoffPatternType(Enum):
    """Types of Wyckoff patterns."""
    ACCUMULATION = "wyckoff_accumulation"
    DISTRIBUTION = "wyckoff_distribution"
    SPRING = "wyckoff_spring"
    UPTHRUST = "wyckoff_upthrust"


class WyckoffPhase(Enum):
    """Phases of Wyckoff accumulation and distribution."""
    # Accumulation phases
    PHASE_A_ACC = "phase_a_accumulation"  # Selling climax, automatic rally, secondary test
    PHASE_B_ACC = "phase_b_accumulation"  # Building cause, trading range
    PHASE_C_ACC = "phase_c_accumulation"  # Spring, test of support
    PHASE_D_ACC = "phase_d_accumulation"  # Sign of strength, last point of support
    PHASE_E_ACC = "phase_e_accumulation"  # Markup, breakout
    
    # Distribution phases
    PHASE_A_DIST = "phase_a_distribution"  # Preliminary supply, buying climax, automatic reaction
    PHASE_B_DIST = "phase_b_distribution"  # Secondary test, trading range
    PHASE_C_DIST = "phase_c_distribution"  # Upthrust, test of resistance
    PHASE_D_DIST = "phase_d_distribution"  # Sign of weakness, last point of supply
    PHASE_E_DIST = "phase_e_distribution"  # Markdown, breakdown


@dataclass
class WyckoffSchematic:
    """
    Represents a Wyckoff schematic pattern.
    
    Attributes:
        pattern_type: Type of the Wyckoff pattern
        start_index: Starting index of the pattern
        end_index: Ending index of the pattern
        current_phase: Current phase of the pattern
        phases: Dictionary mapping phases to their index ranges
        direction: Direction of the pattern (bullish or bearish)
        strength: Strength of the pattern (0.0-1.0)
        volume_confirms: Whether volume confirms the pattern
        target_price: Target price projection
        stop_price: Suggested stop loss price
    """
    pattern_type: WyckoffPatternType
    start_index: int
    end_index: int
    current_phase: WyckoffPhase
    phases: Dict[WyckoffPhase, Tuple[int, int]]
    direction: str
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
            "current_phase": self.current_phase.value,
            "phases": {phase.value: indices for phase, indices in self.phases.items()},
            "direction": self.direction,
            "strength": self.strength,
            "volume_confirms": self.volume_confirms,
            "target_price": self.target_price,
            "stop_price": self.stop_price
        }