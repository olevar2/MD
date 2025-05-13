"""
Elliott Wave Models Module.

This module defines the core data models and enums used in Elliott Wave analysis.
"""

from enum import Enum


class WaveType(Enum):
    """Types of Elliott Waves"""
    IMPULSE = "impulse"  # 5-wave pattern in direction of larger trend
    CORRECTION = "correction"  # 3-wave pattern against the larger trend
    DIAGONAL = "diagonal"  # 5-wave wedge-shaped pattern
    EXTENSION = "extension"  # Extended wave within an impulse wave
    TRIANGLE = "triangle"  # Complex correction in form of a triangle
    UNKNOWN = "unknown"


class WavePosition(Enum):
    """Position in the Elliott Wave sequence"""
    # Impulse wave positions
    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    
    # Corrective wave positions
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    
    # Sub-waves
    SUB_ONE = "i"
    SUB_TWO = "ii"
    SUB_THREE = "iii"
    SUB_FOUR = "iv"
    SUB_FIVE = "v"
    
    # Sub-corrective waves
    SUB_A = "a"
    SUB_B = "b"
    SUB_C = "c"
    SUB_D = "d"
    SUB_E = "e"


class WaveDegree(Enum):
    """Degrees of trend in Elliott Wave Theory"""
    GRAND_SUPERCYCLE = "Grand Supercycle"
    SUPERCYCLE = "Supercycle"
    CYCLE = "Cycle"
    PRIMARY = "Primary"
    INTERMEDIATE = "Intermediate"
    MINOR = "Minor"
    MINUTE = "Minute"
    MINUETTE = "Minuette"
    SUBMINUETTE = "Subminuette"