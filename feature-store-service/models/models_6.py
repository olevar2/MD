"""
Harmonic Pattern Models Module.

This module defines the core data models and enums used in harmonic pattern analysis.
"""

from enum import Enum
from typing import Dict, Any


class PatternType(Enum):
    """Enum for harmonic pattern types."""
    BAT = "bat"
    SHARK = "shark"
    CYPHER = "cypher"
    ABCD = "abcd"
    THREE_DRIVES = "three_drives"
    FIVE_ZERO = "five_zero"
    ALT_BAT = "alt_bat"
    DEEP_CRAB = "deep_crab"
    BUTTERFLY = "butterfly"
    GARTLEY = "gartley"
    CRAB = "crab"


def get_pattern_templates(tolerance: float = 0.05) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Get pattern ratio templates for harmonic patterns.
    
    Args:
        tolerance: Tolerance for pattern ratio matching (as decimal)
        
    Returns:
        Dictionary of pattern templates
    """
    return {
        "bat": {
            "XA_BC": {"ratio": 0.382, "tolerance": tolerance},
            "AB_XA": {"ratio": 0.382, "tolerance": tolerance},
            "BC_AB": {"ratio": 0.382, "tolerance": tolerance},
            "CD_BC": {"ratio": 1.618, "tolerance": tolerance * 2}
        },
        "shark": {
            "XA_BC": {"ratio": 0.886, "tolerance": tolerance},
            "AB_XA": {"ratio": 0.5, "tolerance": tolerance},
            "BC_AB": {"ratio": 1.13, "tolerance": tolerance},
            "CD_BC": {"ratio": 1.618, "tolerance": tolerance * 2}
        },
        "cypher": {
            "XA_BC": {"ratio": 0.382, "tolerance": tolerance},
            "AB_XA": {"ratio": 0.618, "tolerance": tolerance},
            "BC_AB": {"ratio": 1.13, "tolerance": tolerance},
            "CD_BC": {"ratio": 0.786, "tolerance": tolerance}
        },
        "abcd": {
            "AB_XA": {"ratio": 0.618, "tolerance": tolerance},
            "BC_AB": {"ratio": 0.618, "tolerance": tolerance},
            "CD_BC": {"ratio": 0.618, "tolerance": tolerance}
        },
        "three_drives": {
            "drive1_drive2": {"ratio": 0.618, "tolerance": tolerance},
            "drive2_drive3": {"ratio": 1.27, "tolerance": tolerance},
            "correction1_drive1": {"ratio": 0.618, "tolerance": tolerance},
            "correction2_drive2": {"ratio": 0.618, "tolerance": tolerance}
        },
        "five_zero": {
            "XA_BC": {"ratio": 1.618, "tolerance": tolerance * 2},
            "AB_XA": {"ratio": 0.5, "tolerance": tolerance},
            "BC_AB": {"ratio": 1.618, "tolerance": tolerance * 2},
            "CD_BC": {"ratio": 0.5, "tolerance": tolerance}
        },
        "alt_bat": {
            "XA_BC": {"ratio": 0.382, "tolerance": tolerance},
            "AB_XA": {"ratio": 0.382, "tolerance": tolerance},
            "BC_AB": {"ratio": 2.0, "tolerance": tolerance * 2},
            "CD_BC": {"ratio": 1.27, "tolerance": tolerance}
        },
        "deep_crab": {
            "XA_BC": {"ratio": 0.886, "tolerance": tolerance},
            "AB_XA": {"ratio": 0.382, "tolerance": tolerance},
            "BC_AB": {"ratio": 2.618, "tolerance": tolerance * 2},
            "CD_BC": {"ratio": 1.618, "tolerance": tolerance * 2}
        },
        "butterfly": {
            "XA_BC": {"ratio": 0.786, "tolerance": tolerance},
            "AB_XA": {"ratio": 0.786, "tolerance": tolerance},
            "BC_AB": {"ratio": 0.382, "tolerance": tolerance},
            "CD_BC": {"ratio": 1.618, "tolerance": tolerance * 2}
        },
        "gartley": {
            "XA_BC": {"ratio": 0.618, "tolerance": tolerance},
            "AB_XA": {"ratio": 0.618, "tolerance": tolerance},
            "BC_AB": {"ratio": 0.382, "tolerance": tolerance},
            "CD_BC": {"ratio": 1.27, "tolerance": tolerance}
        },
        "crab": {
            "XA_BC": {"ratio": 0.382, "tolerance": tolerance},
            "AB_XA": {"ratio": 0.382, "tolerance": tolerance},
            "BC_AB": {"ratio": 0.618, "tolerance": tolerance},
            "CD_BC": {"ratio": 3.14, "tolerance": tolerance * 2}
        }
    }


def get_fibonacci_ratios() -> Dict[str, float]:
    """
    Get Fibonacci ratios used in harmonic patterns.
    
    Returns:
        Dictionary of Fibonacci ratios
    """
    return {
        "0.382": 0.382,
        "0.5": 0.5,
        "0.618": 0.618,
        "0.707": 0.707,
        "0.786": 0.786,
        "0.886": 0.886,
        "1.0": 1.0,
        "1.13": 1.13,
        "1.27": 1.27,
        "1.414": 1.414,
        "1.618": 1.618,
        "2.0": 2.0,
        "2.24": 2.24,
        "2.618": 2.618,
        "3.14": 3.14,
        "3.618": 3.618
    }