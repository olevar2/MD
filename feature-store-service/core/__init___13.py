"""
Volume Spread Analysis (VSA) Pattern Recognition Package.

This package provides pattern recognition capabilities for VSA methodology.
"""

from models.models_4 import (
    VSAPatternType,
    VSADirection,
    VSABar,
    VSAPattern
)
from core.recognizer_3 import VSAPatternRecognizer
from utils.utils_3 import (
    prepare_vsa_data,
    extract_vsa_bars,
    detect_no_demand,
    detect_no_supply,
    detect_stopping_volume,
    detect_climactic_volume,
    detect_effort_vs_result,
    detect_trap_move
)

__all__ = [
    "VSAPatternType",
    "VSADirection",
    "VSABar",
    "VSAPattern",
    "VSAPatternRecognizer",
    "prepare_vsa_data",
    "extract_vsa_bars",
    "detect_no_demand",
    "detect_no_supply",
    "detect_stopping_volume",
    "detect_climactic_volume",
    "detect_effort_vs_result",
    "detect_trap_move"
]