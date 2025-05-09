"""
Volume Spread Analysis (VSA) Pattern Recognition Package.

This package provides pattern recognition capabilities for VSA methodology.
"""

from feature_store_service.indicators.advanced_patterns.vsa.models import (
    VSAPatternType,
    VSADirection,
    VSABar,
    VSAPattern
)
from feature_store_service.indicators.advanced_patterns.vsa.recognizer import VSAPatternRecognizer
from feature_store_service.indicators.advanced_patterns.vsa.utils import (
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