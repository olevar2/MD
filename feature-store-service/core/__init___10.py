"""
Heikin-Ashi Pattern Recognition Package.

This package provides pattern recognition capabilities for Heikin-Ashi candlesticks.
"""

from models.models import (
    HeikinAshiPatternType,
    HeikinAshiTrendType,
    HeikinAshiCandle,
    HeikinAshiPattern
)
from core.recognizer import HeikinAshiPatternRecognizer
from utils.utils import (
    calculate_heikin_ashi,
    extract_heikin_ashi_candles,
    detect_heikin_ashi_reversal,
    detect_heikin_ashi_continuation,
    detect_heikin_ashi_strong_trend,
    detect_heikin_ashi_weak_trend
)

__all__ = [
    "HeikinAshiPatternType",
    "HeikinAshiTrendType",
    "HeikinAshiCandle",
    "HeikinAshiPattern",
    "HeikinAshiPatternRecognizer",
    "calculate_heikin_ashi",
    "extract_heikin_ashi_candles",
    "detect_heikin_ashi_reversal",
    "detect_heikin_ashi_continuation",
    "detect_heikin_ashi_strong_trend",
    "detect_heikin_ashi_weak_trend"
]