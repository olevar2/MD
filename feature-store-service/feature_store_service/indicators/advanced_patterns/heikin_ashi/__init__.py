"""
Heikin-Ashi Pattern Recognition Package.

This package provides pattern recognition capabilities for Heikin-Ashi candlesticks.
"""

from feature_store_service.indicators.advanced_patterns.heikin_ashi.models import (
    HeikinAshiPatternType,
    HeikinAshiTrendType,
    HeikinAshiCandle,
    HeikinAshiPattern
)
from feature_store_service.indicators.advanced_patterns.heikin_ashi.recognizer import HeikinAshiPatternRecognizer
from feature_store_service.indicators.advanced_patterns.heikin_ashi.utils import (
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