"""
Advanced Chart Patterns Module.

This module provides implementations of common chart patterns for automated recognition.
This file is kept for backward compatibility and imports from the new module structure.
"""

from feature_store_service.indicators.chart_patterns.facade import (
    ChartPatternRecognizer,
    HarmonicPatternFinder,
    CandlestickPatterns
)
from feature_store_service.indicators.chart_patterns.base import PatternType

__all__ = [
    'ChartPatternRecognizer',
    'HarmonicPatternFinder',
    'CandlestickPatterns',
    'PatternType'
]
