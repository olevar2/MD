"""
Advanced Chart Patterns Module.

This module provides implementations of common chart patterns for automated recognition.
This file is kept for backward compatibility and imports from the new module structure.
"""

from core.facade_1 import (
    ChartPatternRecognizer,
    HarmonicPatternFinder,
    CandlestickPatterns
)
from core.base_1 import PatternType

__all__ = [
    'ChartPatternRecognizer',
    'HarmonicPatternFinder',
    'CandlestickPatterns',
    'PatternType'
]
