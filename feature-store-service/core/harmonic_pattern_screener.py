"""
Harmonic Pattern Screener.

This module implements a comprehensive harmonic pattern detection system
that checks for multiple harmonic patterns and provides a pattern evaluation system.

This is a facade that re-exports the components from the harmonic_patterns package.
"""

# Re-export the HarmonicPatternScreener class
from feature_store_service.indicators.harmonic_patterns import (
    PatternType, HarmonicPatternScreener
)

# Define __all__ to control what gets imported with "from harmonic_pattern_screener import *"
__all__ = [
    'PatternType',
    'HarmonicPatternScreener'
]