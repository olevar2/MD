"""
Elliott Wave Analysis Module

This module provides implementation of Elliott Wave pattern detection and analysis,
including wave identification, labeling, and related Fibonacci measurements.

This is a facade that re-exports the components from the elliott_wave package.
"""

# Re-export all components from the elliott_wave package
from analysis_engine.analysis.advanced_ta.elliott_wave.models import (
    WaveType, WavePosition, WaveDegree
)
from analysis_engine.analysis.advanced_ta.elliott_wave.pattern import ElliottWavePattern
from analysis_engine.analysis.advanced_ta.elliott_wave.analyzer import ElliottWaveAnalyzer
from analysis_engine.analysis.advanced_ta.elliott_wave.counter import ElliottWaveCounter
from analysis_engine.analysis.advanced_ta.elliott_wave.utils import (
    detect_zigzag_points, detect_swing_points, calculate_wave_sharpness
)
from analysis_engine.analysis.advanced_ta.elliott_wave.fibonacci import (
    calculate_impulse_fibonacci_levels, calculate_correction_fibonacci_levels, calculate_fibonacci_levels
)
from analysis_engine.analysis.advanced_ta.elliott_wave.validators import (
    validate_elliott_rules, check_if_extended, check_if_diagonal, map_confidence_to_level
)

# Define __all__ to control what gets imported with "from elliott_wave import *"
__all__ = [
    'WaveType', 'WavePosition', 'WaveDegree',
    'ElliottWavePattern', 'ElliottWaveAnalyzer', 'ElliottWaveCounter',
    'detect_zigzag_points', 'detect_swing_points', 'calculate_wave_sharpness',
    'calculate_impulse_fibonacci_levels', 'calculate_correction_fibonacci_levels', 'calculate_fibonacci_levels',
    'validate_elliott_rules', 'check_if_extended', 'check_if_diagonal', 'map_confidence_to_level'
]