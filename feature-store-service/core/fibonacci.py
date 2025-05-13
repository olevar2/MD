"""
Fibonacci Analysis Module.

This module provides implementations of Fibonacci-based technical analysis tools
including retracement levels, extension levels, fans, and time zones.

Note: This module has been refactored into a modular package structure.
      It now imports from the fibonacci/ package to maintain backward compatibility.
      See fibonacci/README.md for more information.
"""

# Re-export all Fibonacci classes from the new package
from core.fibonacci import (
    TrendDirection,
    FibonacciBase,
    FibonacciRetracement,
    FibonacciExtension,
    FibonacciFan,
    FibonacciTimeZones,
    FibonacciCircles,
    FibonacciClusters,
    generate_fibonacci_sequence,
    fibonacci_ratios,
    calculate_fibonacci_retracement_levels,
    calculate_fibonacci_extension_levels,
    format_fibonacci_level,
    is_golden_ratio,
    is_fibonacci_ratio
)

# Define all exports
__all__ = [
    'TrendDirection',
    'FibonacciBase',
    'FibonacciRetracement',
    'FibonacciExtension',
    'FibonacciFan',
    'FibonacciTimeZones',
    'FibonacciCircles',
    'FibonacciClusters',
    'generate_fibonacci_sequence',
    'fibonacci_ratios',
    'calculate_fibonacci_retracement_levels',
    'calculate_fibonacci_extension_levels',
    'format_fibonacci_level',
    'is_golden_ratio',
    'is_fibonacci_ratio'
]