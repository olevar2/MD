"""
Feedback Utilities Package

This package provides utility functions for feedback analysis.
"""

from .statistics import (
    calculate_basic_statistics,
    calculate_percentiles,
    calculate_moving_average,
    calculate_correlation,
    detect_outliers
)

__all__ = [
    'calculate_basic_statistics',
    'calculate_percentiles',
    'calculate_moving_average',
    'calculate_correlation',
    'detect_outliers'
]