"""
Fibonacci Analysis Facade.

This module provides a facade for all Fibonacci analysis tools.
"""

from typing import Dict, Any, List, Optional, Tuple, Union

# Re-export all Fibonacci classes
from feature_store_service.indicators.fibonacci.base import TrendDirection, FibonacciBase
from feature_store_service.indicators.fibonacci.retracements import FibonacciRetracement
from feature_store_service.indicators.fibonacci.extensions import FibonacciExtension
from feature_store_service.indicators.fibonacci.fans import FibonacciFan
from feature_store_service.indicators.fibonacci.time_zones import FibonacciTimeZones
from feature_store_service.indicators.fibonacci.circles import FibonacciCircles
from feature_store_service.indicators.fibonacci.clusters import FibonacciClusters
from feature_store_service.indicators.fibonacci.utils import (
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