"""
Fibonacci Analysis Facade.

This module provides a facade for all Fibonacci analysis tools.
"""

from typing import Dict, Any, List, Optional, Tuple, Union

# Re-export all Fibonacci classes
from core.base_4 import TrendDirection, FibonacciBase
from core.retracements import FibonacciRetracement
from core.extensions import FibonacciExtension
from core.fans import FibonacciFan
from core.time_zones import FibonacciTimeZones
from core.circles import FibonacciCircles
from core.clusters import FibonacciClusters
from utils.utils_5 import (
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