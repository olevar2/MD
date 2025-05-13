"""
Moving Averages Module.

This module provides implementations of various moving average indicators.
It now uses the standardized implementations from common-lib via adapters.
"""

# Import the adapter implementations
from adapters.indicator_adapter import (
    SimpleMovingAverage,
    ExponentialMovingAverage,
    WeightedMovingAverage
)

# Re-export the indicator classes
__all__ = [
    'SimpleMovingAverage',
    'ExponentialMovingAverage',
    'WeightedMovingAverage'
]
