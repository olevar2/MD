"""
Indicators package for common-lib.

This package provides interfaces and utilities for technical indicators
and advanced analysis components, helping to break circular dependencies
between services.
"""

from .indicator_interfaces import (
    IndicatorCategory,
    IBaseIndicator,
    IAdvancedIndicator,
    IPatternRecognizer,
    IFibonacciAnalyzer,
    IIndicatorRegistry,
    IIndicatorAdapter
)

from .fibonacci_interfaces import (
    TrendDirectionType,
    IFibonacciBase,
    IFibonacciRetracement,
    IFibonacciExtension,
    IFibonacciFan,
    IFibonacciTimeZones,
    IFibonacciCircles,
    IFibonacciClusters,
    IFibonacciUtils
)

__all__ = [
    'IndicatorCategory',
    'IBaseIndicator',
    'IAdvancedIndicator',
    'IPatternRecognizer',
    'IFibonacciAnalyzer',
    'IIndicatorRegistry',
    'IIndicatorAdapter',
    # Fibonacci interfaces
    'TrendDirectionType',
    'IFibonacciBase',
    'IFibonacciRetracement',
    'IFibonacciExtension',
    'IFibonacciFan',
    'IFibonacciTimeZones',
    'IFibonacciCircles',
    'IFibonacciClusters',
    'IFibonacciUtils'
]