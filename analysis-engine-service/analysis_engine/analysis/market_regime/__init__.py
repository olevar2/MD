"""
Market Regime Analysis Package

This package provides tools for detecting and classifying market regimes
based on price action, volatility, and trend characteristics.
"""

from .models import (
    MarketRegimeType,
    MarketRegimeResult,
    RegimeChangeResult,
    VolatilityState,
    TrendState
)
from .classifier import MarketRegimeClassifier
from .detector import MarketRegimeDetector
from .analyzer import MarketRegimeAnalyzer

__all__ = [
    'MarketRegimeType',
    'MarketRegimeResult',
    'RegimeChangeResult',
    'VolatilityState',
    'TrendState',
    'MarketRegimeClassifier',
    'MarketRegimeDetector',
    'MarketRegimeAnalyzer'
]