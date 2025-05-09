"""
Candlestick Patterns Package.

This package provides implementations of candlestick patterns like Doji,
Hammer, Engulfing, etc.
"""

from feature_store_service.indicators.chart_patterns.candlestick.base import BaseCandlestickPattern
from feature_store_service.indicators.chart_patterns.candlestick.doji import DojiPattern
from feature_store_service.indicators.chart_patterns.candlestick.hammer import HammerPattern
from feature_store_service.indicators.chart_patterns.candlestick.engulfing import EngulfingPattern

__all__ = [
    'BaseCandlestickPattern',
    'DojiPattern',
    'HammerPattern',
    'EngulfingPattern'
]
