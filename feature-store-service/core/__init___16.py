"""
Candlestick Patterns Package.

This package provides implementations of candlestick patterns like Doji,
Hammer, Engulfing, etc.
"""

from core.base_2 import BaseCandlestickPattern
from core.doji import DojiPattern
from core.hammer import HammerPattern
from core.engulfing import EngulfingPattern

__all__ = [
    'BaseCandlestickPattern',
    'DojiPattern',
    'HammerPattern',
    'EngulfingPattern'
]
