"""
Oscillator indicators for the forex trading platform.

This package provides standardized implementations of oscillator indicators
for use across all services in the platform.
"""

from common_lib.indicators.oscillators.rsi import RSI
from common_lib.indicators.oscillators.macd import MACD

__all__ = [
    'RSI',
    'MACD'
]