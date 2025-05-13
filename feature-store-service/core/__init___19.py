"""
Degraded Mode Indicators

This package provides simplified versions of technical indicators that
can be used in degraded mode operation (high system load or low resource conditions).
These indicators sacrifice some accuracy for improved performance.
"""

from .base import DegradedModeIndicator, degraded_indicator
from .moving_averages import SimplifiedSMA, SimplifiedEMA
from .oscillators import SimplifiedRSI, SimplifiedMACD, SimplifiedStochastic
from .volatility import SimplifiedBollingerBands, SimplifiedATR

__all__ = [
    'DegradedModeIndicator',
    'degraded_indicator',
    'SimplifiedSMA',
    'SimplifiedEMA',
    'SimplifiedRSI',
    'SimplifiedMACD',
    'SimplifiedStochastic',
    'SimplifiedBollingerBands',
    'SimplifiedATR',
]
