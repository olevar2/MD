"""
Volatility Indicators Module.

This module provides implementations of various volatility-based indicators.
It now uses the standardized implementations from common-lib via adapters.
"""

# Import the adapter implementations
from adapters.indicator_adapter import (
    BollingerBands,
    KeltnerChannels,
    DonchianChannels,
    AverageTrueRange,
    PriceEnvelopes,
    HistoricalVolatility
)

# Import utility functions from the volatility package
from core.vix import VIXFixIndicator
from utils.utils_7 import (
    calculate_true_range,
    calculate_volatility_ratio,
    calculate_volatility_percentile,
    calculate_volatility_breakout,
    calculate_volatility_regime
)

# Define __all__ to control what gets imported with "from volatility import *"
__all__ = [
    'BollingerBands',
    'KeltnerChannels',
    'DonchianChannels',
    'AverageTrueRange',
    'PriceEnvelopes',
    'VIXFixIndicator',
    'HistoricalVolatility',
    'calculate_true_range',
    'calculate_volatility_ratio',
    'calculate_volatility_percentile',
    'calculate_volatility_breakout',
    'calculate_volatility_regime'
]