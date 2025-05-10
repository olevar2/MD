"""
Volatility Indicators Module.

This module provides implementations of various volatility-based indicators.

This is a facade that re-exports the components from the volatility package.
"""

# Re-export all components from the volatility package
from feature_store_service.indicators.volatility.bands import (
    BollingerBands, KeltnerChannels, DonchianChannels
)
from feature_store_service.indicators.volatility.range import AverageTrueRange
from feature_store_service.indicators.volatility.envelopes import PriceEnvelopes
from feature_store_service.indicators.volatility.vix import VIXFixIndicator
from feature_store_service.indicators.volatility.historical import HistoricalVolatility
from feature_store_service.indicators.volatility.utils import (
    calculate_true_range, calculate_volatility_ratio, calculate_volatility_percentile,
    calculate_volatility_breakout, calculate_volatility_regime
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