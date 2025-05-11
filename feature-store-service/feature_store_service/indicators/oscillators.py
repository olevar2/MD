"""
Oscillators Module.

This module provides implementations of various oscillator-type indicators.
It now uses the standardized implementations from common-lib via adapters.
"""

# Import the adapter implementations
from feature_store_service.indicators.indicator_adapter import (
    RelativeStrengthIndex,
    Stochastic,
    MACD,
    CommodityChannelIndex,
    WilliamsR,
    RateOfChange
)

# Re-export the indicator classes
__all__ = [
    'RelativeStrengthIndex',
    'Stochastic',
    'MACD',
    'CommodityChannelIndex',
    'WilliamsR',
    'RateOfChange'
]
