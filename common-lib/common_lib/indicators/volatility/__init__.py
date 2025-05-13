"""
Volatility indicators for the forex trading platform.

This package provides standardized implementations of volatility indicators
for use across all services in the platform.
"""

from common_lib.indicators.volatility.bollinger_bands import BollingerBands

__all__ = [
    'BollingerBands'
]