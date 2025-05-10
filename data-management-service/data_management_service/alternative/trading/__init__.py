"""
Trading Module for Alternative Data.

This module provides functionality for generating trading signals from alternative data.
"""

from data_management_service.alternative.trading.signal_generator import (
    BaseTradingSignalGenerator,
    NewsTradingSignalGenerator,
    EconomicTradingSignalGenerator
)

__all__ = [
    "BaseTradingSignalGenerator",
    "NewsTradingSignalGenerator",
    "EconomicTradingSignalGenerator"
]
