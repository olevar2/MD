"""
Backtesting Package for Strategy Execution Engine

This package contains backtesting functionality for the Strategy Execution Engine,
including the backtester, backtest engine, and reporting.
"""

from .backtester import backtester, Backtester

__all__ = [
    "backtester",
    "Backtester"
]
