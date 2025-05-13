"""
Strategy module for the Strategy Execution Engine

This module contains the base strategy interface, strategy loader,
and concrete strategy implementations.
"""

from .base_strategy import BaseStrategy
from .strategy_loader import StrategyLoader
from .ma_crossover_strategy import MACrossoverStrategy
