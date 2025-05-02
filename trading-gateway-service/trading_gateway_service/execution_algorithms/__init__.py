"""
Advanced Order Execution Algorithms Module.

This module provides implementations of advanced order execution algorithms
for the forex trading platform, including:
- Smart Order Routing (SOR)
- Time-Weighted Average Price (TWAP)
- Volume-Weighted Average Price (VWAP)
- Implementation Shortfall

These algorithms help optimize order execution by minimizing market impact,
reducing slippage, and improving overall execution quality.
"""

from .base_algorithm import BaseExecutionAlgorithm
from .smart_order_routing import SmartOrderRoutingAlgorithm
from .twap import TWAPAlgorithm
from .vwap import VWAPAlgorithm
from .implementation_shortfall import ImplementationShortfallAlgorithm

__all__ = [
    'BaseExecutionAlgorithm',
    'SmartOrderRoutingAlgorithm',
    'TWAPAlgorithm',
    'VWAPAlgorithm',
    'ImplementationShortfallAlgorithm',
]
