"""
Interfaces package for the Trading Gateway Service.

This package contains interface definitions for the Trading Gateway Service.
"""

from .broker_adapter import (
    BrokerAdapter,
    OrderType,
    OrderDirection,
    OrderStatus,
    OrderRequest,
    ExecutionReport
)

__all__ = [
    "BrokerAdapter",
    "OrderType",
    "OrderDirection",
    "OrderStatus",
    "OrderRequest",
    "ExecutionReport"
]
