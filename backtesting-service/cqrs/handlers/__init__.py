"""
CQRS handlers for the Backtesting Service.

This module provides the CQRS handlers for the Backtesting Service.
"""

from backtesting_service.cqrs.handlers.command_handlers import (
    RunBacktestCommandHandler,
    OptimizeStrategyCommandHandler,
    RunWalkForwardTestCommandHandler,
    CancelBacktestCommandHandler,
    DeleteBacktestCommandHandler
)
from backtesting_service.cqrs.handlers.query_handlers import (
    GetBacktestQueryHandler,
    ListBacktestsQueryHandler,
    GetOptimizationQueryHandler,
    ListOptimizationsQueryHandler,
    GetWalkForwardTestQueryHandler,
    ListWalkForwardTestsQueryHandler,
    ListStrategiesQueryHandler
)

__all__ = [
    'RunBacktestCommandHandler',
    'OptimizeStrategyCommandHandler',
    'RunWalkForwardTestCommandHandler',
    'CancelBacktestCommandHandler',
    'DeleteBacktestCommandHandler',
    'GetBacktestQueryHandler',
    'ListBacktestsQueryHandler',
    'GetOptimizationQueryHandler',
    'ListOptimizationsQueryHandler',
    'GetWalkForwardTestQueryHandler',
    'ListWalkForwardTestsQueryHandler',
    'ListStrategiesQueryHandler'
]