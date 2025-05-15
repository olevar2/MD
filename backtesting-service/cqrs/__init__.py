"""
CQRS module for the Backtesting Service.

This module provides the CQRS implementation for the Backtesting Service.
"""

from backtesting_service.cqrs.commands import (
    RunBacktestCommand,
    OptimizeStrategyCommand,
    RunWalkForwardTestCommand,
    CancelBacktestCommand,
    DeleteBacktestCommand
)
from backtesting_service.cqrs.queries import (
    GetBacktestQuery,
    ListBacktestsQuery,
    GetOptimizationQuery,
    ListOptimizationsQuery,
    GetWalkForwardTestQuery,
    ListWalkForwardTestsQuery,
    ListStrategiesQuery
)
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
    'RunBacktestCommand',
    'OptimizeStrategyCommand',
    'RunWalkForwardTestCommand',
    'CancelBacktestCommand',
    'DeleteBacktestCommand',
    'GetBacktestQuery',
    'ListBacktestsQuery',
    'GetOptimizationQuery',
    'ListOptimizationsQuery',
    'GetWalkForwardTestQuery',
    'ListWalkForwardTestsQuery',
    'ListStrategiesQuery',
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