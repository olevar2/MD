"""
Dependency injection module for the Backtesting Service.

This module provides the dependency injection for the Backtesting Service.
"""
import logging
from typing import Optional

from common_lib.cqrs.commands import CommandBus
from common_lib.cqrs.queries import QueryBus
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
from backtesting_service.repositories.read_repositories import (
    BacktestReadRepository,
    OptimizationReadRepository,
    WalkForwardReadRepository
)
from backtesting_service.repositories.write_repositories import (
    BacktestWriteRepository,
    OptimizationWriteRepository,
    WalkForwardWriteRepository
)
from backtesting_service.services.backtest_service import BacktestService
from backtesting_service.services.strategy_service import StrategyService

logger = logging.getLogger(__name__)

# Singleton instances
_command_bus: Optional[CommandBus] = None
_query_bus: Optional[QueryBus] = None


def get_command_bus() -> CommandBus:
    """
    Get the command bus.
    
    Returns:
        The command bus
    """
    global _command_bus
    
    if _command_bus is None:
        _command_bus = CommandBus()
        
        # Create repositories
        backtest_write_repository = BacktestWriteRepository()
        optimization_write_repository = OptimizationWriteRepository()
        walk_forward_write_repository = WalkForwardWriteRepository()
        
        # Create services
        backtest_service = BacktestService()
        
        # Create command handlers
        run_backtest_handler = RunBacktestCommandHandler(
            backtest_service=backtest_service,
            repository=backtest_write_repository
        )
        optimize_strategy_handler = OptimizeStrategyCommandHandler(
            backtest_service=backtest_service,
            repository=optimization_write_repository
        )
        run_walk_forward_test_handler = RunWalkForwardTestCommandHandler(
            backtest_service=backtest_service,
            repository=walk_forward_write_repository
        )
        cancel_backtest_handler = CancelBacktestCommandHandler(
            backtest_service=backtest_service,
            repository=backtest_write_repository
        )
        delete_backtest_handler = DeleteBacktestCommandHandler(
            repository=backtest_write_repository
        )
        
        # Register command handlers
        _command_bus.register_handler(RunBacktestCommand, run_backtest_handler)
        _command_bus.register_handler(OptimizeStrategyCommand, optimize_strategy_handler)
        _command_bus.register_handler(RunWalkForwardTestCommand, run_walk_forward_test_handler)
        _command_bus.register_handler(CancelBacktestCommand, cancel_backtest_handler)
        _command_bus.register_handler(DeleteBacktestCommand, delete_backtest_handler)
    
    return _command_bus


def get_query_bus() -> QueryBus:
    """
    Get the query bus.
    
    Returns:
        The query bus
    """
    global _query_bus
    
    if _query_bus is None:
        _query_bus = QueryBus()
        
        # Create repositories
        backtest_read_repository = BacktestReadRepository()
        optimization_read_repository = OptimizationReadRepository()
        walk_forward_read_repository = WalkForwardReadRepository()
        
        # Create services
        strategy_service = StrategyService()
        
        # Create query handlers
        get_backtest_handler = GetBacktestQueryHandler(
            repository=backtest_read_repository
        )
        list_backtests_handler = ListBacktestsQueryHandler(
            repository=backtest_read_repository
        )
        get_optimization_handler = GetOptimizationQueryHandler(
            repository=optimization_read_repository
        )
        list_optimizations_handler = ListOptimizationsQueryHandler(
            repository=optimization_read_repository
        )
        get_walk_forward_test_handler = GetWalkForwardTestQueryHandler(
            repository=walk_forward_read_repository
        )
        list_walk_forward_tests_handler = ListWalkForwardTestsQueryHandler(
            repository=walk_forward_read_repository
        )
        list_strategies_handler = ListStrategiesQueryHandler(
            strategy_service=strategy_service
        )
        
        # Register query handlers
        _query_bus.register_handler(GetBacktestQuery, get_backtest_handler)
        _query_bus.register_handler(ListBacktestsQuery, list_backtests_handler)
        _query_bus.register_handler(GetOptimizationQuery, get_optimization_handler)
        _query_bus.register_handler(ListOptimizationsQuery, list_optimizations_handler)
        _query_bus.register_handler(GetWalkForwardTestQuery, get_walk_forward_test_handler)
        _query_bus.register_handler(ListWalkForwardTestsQuery, list_walk_forward_tests_handler)
        _query_bus.register_handler(ListStrategiesQuery, list_strategies_handler)
    
    return _query_bus