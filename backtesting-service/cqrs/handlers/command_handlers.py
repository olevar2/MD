"""
Command handlers for the Backtesting Service.

This module provides the command handlers for the Backtesting Service.
"""
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from common_lib.cqrs.commands import CommandHandler
from backtesting_service.cqrs.commands import (
    RunBacktestCommand,
    OptimizeStrategyCommand,
    RunWalkForwardTestCommand,
    CancelBacktestCommand,
    DeleteBacktestCommand
)
from backtesting_service.models.backtest_models import (
    BacktestResult,
    BacktestStatus,
    OptimizationResult,
    WalkForwardTestResult
)
from backtesting_service.repositories.write_repositories import (
    BacktestWriteRepository,
    OptimizationWriteRepository,
    WalkForwardWriteRepository
)
from backtesting_service.services.backtest_service import BacktestService

logger = logging.getLogger(__name__)


class RunBacktestCommandHandler(CommandHandler):
    """Handler for RunBacktestCommand."""
    
    def __init__(
        self,
        backtest_service: BacktestService,
        repository: BacktestWriteRepository
    ):
        """
        Initialize the handler.
        
        Args:
            backtest_service: Backtest service
            repository: Backtest write repository
        """
        self.backtest_service = backtest_service
        self.repository = repository
    
    async def handle(self, command: RunBacktestCommand) -> str:
        """
        Handle the command.
        
        Args:
            command: The command
            
        Returns:
            The ID of the backtest
        """
        logger.info(f"Handling RunBacktestCommand: {command}")
        
        # Generate a unique ID for the backtest
        backtest_id = str(uuid.uuid4())
        
        # Create initial backtest result
        backtest_result = BacktestResult(
            backtest_id=backtest_id,
            strategy_id=command.strategy_id,
            symbol=command.symbol,
            timeframe=command.timeframe,
            start_date=command.start_date,
            end_date=command.end_date,
            initial_balance=command.initial_balance,
            parameters=command.parameters or {},
            status=BacktestStatus.PENDING,
            start_time=datetime.now()
        )
        
        # Save initial backtest result
        await self.repository.add(backtest_result)
        
        # Start backtest in background
        self.backtest_service.start_backtest(backtest_id, command)
        
        return backtest_id


class OptimizeStrategyCommandHandler(CommandHandler):
    """Handler for OptimizeStrategyCommand."""
    
    def __init__(
        self,
        backtest_service: BacktestService,
        repository: OptimizationWriteRepository
    ):
        """
        Initialize the handler.
        
        Args:
            backtest_service: Backtest service
            repository: Optimization write repository
        """
        self.backtest_service = backtest_service
        self.repository = repository
    
    async def handle(self, command: OptimizeStrategyCommand) -> str:
        """
        Handle the command.
        
        Args:
            command: The command
            
        Returns:
            The ID of the optimization
        """
        logger.info(f"Handling OptimizeStrategyCommand: {command}")
        
        # Generate a unique ID for the optimization
        optimization_id = str(uuid.uuid4())
        
        # Create initial optimization result
        optimization_result = OptimizationResult(
            optimization_id=optimization_id,
            strategy_id=command.strategy_id,
            symbol=command.symbol,
            timeframe=command.timeframe,
            start_date=command.start_date,
            end_date=command.end_date,
            initial_balance=command.initial_balance,
            parameters_to_optimize=command.parameters_to_optimize,
            optimization_metric=command.optimization_metric,
            num_iterations=command.num_iterations,
            status="pending",
            start_time=datetime.now()
        )
        
        # Save initial optimization result
        await self.repository.add(optimization_result)
        
        # Start optimization in background
        self.backtest_service.start_optimization(optimization_id, command)
        
        return optimization_id


class RunWalkForwardTestCommandHandler(CommandHandler):
    """Handler for RunWalkForwardTestCommand."""
    
    def __init__(
        self,
        backtest_service: BacktestService,
        repository: WalkForwardWriteRepository
    ):
        """
        Initialize the handler.
        
        Args:
            backtest_service: Backtest service
            repository: Walk-forward test write repository
        """
        self.backtest_service = backtest_service
        self.repository = repository
    
    async def handle(self, command: RunWalkForwardTestCommand) -> str:
        """
        Handle the command.
        
        Args:
            command: The command
            
        Returns:
            The ID of the walk-forward test
        """
        logger.info(f"Handling RunWalkForwardTestCommand: {command}")
        
        # Generate a unique ID for the walk-forward test
        test_id = str(uuid.uuid4())
        
        # Create initial walk-forward test result
        test_result = WalkForwardTestResult(
            test_id=test_id,
            strategy_id=command.strategy_id,
            symbol=command.symbol,
            timeframe=command.timeframe,
            start_date=command.start_date,
            end_date=command.end_date,
            initial_balance=command.initial_balance,
            parameters=command.parameters or {},
            optimization_window=command.optimization_window,
            test_window=command.test_window,
            optimization_metric=command.optimization_metric,
            parameters_to_optimize=command.parameters_to_optimize,
            status="pending",
            start_time=datetime.now()
        )
        
        # Save initial walk-forward test result
        await self.repository.add(test_result)
        
        # Start walk-forward test in background
        self.backtest_service.start_walk_forward_test(test_id, command)
        
        return test_id


class CancelBacktestCommandHandler(CommandHandler):
    """Handler for CancelBacktestCommand."""
    
    def __init__(
        self,
        backtest_service: BacktestService,
        repository: BacktestWriteRepository
    ):
        """
        Initialize the handler.
        
        Args:
            backtest_service: Backtest service
            repository: Backtest write repository
        """
        self.backtest_service = backtest_service
        self.repository = repository
    
    async def handle(self, command: CancelBacktestCommand) -> None:
        """
        Handle the command.
        
        Args:
            command: The command
        """
        logger.info(f"Handling CancelBacktestCommand: {command}")
        
        # Cancel the backtest
        await self.backtest_service.cancel_backtest(command.backtest_id)
        
        # Update the backtest status
        backtest = await self.repository.get_by_id(command.backtest_id)
        if backtest:
            backtest.status = BacktestStatus.CANCELLED
            backtest.end_time = datetime.now()
            await self.repository.update(backtest)


class DeleteBacktestCommandHandler(CommandHandler):
    """Handler for DeleteBacktestCommand."""
    
    def __init__(
        self,
        repository: BacktestWriteRepository
    ):
        """
        Initialize the handler.
        
        Args:
            repository: Backtest write repository
        """
        self.repository = repository
    
    async def handle(self, command: DeleteBacktestCommand) -> None:
        """
        Handle the command.
        
        Args:
            command: The command
        """
        logger.info(f"Handling DeleteBacktestCommand: {command}")
        
        # Delete the backtest
        await self.repository.delete(command.backtest_id)