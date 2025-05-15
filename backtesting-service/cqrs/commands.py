"""
Command models for the Backtesting Service.

This module provides the command models for the Backtesting Service.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from common_lib.cqrs.commands import Command


class RunBacktestCommand(Command):
    """Command to run a backtest."""
    
    strategy_id: str = Field(..., description="ID of the strategy to backtest")
    symbol: str = Field(..., description="Symbol to backtest")
    timeframe: str = Field(..., description="Timeframe for the backtest")
    start_date: datetime = Field(..., description="Start date for the backtest")
    end_date: datetime = Field(..., description="End date for the backtest")
    initial_balance: float = Field(10000.0, description="Initial balance for the backtest")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Parameters for the backtest")


class OptimizeStrategyCommand(Command):
    """Command to optimize a strategy."""
    
    strategy_id: str = Field(..., description="ID of the strategy to optimize")
    symbol: str = Field(..., description="Symbol to optimize for")
    timeframe: str = Field(..., description="Timeframe for the optimization")
    start_date: datetime = Field(..., description="Start date for the optimization")
    end_date: datetime = Field(..., description="End date for the optimization")
    initial_balance: float = Field(10000.0, description="Initial balance for the optimization")
    parameters_to_optimize: Dict[str, Dict[str, Any]] = Field(..., description="Parameters to optimize with ranges")
    optimization_metric: str = Field("sharpe_ratio", description="Metric to optimize for")
    num_iterations: int = Field(100, description="Number of iterations for the optimization")


class RunWalkForwardTestCommand(Command):
    """Command to run a walk-forward test."""
    
    strategy_id: str = Field(..., description="ID of the strategy to test")
    symbol: str = Field(..., description="Symbol to test")
    timeframe: str = Field(..., description="Timeframe for the test")
    start_date: datetime = Field(..., description="Start date for the test")
    end_date: datetime = Field(..., description="End date for the test")
    initial_balance: float = Field(10000.0, description="Initial balance for the test")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Base parameters for the strategy")
    optimization_window: int = Field(..., description="Number of bars in the optimization window")
    test_window: int = Field(..., description="Number of bars in the test window")
    optimization_metric: str = Field("sharpe_ratio", description="Metric to optimize for")
    parameters_to_optimize: Dict[str, Dict[str, Any]] = Field(..., description="Parameters to optimize with ranges")


class CancelBacktestCommand(Command):
    """Command to cancel a running backtest."""
    
    backtest_id: str = Field(..., description="ID of the backtest to cancel")


class DeleteBacktestCommand(Command):
    """Command to delete a backtest."""
    
    backtest_id: str = Field(..., description="ID of the backtest to delete")