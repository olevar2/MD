"""
Backtesting Models

This module defines the data models for backtesting requests and responses.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class PerformanceMetrics(BaseModel):
    """
    Performance metrics for a backtest.
    """
    total_return: float = Field(..., description="Total return of the backtest")
    annualized_return: float = Field(..., description="Annualized return of the backtest")
    sharpe_ratio: float = Field(..., description="Sharpe ratio of the backtest")
    max_drawdown: float = Field(..., description="Maximum drawdown of the backtest")
    win_rate: float = Field(..., description="Win rate of the backtest")
    profit_factor: float = Field(..., description="Profit factor of the backtest")
    average_trade: float = Field(..., description="Average trade profit/loss")
    average_winning_trade: float = Field(..., description="Average winning trade profit")
    average_losing_trade: float = Field(..., description="Average losing trade loss")


class TradeResult(BaseModel):
    """
    Result of a trade.
    """
    direction: str = Field(..., description="Direction of the trade (long or short)")
    size: float = Field(..., description="Size of the trade")
    entry_price: float = Field(..., description="Entry price of the trade")
    entry_time: datetime = Field(..., description="Entry time of the trade")
    exit_price: float = Field(..., description="Exit price of the trade")
    exit_time: datetime = Field(..., description="Exit time of the trade")
    pnl: float = Field(..., description="Profit/loss of the trade")
    commission: float = Field(..., description="Commission paid for the trade")
    duration: float = Field(..., description="Duration of the trade in hours")


class BacktestRequest(BaseModel):
    """
    Request model for running a backtest.
    """
    strategy_id: str = Field(..., description="ID of the strategy to backtest")
    symbol: str = Field(..., description="Symbol to backtest")
    timeframe: str = Field(..., description="Timeframe for the backtest")
    start_date: datetime = Field(..., description="Start date for the backtest")
    end_date: datetime = Field(..., description="End date for the backtest")
    initial_balance: Optional[float] = Field(10000.0, description="Initial balance for the backtest")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Parameters for the backtest")


class BacktestResponse(BaseModel):
    """
    Response model for a backtest request.
    """
    backtest_id: str = Field(..., description="ID of the backtest")
    status: str = Field(..., description="Status of the backtest")
    message: str = Field(..., description="Message about the backtest")
    estimated_completion_time: Optional[datetime] = Field(None, description="Estimated completion time")


class BacktestResult(BaseModel):
    """
    Result model for a completed backtest.
    """
    backtest_id: str = Field(..., description="ID of the backtest")
    strategy_id: str = Field(..., description="ID of the strategy")
    start_date: datetime = Field(..., description="Start date of the backtest")
    end_date: datetime = Field(..., description="End date of the backtest")
    initial_balance: float = Field(..., description="Initial balance for the backtest")
    final_balance: float = Field(..., description="Final balance after the backtest")
    total_trades: int = Field(..., description="Total number of trades")
    winning_trades: int = Field(..., description="Number of winning trades")
    losing_trades: int = Field(..., description="Number of losing trades")
    performance_metrics: PerformanceMetrics = Field(..., description="Performance metrics for the backtest")
    trades: List[Dict[str, Any]] = Field(..., description="List of trades")
    equity_curve: List[Dict[str, Any]] = Field(..., description="Equity curve for the backtest")
    parameters: Dict[str, Any] = Field(..., description="Parameters used for the backtest")


class BacktestStatus(str):
    """
    Status of a backtest.
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StrategyMetadata(BaseModel):
    """
    Metadata for a trading strategy.
    """
    strategy_id: str = Field(..., description="ID of the strategy")
    name: str = Field(..., description="Name of the strategy")
    description: str = Field(..., description="Description of the strategy")
    version: str = Field(..., description="Version of the strategy")
    author: str = Field(..., description="Author of the strategy")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    parameters: Dict[str, Any] = Field(..., description="Parameters for the strategy")
    supported_symbols: List[str] = Field(..., description="Symbols supported by the strategy")
    supported_timeframes: List[str] = Field(..., description="Timeframes supported by the strategy")


class StrategyListResponse(BaseModel):
    """
    Response model for listing available strategies.
    """
    strategies: List[StrategyMetadata] = Field(..., description="List of available strategies")
    count: int = Field(..., description="Number of strategies")


class BacktestListResponse(BaseModel):
    """
    Response model for listing backtests.
    """
    backtests: List[Dict[str, Any]] = Field(..., description="List of backtests")
    count: int = Field(..., description="Number of backtests")


class OptimizationRequest(BaseModel):
    """
    Request model for optimizing a strategy.
    """
    strategy_id: str = Field(..., description="ID of the strategy to optimize")
    symbol: str = Field(..., description="Symbol to optimize for")
    timeframe: str = Field(..., description="Timeframe for the optimization")
    start_date: datetime = Field(..., description="Start date for the optimization")
    end_date: datetime = Field(..., description="End date for the optimization")
    initial_balance: Optional[float] = Field(10000.0, description="Initial balance for the optimization")
    parameters_to_optimize: Dict[str, Dict[str, Any]] = Field(..., description="Parameters to optimize with ranges")
    optimization_metric: str = Field("sharpe_ratio", description="Metric to optimize for")
    optimization_method: str = Field("grid_search", description="Optimization method to use")
    max_evaluations: Optional[int] = Field(100, description="Maximum number of evaluations")


class OptimizationResponse(BaseModel):
    """
    Response model for an optimization request.
    """
    optimization_id: str = Field(..., description="ID of the optimization")
    status: str = Field(..., description="Status of the optimization")
    message: str = Field(..., description="Message about the optimization")
    estimated_completion_time: Optional[datetime] = Field(None, description="Estimated completion time")


class OptimizationResult(BaseModel):
    """
    Result model for a completed optimization.
    """
    optimization_id: str = Field(..., description="ID of the optimization")
    strategy_id: str = Field(..., description="ID of the strategy")
    symbol: str = Field(..., description="Symbol optimized for")
    timeframe: str = Field(..., description="Timeframe for the optimization")
    start_date: datetime = Field(..., description="Start date of the optimization")
    end_date: datetime = Field(..., description="End date of the optimization")
    optimization_metric: str = Field(..., description="Metric optimized for")
    optimization_method: str = Field(..., description="Optimization method used")
    best_parameters: Dict[str, Any] = Field(..., description="Best parameters found")
    best_metric_value: float = Field(..., description="Best metric value found")
    evaluations: int = Field(..., description="Number of evaluations performed")
    all_results: List[Dict[str, Any]] = Field(..., description="All optimization results")


class WalkForwardTestRequest(BaseModel):
    """
    Request model for running a walk-forward test.
    """
    strategy_id: str = Field(..., description="ID of the strategy to test")
    symbol: str = Field(..., description="Symbol to test")
    timeframe: str = Field(..., description="Timeframe for the test")
    start_date: datetime = Field(..., description="Start date for the test")
    end_date: datetime = Field(..., description="End date for the test")
    initial_balance: Optional[float] = Field(10000.0, description="Initial balance for the test")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Base parameters for the strategy")
    optimization_window: int = Field(..., description="Number of bars in the optimization window")
    test_window: int = Field(..., description="Number of bars in the test window")
    optimization_metric: str = Field("sharpe_ratio", description="Metric to optimize for")
    parameters_to_optimize: Dict[str, Dict[str, Any]] = Field(..., description="Parameters to optimize with ranges")


class WalkForwardTestResponse(BaseModel):
    """
    Response model for a walk-forward test request.
    """
    test_id: str = Field(..., description="ID of the walk-forward test")
    status: str = Field(..., description="Status of the test")
    message: str = Field(..., description="Message about the test")
    estimated_completion_time: Optional[datetime] = Field(None, description="Estimated completion time")


class WalkForwardTestResult(BaseModel):
    """
    Result model for a completed walk-forward test.
    """
    test_id: str = Field(..., description="ID of the walk-forward test")
    strategy_id: str = Field(..., description="ID of the strategy")
    symbol: str = Field(..., description="Symbol tested")
    timeframe: str = Field(..., description="Timeframe for the test")
    start_date: datetime = Field(..., description="Start date of the test")
    end_date: datetime = Field(..., description="End date of the test")
    initial_balance: float = Field(..., description="Initial balance for the test")
    final_balance: float = Field(..., description="Final balance after the test")
    total_trades: int = Field(..., description="Total number of trades")
    winning_trades: int = Field(..., description="Number of winning trades")
    losing_trades: int = Field(..., description="Number of losing trades")
    performance_metrics: PerformanceMetrics = Field(..., description="Performance metrics for the test")
    trades: List[Dict[str, Any]] = Field(..., description="List of trades")
    equity_curve: List[Dict[str, Any]] = Field(..., description="Equity curve for the test")
    windows: List[Dict[str, Any]] = Field(..., description="List of optimization and test windows")
    parameters_by_window: Dict[str, Dict[str, Any]] = Field(..., description="Parameters used for each window")