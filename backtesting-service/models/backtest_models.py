from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime

class BacktestStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StrategyType(str, Enum):
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    PATTERN_RECOGNITION = "pattern_recognition"
    MACHINE_LEARNING = "machine_learning"
    CUSTOM = "custom"

class StrategyConfig(BaseModel):
    strategy_id: Optional[str] = Field(default=None, description="ID of a predefined strategy")
    strategy_type: StrategyType = Field(..., description="Type of strategy to backtest")
    parameters: Dict[str, Any] = Field(..., description="Strategy-specific parameters")
    entry_conditions: List[Dict[str, Any]] = Field(..., description="Conditions for entering a position")
    exit_conditions: List[Dict[str, Any]] = Field(..., description="Conditions for exiting a position")
    risk_management: Dict[str, Any] = Field(..., description="Risk management parameters")

class BacktestRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of symbols to backtest")
    timeframe: str = Field(..., description="Timeframe for the backtest (e.g., '1h', '1d')")
    start_date: str = Field(..., description="Start date for the backtest (ISO format)")
    end_date: str = Field(..., description="End date for the backtest (ISO format)")
    initial_capital: float = Field(..., description="Initial capital for the backtest")
    strategy: StrategyConfig = Field(..., description="Strategy configuration")
    commission: float = Field(default=0.0, description="Commission rate")
    slippage: float = Field(default=0.0, description="Slippage in pips")
    leverage: float = Field(default=1.0, description="Leverage to use")
    additional_parameters: Optional[Dict[str, Any]] = Field(default=None, description="Additional backtest parameters")

class TradeResult(BaseModel):
    trade_id: str = Field(..., description="Unique ID for the trade")
    symbol: str = Field(..., description="Symbol traded")
    direction: str = Field(..., description="Long or short")
    entry_time: datetime = Field(..., description="Entry time")
    entry_price: float = Field(..., description="Entry price")
    exit_time: Optional[datetime] = Field(default=None, description="Exit time")
    exit_price: Optional[float] = Field(default=None, description="Exit price")
    size: float = Field(..., description="Position size")
    profit_loss: float = Field(..., description="Profit or loss")
    profit_loss_pct: float = Field(..., description="Profit or loss percentage")
    exit_reason: Optional[str] = Field(default=None, description="Reason for exiting the trade")

class PerformanceMetrics(BaseModel):
    total_return: float = Field(..., description="Total return percentage")
    annualized_return: float = Field(..., description="Annualized return percentage")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    win_rate: float = Field(..., description="Win rate percentage")
    profit_factor: float = Field(..., description="Profit factor")
    average_win: float = Field(..., description="Average winning trade")
    average_loss: float = Field(..., description="Average losing trade")
    total_trades: int = Field(..., description="Total number of trades")
    winning_trades: int = Field(..., description="Number of winning trades")
    losing_trades: int = Field(..., description="Number of losing trades")

class BacktestResult(BaseModel):
    backtest_id: str = Field(..., description="Unique ID for the backtest")
    request: BacktestRequest = Field(..., description="Original backtest request")
    status: BacktestStatus = Field(..., description="Status of the backtest")
    start_time: datetime = Field(..., description="Start time of the backtest execution")
    end_time: Optional[datetime] = Field(default=None, description="End time of the backtest execution")
    execution_time_ms: Optional[int] = Field(default=None, description="Execution time in milliseconds")
    trades: List[TradeResult] = Field(default=[], description="List of trades executed during the backtest")
    performance_metrics: Optional[PerformanceMetrics] = Field(default=None, description="Performance metrics of the backtest")
    equity_curve: Optional[List[Dict[str, Any]]] = Field(default=None, description="Equity curve data points")
    drawdown_curve: Optional[List[Dict[str, Any]]] = Field(default=None, description="Drawdown curve data points")
    error_message: Optional[str] = Field(default=None, description="Error message if the backtest failed")

class BacktestResponse(BaseModel):
    backtest_id: str = Field(..., description="Unique ID for the backtest")
    status: BacktestStatus = Field(..., description="Status of the backtest")
    message: str = Field(..., description="Status message")
    estimated_completion_time: Optional[datetime] = Field(default=None, description="Estimated completion time")