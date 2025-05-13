"""
Historical Portfolio Models.

Defines models for historical portfolio tracking.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class PortfolioSnapshot(BaseModel):
    """Model for a point-in-time snapshot of portfolio state."""
    
    id: Optional[str] = Field(None, description="Unique identifier")
    account_id: str = Field(..., description="Account ID this snapshot belongs to")
    timestamp: datetime = Field(..., description="Time when the snapshot was taken")
    balance: float = Field(..., description="Account balance")
    equity: float = Field(..., description="Account equity (balance + unrealized P&L)")
    open_positions_count: int = Field(..., description="Number of open positions")
    margin_used: float = Field(..., description="Amount of margin used")
    free_margin: float = Field(..., description="Available margin")
    unrealized_pnl: float = Field(..., description="Total unrealized P&L")
    
    class Config:
        orm_mode = True


class PerformanceRecord(BaseModel):
    """Model for a record of portfolio performance metrics."""
    
    id: Optional[str] = Field(None, description="Unique identifier")
    account_id: str = Field(..., description="Account ID this performance record belongs to")
    timestamp: datetime = Field(..., description="Time when the record was created")
    period: str = Field(..., description="Period this record covers (e.g., 'daily', '30D')")
    win_rate: float = Field(..., description="Win rate")
    profit_factor: float = Field(..., description="Profit factor")
    net_profit: float = Field(..., description="Net profit for the period")
    total_trades: int = Field(..., description="Total trades in the period")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    
    class Config:
        orm_mode = True


class DrawdownInfo(BaseModel):
    """Model for drawdown information."""
    
    start: datetime = Field(..., description="Start time of drawdown")
    end: datetime = Field(..., description="End time of drawdown")
    amount: float = Field(..., description="Drawdown amount")
    percentage: float = Field(..., description="Drawdown as percentage")
    duration_days: int = Field(..., description="Duration in days")


class DrawdownAnalysis(BaseModel):
    """Model for comprehensive drawdown analysis."""
    
    max_drawdown: float = Field(..., description="Maximum drawdown amount")
    max_drawdown_percentage: float = Field(..., description="Maximum drawdown as percentage")
    max_drawdown_start: Optional[datetime] = Field(None, description="Start of maximum drawdown period")
    max_drawdown_end: Optional[datetime] = Field(None, description="End of maximum drawdown period")
    average_drawdown: float = Field(..., description="Average drawdown amount")
    average_drawdown_length_days: float = Field(..., description="Average drawdown duration in days")
    drawdowns: List[DrawdownInfo] = Field(default_factory=list, description="List of all drawdowns")
