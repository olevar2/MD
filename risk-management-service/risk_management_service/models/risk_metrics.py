"""
Risk Metrics Models Module.

Defines models for risk metrics and related objects.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AccountRiskInfo(BaseModel):
    """Model for account risk information."""
    account_id: str
    balance: float
    equity: float
    margin_used: float
    free_margin: float
    margin_level: Optional[float] = None
    currency: str
    leverage: float
    updated_at: datetime
    
    class Config:
        orm_mode = True


class PositionRisk(BaseModel):
    """Model for position risk information."""
    position_id: str
    account_id: str
    symbol: str
    direction: str
    size: float
    entry_price: float
    current_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float
    risk_amount: float
    risk_pct: float
    reward_amount: Optional[float] = None
    reward_pct: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    margin_used: float
    
    class Config:
        orm_mode = True


class RiskMetrics(BaseModel):
    """Model for overall risk metrics."""
    account_id: str
    timestamp: datetime
    
    # Position metrics
    open_positions_count: int
    positions_by_direction: Dict[str, int]
    positions_by_symbol: Dict[str, int]
    
    # Exposure metrics
    total_exposure: float
    total_exposure_pct: float
    largest_exposure: float
    largest_exposure_pct: float
    largest_exposure_symbol: str
    
    # P&L metrics
    unrealized_pnl: float
    unrealized_pnl_pct: float
    daily_pnl: float
    daily_pnl_pct: float
    
    # Risk metrics
    max_drawdown_pct: float
    current_drawdown_pct: float
    var_pct: float
    var_amount: float
    cvar_pct: float
    cvar_amount: float
    
    # Margin metrics
    margin_used: float
    margin_used_pct: float
    free_margin: float
    margin_level: float
    
    class Config:
        orm_mode = True


class HistoricalRiskMetrics(BaseModel):
    """Model for historical risk metrics."""
    account_id: str
    date: datetime
    metrics: RiskMetrics
    
    class Config:
        orm_mode = True


class RiskScenario(BaseModel):
    """Model for a risk scenario for stress testing."""
    id: str
    name: str
    description: Optional[str] = None
    market_changes: Dict[str, float]  # symbol: pct_change
    volatility_changes: Dict[str, float]  # symbol: pct_change
    created_at: datetime
    
    class Config:
        orm_mode = True


class StressTestResult(BaseModel):
    """Model for a stress test result."""
    id: str
    account_id: str
    scenario_id: str
    scenario_name: str
    original_equity: float
    projected_equity: float
    equity_change_pct: float
    margin_level_change_pct: float
    position_impacts: List[Dict[str, Any]]
    run_at: datetime
    
    class Config:
        orm_mode = True


class RiskAlert(BaseModel):
    """Model for a risk alert."""
    id: str
    account_id: str
    alert_type: str
    severity: str
    message: str
    related_check_id: Optional[str] = None
    created_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True
