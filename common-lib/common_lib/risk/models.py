"""
Risk Management Models Module

This module provides standardized risk management models used across services,
helping to break circular dependencies between services.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple
from pydantic import BaseModel, Field

from common_lib.risk.interfaces import RiskRegimeType, RiskParameterType, IRiskParameters


class RiskLimit(BaseModel):
    """Model for a risk limit."""
    limit_type: str
    limit_value: float
    symbol: Optional[str] = None
    account_id: Optional[str] = None
    description: Optional[str] = None
    active: bool = True
    
    class Config:
        orm_mode = True


class RiskProfile(BaseModel):
    """Model for a risk profile with predefined limits."""
    id: str
    name: str
    description: Optional[str] = None
    risk_level: str
    limits: List[RiskLimit]
    created_at: datetime
    updated_at: Optional[datetime] = None
    
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


class RiskParameters(BaseModel, IRiskParameters):
    """Standard implementation of risk parameters."""
    position_size_method: str = "fixed_percent"
    position_size_value: float = 1.0  # 1% of account by default
    max_position_size: float = 5.0    # 5% of account maximum
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 3.0
    max_risk_per_trade_pct: float = 1.0
    max_correlation_allowed: float = 0.7
    max_portfolio_heat: float = 20.0  # Maximum % of account at risk
    volatility_scaling_enabled: bool = True
    news_sensitivity: float = 0.5     # 0-1 scale
    regime_adaptation_level: float = 0.5  # 0-1 scale
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskParameters':
        """Create parameters from dictionary."""
        return cls(**data)
    
    def adjust_for_regime(self, regime_type: RiskRegimeType) -> 'RiskParameters':
        """Create a copy of risk parameters adjusted for the given risk regime."""
        params = self.copy()
        
        # Apply adjustments based on regime
        if regime_type == RiskRegimeType.LOW_RISK:
            params.position_size_value *= 1.2
            params.max_position_size *= 1.1
            params.stop_loss_atr_multiplier *= 0.9
            params.max_portfolio_heat *= 1.1
        elif regime_type == RiskRegimeType.MODERATE_RISK:
            # No changes for moderate risk (baseline)
            pass
        elif regime_type == RiskRegimeType.HIGH_RISK:
            params.position_size_value *= 0.8
            params.max_position_size *= 0.9
            params.stop_loss_atr_multiplier *= 1.1
            params.max_portfolio_heat *= 0.9
        elif regime_type == RiskRegimeType.EXTREME_RISK:
            params.position_size_value *= 0.6
            params.max_position_size *= 0.7
            params.stop_loss_atr_multiplier *= 1.3
            params.max_portfolio_heat *= 0.7
        elif regime_type == RiskRegimeType.CRISIS:
            params.position_size_value *= 0.4
            params.max_position_size *= 0.5
            params.stop_loss_atr_multiplier *= 1.5
            params.max_portfolio_heat *= 0.5
        
        return params


class RiskLimitBreachAction(str, Enum):
    """Actions to take when a risk limit is breached."""
    NOTIFY_ONLY = "notify_only"
    PREVENT_NEW_POSITIONS = "prevent_new"
    CLOSE_POSITIONS = "close_positions"
    REDUCE_POSITION_SIZE = "reduce_size"
    SUSPEND_ACCOUNT = "suspend_account"


class RiskLimitBreachSeverity(str, Enum):
    """Severity levels for risk limit breaches."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskLimitBreachStatus(str, Enum):
    """Status of a risk limit breach."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    OVERRIDDEN = "overridden"


class RiskLimitBreachNotification(BaseModel):
    """Notification for a risk limit breach."""
    breach_id: str
    account_id: str
    limit_type: str
    limit_value: float
    actual_value: float
    severity: RiskLimitBreachSeverity
    status: RiskLimitBreachStatus
    action: RiskLimitBreachAction
    created_at: datetime
    symbol: Optional[str] = None
    message: Optional[str] = None
    
    class Config:
        orm_mode = True
"""
