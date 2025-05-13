"""
Account Models Module.

Defines models for account balances and related objects.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class AccountBalance(BaseModel):
    """Model for an account balance."""
    
    id: Optional[str] = Field(None, description="Unique identifier")
    user_id: str = Field(..., description="ID of user who owns this account")
    balance: float = Field(..., description="Current account balance")
    margin_used: float = Field(0, description="Amount of margin currently used")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")
    
    class Config:
        orm_mode = True


class BalanceChange(BaseModel):
    """Model for a record of balance change."""
    
    id: Optional[str] = Field(None, description="Unique identifier")
    account_id: str = Field(..., description="Account ID this change belongs to")
    amount: float = Field(..., description="Amount changed (positive or negative)")
    balance_before: float = Field(..., description="Balance before change")
    balance_after: float = Field(..., description="Balance after change")
    reason: str = Field(..., description="Reason for balance change")
    timestamp: datetime = Field(..., description="When the change occurred")
    
    class Config:
        orm_mode = True


class AccountDetails(AccountBalance):
    """Extended account balance with additional details."""
    
    free_margin: float = Field(..., description="Available margin (balance - margin_used)")
    equity: float = Field(..., description="Account equity (balance + unrealized P&L)")
    unrealized_pnl: float = Field(0, description="Current unrealized profit/loss")
    total_positions: int = Field(0, description="Total number of positions")
    open_positions: int = Field(0, description="Number of open positions")
    winning_positions: int = Field(0, description="Number of winning closed positions")
    losing_positions: int = Field(0, description="Number of losing closed positions")
    win_rate: float = Field(0, description="Win rate as decimal (0-1)")
    total_pnl: float = Field(0, description="Total realized profit/loss")
    recent_changes: List[BalanceChange] = Field(default_factory=list, description="Recent balance changes")


class AccountCreate(BaseModel):
    """Model for creating a new account."""
    
    user_id: str = Field(..., description="ID of user who will own this account")
    initial_balance: float = Field(..., description="Initial account balance")


class AccountUpdate(BaseModel):
    """Model for updating an account."""
    
    balance: Optional[float] = Field(None, description="New account balance")
    margin_used: Optional[float] = Field(None, description="New margin used")
