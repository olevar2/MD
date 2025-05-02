"""
Position Models Module

This module defines data models for trading positions with multi-asset support.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid

class PositionStatus(str, Enum):
    """Status of a trading position"""
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"


class PositionDirection(str, Enum):
    """Direction of a trading position"""
    LONG = "long"
    SHORT = "short"


class AssetClass(str, Enum):
    """Asset classes supported by the platform"""
    FOREX = "forex"
    CRYPTO = "crypto"
    STOCKS = "stocks"
    COMMODITIES = "commodities"
    INDICES = "indices"
    BONDS = "bonds"
    ETF = "etf"


class PositionPerformance(BaseModel):
    """Performance metrics for a position"""
    roi_pct: float = Field(..., description="Return on investment percentage")
    profit_loss: float = Field(..., description="Profit or loss in account currency")
    max_drawdown_pct: Optional[float] = Field(None, description="Maximum drawdown percentage")
    max_profit_pct: Optional[float] = Field(None, description="Maximum profit percentage")
    duration_hours: Optional[float] = Field(None, description="Duration in hours")
    avg_volatility: Optional[float] = Field(None, description="Average volatility during holding period")


class PositionBase(BaseModel):
    """Base model for trading positions"""
    symbol: str = Field(..., description="Trading symbol")
    direction: PositionDirection = Field(..., description="Position direction")
    quantity: float = Field(..., gt=0, description="Position quantity")
    entry_price: float = Field(..., gt=0, description="Entry price")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")
    account_id: str = Field(..., description="Account ID")
    strategy_id: Optional[str] = Field(None, description="Strategy ID that created this position")
    
    # Multi-asset fields
    asset_class: Optional[AssetClass] = Field(None, description="Asset class")
    margin_rate: Optional[float] = Field(None, description="Margin rate for this position")
    fee: Optional[float] = Field(0.0, description="Trading fee")
    pip_value: Optional[float] = Field(None, description="Pip value (for forex)")
    market_type: Optional[str] = Field(None, description="Market type")
    base_currency: Optional[str] = Field(None, description="Base currency for forex/crypto pairs")
    quote_currency: Optional[str] = Field(None, description="Quote currency for forex/crypto pairs")
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = Field({}, description="Additional metadata")


class PositionCreate(PositionBase):
    """Model for creating a new position"""
    pass


class PositionUpdate(BaseModel):
    """Model for updating a position"""
    current_price: Optional[float] = Field(None, description="Current market price")
    stop_loss: Optional[float] = Field(None, description="Updated stop loss price")
    take_profit: Optional[float] = Field(None, description="Updated take profit price")
    status: Optional[PositionStatus] = Field(None, description="Position status")
    exit_price: Optional[float] = Field(None, description="Exit price")
    exit_date: Optional[datetime] = Field(None, description="Exit date")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")


class Position(PositionBase):
    """Complete position model"""
    id: str = Field(..., description="Position ID")
    status: PositionStatus = Field(PositionStatus.OPEN, description="Position status")
    current_price: Optional[float] = Field(None, description="Current market price")
    current_value: Optional[float] = Field(None, description="Current position value")
    unrealized_pl: Optional[float] = Field(None, description="Unrealized profit/loss")
    realized_pl: Optional[float] = Field(0.0, description="Realized profit/loss")
    margin_used: Optional[float] = Field(None, description="Margin used by this position")
    entry_date: datetime = Field(default_factory=datetime.now, description="Entry date and time")
    exit_date: Optional[datetime] = Field(None, description="Exit date and time")
    exit_price: Optional[float] = Field(None, description="Exit price")
    performance: Optional[PositionPerformance] = Field(None, description="Position performance metrics")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update time")

    @validator('id', pre=True, always=True)
    def set_id(cls, id_value):
        """Set ID if not provided"""
        return id_value or str(uuid.uuid4())

    class Config:
        """Pydantic configuration"""
        orm_mode = True
