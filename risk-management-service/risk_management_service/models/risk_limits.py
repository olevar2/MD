"""
Risk Limits Models Module.

Defines models for risk limits and related objects.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class LimitType(str, Enum):
    """Enumeration of risk limit types."""
    MAX_POSITION_SIZE = "MAX_POSITION_SIZE"
    MAX_POSITIONS_COUNT = "MAX_POSITIONS_COUNT"
    MAX_SINGLE_EXPOSURE_PCT = "MAX_SINGLE_EXPOSURE_PCT"
    MAX_TOTAL_EXPOSURE_PCT = "MAX_TOTAL_EXPOSURE_PCT"
    MAX_DRAWDOWN_PCT = "MAX_DRAWDOWN_PCT"
    MAX_DAILY_LOSS_PCT = "MAX_DAILY_LOSS_PCT"
    MAX_VAR_PCT = "MAX_VAR_PCT"
    MIN_FREE_MARGIN_PCT = "MIN_FREE_MARGIN_PCT"


class RiskLimitBase(BaseModel):
    """Base model for risk limits."""
    limit_type: LimitType
    limit_value: float
    description: Optional[str] = None


class RiskLimitCreate(RiskLimitBase):
    """Model for creating a risk limit."""
    account_id: str


class RiskLimitUpdate(BaseModel):
    """Model for updating a risk limit."""
    limit_value: Optional[float] = None
    description: Optional[str] = None
    active: Optional[bool] = None


class RiskLimit(RiskLimitBase):
    """Model for a risk limit."""
    id: str
    account_id: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    active: bool = True
    
    class Config:
        orm_mode = True


class ProfileLimit(BaseModel):
    """Model for a limit within a risk profile."""
    limit_type: LimitType
    limit_value: float
    description: Optional[str] = None


class RiskProfile(BaseModel):
    """Model for a risk profile with predefined limits."""
    id: str
    name: str
    description: Optional[str] = None
    risk_level: str
    limits: List[ProfileLimit]
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True


class RiskProfileCreate(BaseModel):
    """Model for creating a risk profile."""
    name: str
    description: Optional[str] = None
    risk_level: str
    limits: List[ProfileLimit]


class RiskProfileUpdate(BaseModel):
    """Model for updating a risk profile."""
    name: Optional[str] = None
    description: Optional[str] = None
    risk_level: Optional[str] = None
    limits: Optional[List[ProfileLimit]] = None


class RiskCheck(BaseModel):
    """Model for a risk check result."""
    id: str
    account_id: str
    check_type: str
    result: str
    violations_count: int
    created_at: datetime
    details: Optional[Dict[str, Any]] = None
    
    class Config:
        orm_mode = True


class RiskViolation(BaseModel):
    """Model for a risk limit violation."""
    id: str
    account_id: str
    limit_id: str
    check_id: Optional[str] = None
    limit_type: LimitType
    limit_value: float
    actual_value: float
    severity: str
    created_at: datetime
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    
    class Config:
        orm_mode = True
