"""
Database Models Module.

Defines SQLAlchemy models for the risk management database.
"""
import json
from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy import (
    Column, String, Float, Boolean, Text, JSON, DateTime,
    ForeignKey, Index, UniqueConstraint, Integer
)
from sqlalchemy.orm import relationship

from risk_management_service.db.connection import Base


class RiskLimitDb(Base):
    """SQLAlchemy model for risk limits."""
    
    __tablename__ = "risk_limits"
    
    limit_id = Column(String(36), primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    limit_type = Column(String(50), nullable=False, index=True)
    scope = Column(String(20), nullable=False, index=True)
    account_id = Column(String(36), nullable=True, index=True)
    strategy_id = Column(String(36), nullable=True, index=True)
    value = Column(Float, nullable=False)
    unit = Column(String(20), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    metadata = Column(JSON, default=lambda: {}, nullable=True)
    
    # Relationships
    breaches = relationship("RiskLimitBreachDb", back_populates="risk_limit", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("ix_risk_limits_account_strategy", "account_id", "strategy_id"),
        UniqueConstraint("name", "scope", "account_id", "strategy_id", 
                         name="uq_risk_limits_name_scope_account_strategy"),
    )


class RiskLimitBreachDb(Base):
    """SQLAlchemy model for risk limit breaches."""
    
    __tablename__ = "risk_limit_breaches"
    
    breach_id = Column(String(36), primary_key=True)
    limit_id = Column(String(36), ForeignKey("risk_limits.limit_id"), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)
    current_value = Column(Float, nullable=False)
    limit_value = Column(Float, nullable=False)
    action_taken = Column(String(50), nullable=False)
    description = Column(Text, nullable=False)
    breach_time = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    status = Column(String(20), nullable=False, default="active", index=True)
    account_id = Column(String(36), nullable=True, index=True)
    strategy_id = Column(String(36), nullable=True, index=True)
    resolved_time = Column(DateTime, nullable=True)
    override_reason = Column(Text, nullable=True)
    override_by = Column(String(100), nullable=True)
    metadata = Column(JSON, default=lambda: {}, nullable=True)
    
    # Relationships
    risk_limit = relationship("RiskLimitDb", back_populates="breaches")
    
    # Indexes
    __table_args__ = (
        Index("ix_risk_breaches_account_strategy", "account_id", "strategy_id"),
    )


class RiskMetricDb(Base):
    """SQLAlchemy model for risk metrics."""
    
    __tablename__ = "risk_metrics"
    
    metric_id = Column(String(36), primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    account_id = Column(String(36), nullable=True, index=True)
    strategy_id = Column(String(36), nullable=True, index=True)
    symbol = Column(String(20), nullable=True, index=True)
    value = Column(Float, nullable=False)
    unit = Column(String(20), nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    metadata = Column(JSON, default=lambda: {}, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index("ix_risk_metrics_account_strategy", "account_id", "strategy_id"),
        Index("ix_risk_metrics_name_account", "name", "account_id"),
    )


class RiskSettingDb(Base):
    """SQLAlchemy model for risk settings."""
    
    __tablename__ = "risk_settings"
    
    setting_id = Column(String(36), primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    value = Column(Text, nullable=False)
    value_type = Column(String(20), nullable=False)  # string, number, boolean, json
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def get_typed_value(self) -> Any:
        """
        Get the value with its proper type.
        
        Returns:
            Typed value based on value_type
        """
        if self.value_type == "number":
            return float(self.value)
        elif self.value_type == "boolean":
            return self.value.lower() in ("true", "yes", "1")
        elif self.value_type == "json":
            return json.loads(self.value)
        else:  # string or default
            return self.value