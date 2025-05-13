"""
Database Models Module.

Defines SQLAlchemy models for portfolio management data.
"""
from datetime import datetime
from typing import Any, Dict

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property

from core.connection import Base


class PositionDb(Base):
    """Database model for trading positions."""
    
    __tablename__ = "positions"
    __table_args__ = {"schema": "portfolio"}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    position_id = Column(String, unique=True, nullable=False, index=True)
    account_id = Column(String, nullable=False, index=True)
    symbol = Column(String, nullable=False, index=True)
    side = Column(String, nullable=False)  # "long" or "short"
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, nullable=False)
    realized_pnl = Column(Float, nullable=False, default=0.0)
    take_profit = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    open_time = Column(DateTime(timezone=True), nullable=False, index=True)
    close_time = Column(DateTime(timezone=True), nullable=True)
    status = Column(String, nullable=False, index=True)  # "open", "closed", "partially_closed"
    metadata = Column(JSON, nullable=True)
    tags = Column(JSON, nullable=True)
    
    # Audit columns
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary representation."""
        return {
            "position_id": self.position_id,
            "account_id": self.account_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "take_profit": self.take_profit,
            "stop_loss": self.stop_loss,
            "open_time": self.open_time,
            "close_time": self.close_time,
            "status": self.status,
            "metadata": self.metadata,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


class AccountBalanceDb(Base):
    """Database model for account balances."""
    
    __tablename__ = "account_balances"
    __table_args__ = {"schema": "portfolio"}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    account_id = Column(String, nullable=False, index=True)
    balance = Column(Float, nullable=False)
    equity = Column(Float, nullable=False)
    margin = Column(Float, nullable=False, default=0.0)
    free_margin = Column(Float, nullable=False, default=0.0)
    margin_level = Column(Float, nullable=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Audit columns
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary representation."""
        return {
            "account_id": self.account_id,
            "balance": self.balance,
            "equity": self.equity,
            "margin": self.margin,
            "free_margin": self.free_margin,
            "margin_level": self.margin_level,
            "timestamp": self.timestamp,
            "created_at": self.created_at
        }


class BalanceChangeDb(Base):
    """Database model for balance change records."""
    
    __tablename__ = "balance_changes"
    __table_args__ = {"schema": "portfolio"}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    change_id = Column(String, unique=True, nullable=False)
    account_id = Column(String, nullable=False, index=True)
    amount = Column(Float, nullable=False)
    balance_before = Column(Float, nullable=False)
    balance_after = Column(Float, nullable=False)
    change_type = Column(String, nullable=False, index=True)
    reference_id = Column(String, nullable=True, index=True)
    description = Column(Text, nullable=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Audit columns
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary representation."""
        return {
            "change_id": self.change_id,
            "account_id": self.account_id,
            "amount": self.amount,
            "balance_before": self.balance_before,
            "balance_after": self.balance_after,
            "change_type": self.change_type,
            "reference_id": self.reference_id,
            "description": self.description,
            "timestamp": self.timestamp,
            "created_at": self.created_at
        }


class DailyPerformanceDb(Base):
    """Database model for daily performance records."""
    
    __tablename__ = "daily_performance"
    __table_args__ = {"schema": "portfolio"}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    account_id = Column(String, nullable=False, index=True)
    date = Column(DateTime(timezone=True), nullable=False, index=True)
    starting_balance = Column(Float, nullable=False)
    ending_balance = Column(Float, nullable=False)
    deposits = Column(Float, nullable=False, default=0.0)
    withdrawals = Column(Float, nullable=False, default=0.0)
    realized_pnl = Column(Float, nullable=False, default=0.0)
    fees = Column(Float, nullable=False, default=0.0)
    swaps = Column(Float, nullable=False, default=0.0)
    trades_count = Column(Integer, nullable=False, default=0)
    win_count = Column(Integer, nullable=False, default=0)
    loss_count = Column(Integer, nullable=False, default=0)
    
    # Audit columns
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    @hybrid_property
    def net_profit_loss(self) -> float:
        """Calculate net profit/loss excluding deposits and withdrawals."""
        return self.realized_pnl - self.fees - self.swaps
    
    @hybrid_property
    def win_rate(self) -> float:
        """Calculate win rate as percentage."""
        if self.trades_count == 0:
            return 0.0
        return (self.win_count / self.trades_count) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary representation."""
        return {
            "account_id": self.account_id,
            "date": self.date,
            "starting_balance": self.starting_balance,
            "ending_balance": self.ending_balance,
            "deposits": self.deposits,
            "withdrawals": self.withdrawals,
            "realized_pnl": self.realized_pnl,
            "fees": self.fees,
            "swaps": self.swaps,
            "trades_count": self.trades_count,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "net_profit_loss": self.net_profit_loss,
            "win_rate": self.win_rate,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


class AccountInfoDb(Base):
    """Database model for account information."""
    
    __tablename__ = "accounts"
    __table_args__ = {"schema": "portfolio"}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    account_id = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=True)
    currency = Column(String, nullable=False, default="USD")
    leverage = Column(Float, nullable=False, default=100.0)
    status = Column(String, nullable=False, default="active")  # active, suspended, closed
    is_demo = Column(Boolean, nullable=False, default=False)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary representation."""
        return {
            "account_id": self.account_id,
            "name": self.name,
            "currency": self.currency,
            "leverage": self.leverage,
            "status": self.status,
            "is_demo": self.is_demo,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }