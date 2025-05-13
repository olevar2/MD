"""
Risk Management Interfaces Module

This module provides interfaces for risk management components used across services,
helping to break circular dependencies between services.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime

from common_lib.interfaces.trading import OrderType, OrderSide


class RiskLimitType(str, Enum):
    """Types of risk limits that can be enforced."""
    MAX_POSITION_SIZE = "max_position_size"
    MAX_LEVERAGE = "max_leverage"
    MAX_DRAWDOWN = "max_drawdown"
    MAX_EXPOSURE = "max_exposure"
    MAX_RISK_PER_TRADE = "max_risk_per_trade"
    MAX_CORRELATION = "max_correlation"
    MAX_CONCENTRATION = "max_concentration"
    MAX_VOLATILITY = "max_volatility"
    CIRCUIT_BREAKER = "circuit_breaker"


class RiskCheckResult:
    """Result of a risk check operation."""
    
    def __init__(
        self,
        is_valid: bool,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        breached_limits: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize a risk check result.
        
        Args:
            is_valid: Whether the check passed
            message: Message explaining the result
            details: Optional details about the check
            breached_limits: Optional list of breached limits
        """
        self.is_valid = is_valid
        self.message = message
        self.details = details or {}
        self.breached_limits = breached_limits or []
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "is_valid": self.is_valid,
            "message": self.message,
            "details": self.details,
            "breached_limits": self.breached_limits
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskCheckResult':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            RiskCheckResult instance
        """
        return cls(
            is_valid=data.get("is_valid", False),
            message=data.get("message", ""),
            details=data.get("details"),
            breached_limits=data.get("breached_limits")
        )


class IRiskManager(ABC):
    """Interface for risk management operations."""
    
    @abstractmethod
    async def check_order(
        self,
        symbol: str,
        order_type: Union[str, OrderType],
        side: Union[str, OrderSide],
        quantity: float,
        price: Optional[float] = None,
        account_id: Optional[str] = None
    ) -> RiskCheckResult:
        """
        Check if an order complies with risk limits.
        
        Args:
            symbol: Trading symbol
            order_type: Order type
            side: Order side
            quantity: Order quantity
            price: Optional order price
            account_id: Optional account ID
            
        Returns:
            Risk check result
        """
        pass
    
    @abstractmethod
    async def get_position_risk(
        self,
        symbol: Optional[str] = None,
        account_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get risk metrics for current positions.
        
        Args:
            symbol: Optional symbol to filter by
            account_id: Optional account ID
            
        Returns:
            Risk metrics for positions
        """
        pass
    
    @abstractmethod
    async def get_portfolio_risk(
        self,
        account_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get risk metrics for the entire portfolio.
        
        Args:
            account_id: Optional account ID
            
        Returns:
            Risk metrics for portfolio
        """
        pass
    
    @abstractmethod
    async def set_risk_limit(
        self,
        limit_type: Union[str, RiskLimitType],
        value: float,
        symbol: Optional[str] = None,
        account_id: Optional[str] = None
    ) -> bool:
        """
        Set a risk limit.
        
        Args:
            limit_type: Type of risk limit
            value: Limit value
            symbol: Optional symbol to apply limit to
            account_id: Optional account ID
            
        Returns:
            True if limit was set successfully
        """
        pass
    
    @abstractmethod
    async def get_risk_limits(
        self,
        limit_type: Optional[Union[str, RiskLimitType]] = None,
        symbol: Optional[str] = None,
        account_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get current risk limits.
        
        Args:
            limit_type: Optional type of risk limit to filter by
            symbol: Optional symbol to filter by
            account_id: Optional account ID
            
        Returns:
            Current risk limits
        """
        pass
    
    @abstractmethod
    async def check_risk_limits(
        self,
        account_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Check if any risk limits are currently breached.
        
        Args:
            account_id: Optional account ID
            
        Returns:
            List of breached risk limits
        """
        pass
    
    @abstractmethod
    async def add_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: Union[str, OrderSide],
        account_id: Optional[str] = None,
        leverage: float = 1.0
    ) -> bool:
        """
        Add a position for risk tracking.
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity
            price: Position price
            side: Position side
            account_id: Optional account ID
            leverage: Optional position leverage
            
        Returns:
            True if position was added successfully
        """
        pass
    
    @abstractmethod
    async def update_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        account_id: Optional[str] = None
    ) -> bool:
        """
        Update an existing position.
        
        Args:
            symbol: Trading symbol
            quantity: New position quantity
            price: New position price
            account_id: Optional account ID
            
        Returns:
            True if position was updated successfully
        """
        pass
    
    @abstractmethod
    async def remove_position(
        self,
        symbol: str,
        account_id: Optional[str] = None
    ) -> bool:
        """
        Remove a position from risk tracking.
        
        Args:
            symbol: Trading symbol
            account_id: Optional account ID
            
        Returns:
            True if position was removed successfully
        """
        pass
    
    @abstractmethod
    async def get_risk_metrics(
        self,
        account_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive risk metrics.
        
        Args:
            account_id: Optional account ID
            
        Returns:
            Comprehensive risk metrics
        """
        pass
