"""
Risk Management Service Adapter for Trading Gateway Service.

This module provides adapter implementations for the Risk Management Service interfaces,
allowing the Trading Gateway Service to interact with the Risk Management Service
without direct dependencies.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from common_lib.interfaces.risk_management import (
    IRiskManager, RiskLimitType, RiskCheckResult
)
from common_lib.interfaces.trading import OrderType, OrderSide
from common_lib.adapters.risk_management_adapter import RiskManagementAdapter
from common_lib.service_client.base_client import ServiceClientConfig
from common_lib.adapters.factory import AdapterFactory

logger = logging.getLogger(__name__)


class TradingRiskManagementAdapter:
    """
    Adapter for Risk Management Service operations in the Trading Gateway Service.
    
    This adapter provides methods for risk management operations specific to
    the Trading Gateway Service.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adapter.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Create service client configuration
        service_config = ServiceClientConfig(
            base_url=self.config.get("risk_management_api_url", "http://risk-management-service:8000/api/v1"),
            timeout=self.config.get("timeout", 30),
            retry_attempts=self.config.get("retry_attempts", 3),
            retry_backoff=self.config.get("retry_backoff", 1.5)
        )
        
        # Create adapter factory
        adapter_factory = AdapterFactory(
            config_provider={"risk-management-service": service_config}
        )
        
        # Create risk management adapter
        self.risk_manager = adapter_factory.create_risk_management()
    
    async def validate_order(self, order_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate an order against risk limits.
        
        Args:
            order_request: Order request dictionary
            
        Returns:
            Validation result
        """
        # Extract order details
        symbol = order_request.get("symbol")
        order_type = order_request.get("order_type")
        side = order_request.get("side")
        quantity = order_request.get("quantity")
        price = order_request.get("price")
        account_id = order_request.get("account_id")
        
        # Check order against risk limits
        result = await self.risk_manager.check_order(
            symbol=symbol,
            order_type=order_type,
            side=side,
            quantity=quantity,
            price=price,
            account_id=account_id
        )
        
        # Convert RiskCheckResult to dictionary
        return result.to_dict()
    
    async def get_position_risk(self, symbol: str, account_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get risk metrics for a position.
        
        Args:
            symbol: Trading symbol
            account_id: Optional account ID
            
        Returns:
            Risk metrics for the position
        """
        return await self.risk_manager.get_position_risk(symbol=symbol, account_id=account_id)
    
    async def get_portfolio_risk(self, account_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get risk metrics for the entire portfolio.
        
        Args:
            account_id: Optional account ID
            
        Returns:
            Risk metrics for the portfolio
        """
        return await self.risk_manager.get_portfolio_risk(account_id=account_id)
    
    async def track_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str,
        account_id: Optional[str] = None,
        leverage: float = 1.0
    ) -> bool:
        """
        Track a position for risk management.
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity
            price: Position price
            side: Position side
            account_id: Optional account ID
            leverage: Optional position leverage
            
        Returns:
            True if position was tracked successfully
        """
        return await self.risk_manager.add_position(
            symbol=symbol,
            quantity=quantity,
            price=price,
            side=side,
            account_id=account_id,
            leverage=leverage
        )
    
    async def update_tracked_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        account_id: Optional[str] = None
    ) -> bool:
        """
        Update a tracked position.
        
        Args:
            symbol: Trading symbol
            quantity: New position quantity
            price: New position price
            account_id: Optional account ID
            
        Returns:
            True if position was updated successfully
        """
        return await self.risk_manager.update_position(
            symbol=symbol,
            quantity=quantity,
            price=price,
            account_id=account_id
        )
    
    async def remove_tracked_position(
        self,
        symbol: str,
        account_id: Optional[str] = None
    ) -> bool:
        """
        Remove a tracked position.
        
        Args:
            symbol: Trading symbol
            account_id: Optional account ID
            
        Returns:
            True if position was removed successfully
        """
        return await self.risk_manager.remove_position(
            symbol=symbol,
            account_id=account_id
        )
    
    async def check_risk_limits(self, account_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Check if any risk limits are currently breached.
        
        Args:
            account_id: Optional account ID
            
        Returns:
            List of breached risk limits
        """
        return await self.risk_manager.check_risk_limits(account_id=account_id)
    
    async def set_risk_limit(
        self,
        limit_type: str,
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
        return await self.risk_manager.set_risk_limit(
            limit_type=limit_type,
            value=value,
            symbol=symbol,
            account_id=account_id
        )
    
    async def get_risk_metrics(self, account_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive risk metrics.
        
        Args:
            account_id: Optional account ID
            
        Returns:
            Comprehensive risk metrics
        """
        return await self.risk_manager.get_risk_metrics(account_id=account_id)
