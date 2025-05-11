"""
Risk Management Service Adapter.

This module provides adapter implementations for the Risk Management Service interfaces.
These adapters allow other services to interact with the Risk Management Service
without direct dependencies, breaking circular dependencies.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from common_lib.interfaces.risk_management import (
    IRiskManager, RiskLimitType, RiskCheckResult
)
from common_lib.interfaces.trading import OrderType, OrderSide
from common_lib.service_client.base_client import ServiceClientConfig
from common_lib.service_client.http_client import AsyncHTTPServiceClient
from common_lib.errors.base_exceptions import (
    ServiceUnavailableError, ResourceNotFoundError, ValidationError
)

logger = logging.getLogger(__name__)


class RiskManagementAdapter(IRiskManager):
    """Adapter for Risk Management Service operations."""
    
    def __init__(self, client: Optional[AsyncHTTPServiceClient] = None, config: Optional[ServiceClientConfig] = None):
        """
        Initialize the adapter.
        
        Args:
            client: Optional pre-configured HTTP client
            config: Optional client configuration
        """
        self.client = client or AsyncHTTPServiceClient(
            config or ServiceClientConfig(
                base_url="http://risk-management-service:8000/api/v1",
                timeout=30
            )
        )
        
        # Local cache for risk limits
        self._risk_limits = {}
        self._positions = {}
    
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
        # Convert enum values to strings if needed
        order_type_str = order_type.value if isinstance(order_type, OrderType) else order_type
        side_str = side.value if isinstance(side, OrderSide) else side
        
        # Prepare request payload
        payload = {
            "symbol": symbol,
            "order_type": order_type_str,
            "side": side_str,
            "quantity": quantity
        }
        
        if price is not None:
            payload["price"] = price
        
        if account_id:
            payload["account_id"] = account_id
        
        try:
            # Make API call to risk management service
            response = await self.client.post("/risk/check-order", json=payload)
            
            # Convert response to RiskCheckResult
            return RiskCheckResult(
                is_valid=response.get("is_valid", False),
                message=response.get("message", ""),
                details=response.get("details"),
                breached_limits=response.get("breached_limits")
            )
        except ServiceUnavailableError:
            logger.error("Risk management service unavailable, using fallback")
            # Fallback to local risk check
            return await self._fallback_check_order(symbol, order_type_str, side_str, quantity, price, account_id)
        except Exception as e:
            logger.error(f"Error checking order with risk management service: {str(e)}")
            # Fallback to local risk check
            return await self._fallback_check_order(symbol, order_type_str, side_str, quantity, price, account_id)
    
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
        params = {}
        if symbol:
            params["symbol"] = symbol
        if account_id:
            params["account_id"] = account_id
        
        try:
            # Make API call to risk management service
            response = await self.client.get("/risk/position", params=params)
            return response
        except ServiceUnavailableError:
            logger.error("Risk management service unavailable, using fallback")
            # Fallback to local position risk
            return self._fallback_get_position_risk(symbol, account_id)
        except Exception as e:
            logger.error(f"Error getting position risk from risk management service: {str(e)}")
            # Fallback to local position risk
            return self._fallback_get_position_risk(symbol, account_id)
    
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
        params = {}
        if account_id:
            params["account_id"] = account_id
        
        try:
            # Make API call to risk management service
            response = await self.client.get("/risk/portfolio", params=params)
            return response
        except ServiceUnavailableError:
            logger.error("Risk management service unavailable, using fallback")
            # Fallback to local portfolio risk
            return self._fallback_get_portfolio_risk(account_id)
        except Exception as e:
            logger.error(f"Error getting portfolio risk from risk management service: {str(e)}")
            # Fallback to local portfolio risk
            return self._fallback_get_portfolio_risk(account_id)
    
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
        # Convert enum value to string if needed
        limit_type_str = limit_type.value if isinstance(limit_type, RiskLimitType) else limit_type
        
        # Prepare request payload
        payload = {
            "limit_type": limit_type_str,
            "value": value
        }
        
        if symbol:
            payload["symbol"] = symbol
        
        if account_id:
            payload["account_id"] = account_id
        
        try:
            # Make API call to risk management service
            response = await self.client.post("/risk/limit", json=payload)
            
            # Update local cache
            key = f"{account_id or 'default'}:{symbol or 'global'}:{limit_type_str}"
            self._risk_limits[key] = value
            
            return response.get("success", False)
        except Exception as e:
            logger.error(f"Error setting risk limit with risk management service: {str(e)}")
            
            # Update local cache anyway
            key = f"{account_id or 'default'}:{symbol or 'global'}:{limit_type_str}"
            self._risk_limits[key] = value
            
            return False
    
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
        # Convert enum value to string if needed
        limit_type_str = limit_type.value if isinstance(limit_type, RiskLimitType) else limit_type
        
        params = {}
        if limit_type_str:
            params["limit_type"] = limit_type_str
        if symbol:
            params["symbol"] = symbol
        if account_id:
            params["account_id"] = account_id
        
        try:
            # Make API call to risk management service
            response = await self.client.get("/risk/limits", params=params)
            return response
        except ServiceUnavailableError:
            logger.error("Risk management service unavailable, using fallback")
            # Fallback to local risk limits
            return self._fallback_get_risk_limits(limit_type_str, symbol, account_id)
        except Exception as e:
            logger.error(f"Error getting risk limits from risk management service: {str(e)}")
            # Fallback to local risk limits
            return self._fallback_get_risk_limits(limit_type_str, symbol, account_id)
    
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
        params = {}
        if account_id:
            params["account_id"] = account_id
        
        try:
            # Make API call to risk management service
            response = await self.client.get("/risk/check-limits", params=params)
            return response.get("breached_limits", [])
        except ServiceUnavailableError:
            logger.error("Risk management service unavailable, using fallback")
            # Fallback to local risk limit check
            return self._fallback_check_risk_limits(account_id)
        except Exception as e:
            logger.error(f"Error checking risk limits with risk management service: {str(e)}")
            # Fallback to local risk limit check
            return self._fallback_check_risk_limits(account_id)
    
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
        # Convert enum value to string if needed
        side_str = side.value if isinstance(side, OrderSide) else side
        
        # Prepare request payload
        payload = {
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "side": side_str,
            "leverage": leverage
        }
        
        if account_id:
            payload["account_id"] = account_id
        
        try:
            # Make API call to risk management service
            response = await self.client.post("/risk/position", json=payload)
            
            # Update local cache
            key = f"{account_id or 'default'}:{symbol}"
            self._positions[key] = {
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "side": side_str,
                "leverage": leverage
            }
            
            return response.get("success", False)
        except Exception as e:
            logger.error(f"Error adding position with risk management service: {str(e)}")
            
            # Update local cache anyway
            key = f"{account_id or 'default'}:{symbol}"
            self._positions[key] = {
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "side": side_str,
                "leverage": leverage
            }
            
            return False
    
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
        # Prepare request payload
        payload = {
            "symbol": symbol,
            "quantity": quantity,
            "price": price
        }
        
        if account_id:
            payload["account_id"] = account_id
        
        try:
            # Make API call to risk management service
            response = await self.client.put("/risk/position", json=payload)
            
            # Update local cache
            key = f"{account_id or 'default'}:{symbol}"
            if key in self._positions:
                self._positions[key].update({
                    "quantity": quantity,
                    "price": price
                })
            
            return response.get("success", False)
        except Exception as e:
            logger.error(f"Error updating position with risk management service: {str(e)}")
            
            # Update local cache anyway
            key = f"{account_id or 'default'}:{symbol}"
            if key in self._positions:
                self._positions[key].update({
                    "quantity": quantity,
                    "price": price
                })
            
            return False
    
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
        params = {
            "symbol": symbol
        }
        
        if account_id:
            params["account_id"] = account_id
        
        try:
            # Make API call to risk management service
            response = await self.client.delete("/risk/position", params=params)
            
            # Update local cache
            key = f"{account_id or 'default'}:{symbol}"
            if key in self._positions:
                del self._positions[key]
            
            return response.get("success", False)
        except Exception as e:
            logger.error(f"Error removing position with risk management service: {str(e)}")
            
            # Update local cache anyway
            key = f"{account_id or 'default'}:{symbol}"
            if key in self._positions:
                del self._positions[key]
            
            return False
    
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
        params = {}
        if account_id:
            params["account_id"] = account_id
        
        try:
            # Make API call to risk management service
            response = await self.client.get("/risk/metrics", params=params)
            return response
        except ServiceUnavailableError:
            logger.error("Risk management service unavailable, using fallback")
            # Fallback to local risk metrics
            return self._fallback_get_risk_metrics(account_id)
        except Exception as e:
            logger.error(f"Error getting risk metrics from risk management service: {str(e)}")
            # Fallback to local risk metrics
            return self._fallback_get_risk_metrics(account_id)
    
    # Fallback methods for degraded mode
    
    async def _fallback_check_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        account_id: Optional[str] = None
    ) -> RiskCheckResult:
        """Fallback method for checking orders when service is unavailable."""
        logger.info(f"Using fallback risk check for {symbol} {side} {quantity}")
        
        # Apply conservative risk limits
        max_position_size = 1000.0  # Conservative default
        
        # Check if we have a cached limit
        key = f"{account_id or 'default'}:{symbol}:{RiskLimitType.MAX_POSITION_SIZE.value}"
        if key in self._risk_limits:
            max_position_size = self._risk_limits[key]
        
        # Simple check based on position size
        if quantity > max_position_size:
            return RiskCheckResult(
                is_valid=False,
                message=f"Order quantity {quantity} exceeds maximum position size {max_position_size}",
                details={
                    "max_position_size": max_position_size,
                    "order_quantity": quantity
                },
                breached_limits=[
                    {
                        "limit_type": RiskLimitType.MAX_POSITION_SIZE.value,
                        "limit_value": max_position_size,
                        "current_value": quantity,
                        "symbol": symbol
                    }
                ]
            )
        
        return RiskCheckResult(
            is_valid=True,
            message="Order passed fallback risk check",
            details={
                "fallback": True,
                "max_position_size": max_position_size,
                "order_quantity": quantity
            }
        )
    
    def _fallback_get_position_risk(
        self,
        symbol: Optional[str] = None,
        account_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fallback method for getting position risk when service is unavailable."""
        logger.info(f"Using fallback position risk for {symbol}")
        
        # Filter positions by symbol and account_id
        positions = {}
        for key, position in self._positions.items():
            pos_account_id, pos_symbol = key.split(":")
            if (symbol is None or pos_symbol == symbol) and (account_id is None or pos_account_id == account_id):
                positions[pos_symbol] = position
        
        # Calculate simple risk metrics
        result = {
            "positions": positions,
            "risk_metrics": {
                "var_95": 0.0,
                "expected_shortfall": 0.0,
                "max_drawdown": 0.0
            }
        }
        
        return result
    
    def _fallback_get_portfolio_risk(
        self,
        account_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fallback method for getting portfolio risk when service is unavailable."""
        logger.info("Using fallback portfolio risk")
        
        # Filter positions by account_id
        positions = {}
        for key, position in self._positions.items():
            pos_account_id, pos_symbol = key.split(":")
            if account_id is None or pos_account_id == account_id:
                positions[pos_symbol] = position
        
        # Calculate simple portfolio metrics
        total_exposure = sum(
            abs(position["quantity"] * position["price"] * position["leverage"])
            for position in positions.values()
        )
        
        result = {
            "total_exposure": total_exposure,
            "risk_metrics": {
                "portfolio_var_95": 0.0,
                "portfolio_expected_shortfall": 0.0,
                "portfolio_sharpe_ratio": 0.0,
                "portfolio_max_drawdown": 0.0
            }
        }
        
        return result
    
    def _fallback_get_risk_limits(
        self,
        limit_type: Optional[str] = None,
        symbol: Optional[str] = None,
        account_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fallback method for getting risk limits when service is unavailable."""
        logger.info("Using fallback risk limits")
        
        # Filter limits by type, symbol, and account_id
        limits = {}
        for key, value in self._risk_limits.items():
            parts = key.split(":")
            if len(parts) == 3:
                limit_account_id, limit_symbol, limit_type_str = parts
                if (limit_type is None or limit_type_str == limit_type) and \
                   (symbol is None or limit_symbol == symbol or limit_symbol == "global") and \
                   (account_id is None or limit_account_id == account_id or limit_account_id == "default"):
                    if limit_symbol not in limits:
                        limits[limit_symbol] = {}
                    limits[limit_symbol][limit_type_str] = value
        
        return {"limits": limits}
    
    def _fallback_check_risk_limits(
        self,
        account_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fallback method for checking risk limits when service is unavailable."""
        logger.info("Using fallback risk limit check")
        
        # Simple implementation that assumes no limits are breached
        return []
    
    def _fallback_get_risk_metrics(
        self,
        account_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fallback method for getting risk metrics when service is unavailable."""
        logger.info("Using fallback risk metrics")
        
        # Combine position and portfolio risk
        position_risk = self._fallback_get_position_risk(account_id=account_id)
        portfolio_risk = self._fallback_get_portfolio_risk(account_id=account_id)
        
        return {
            "positions": position_risk.get("positions", {}),
            "portfolio": {
                "total_exposure": portfolio_risk.get("total_exposure", 0.0)
            },
            "risk_metrics": {
                **position_risk.get("risk_metrics", {}),
                **portfolio_risk.get("risk_metrics", {})
            }
        }
