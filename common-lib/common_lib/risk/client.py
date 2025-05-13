"""
Risk Management Client Module

This module provides a standardized client for interacting with the risk management service,
helping to break circular dependencies between services.
"""
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from common_lib.service_client.http_client import AsyncHTTPServiceClient
from common_lib.service_client.base_client import ServiceClientConfig
from common_lib.interfaces.risk_management import (
    IRiskManager, RiskCheckResult, RiskLimitType
)
from common_lib.interfaces.trading import OrderType, OrderSide
from common_lib.resilience import (
    retry_with_backoff, circuit_breaker, timeout, bulkhead
)
from common_lib.risk.models import (
    RiskMetrics, RiskParameters, RiskProfile, RiskLimit, PositionRisk
)

logger = logging.getLogger(__name__)


class RiskManagementClient(IRiskManager):
    """
    Client for the Risk Management Service.
    
    This client provides a standardized interface for interacting with the
    Risk Management Service, with built-in resilience patterns.
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        config: Optional[ServiceClientConfig] = None,
        client: Optional[AsyncHTTPServiceClient] = None
    ):
        """
        Initialize the Risk Management Client.
        
        Args:
            base_url: Optional base URL for the Risk Management Service
            config: Optional client configuration
            client: Optional pre-configured HTTP client
        """
        self.config = config or ServiceClientConfig(
            base_url=base_url or "http://risk-management-service:8000/api/v1",
            timeout=30,
            retry_config={
                "max_retries": 3,
                "backoff_factor": 0.5,
                "retry_statuses": [408, 429, 500, 502, 503, 504]
            },
            circuit_breaker_config={
                "failure_threshold": 5,
                "recovery_timeout": 30,
                "expected_exception_types": [
                    "ConnectionError",
                    "Timeout",
                    "TooManyRedirects"
                ]
            }
        )
        
        self.client = client or AsyncHTTPServiceClient(self.config)
        logger.info(f"Initialized Risk Management Client with base URL: {self.config.base_url}")
        
        # Local cache for fallback mode
        self._risk_limits = {}
        self._positions = {}
    
    @retry_with_backoff()
    @circuit_breaker()
    @timeout()
    @bulkhead()
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
        try:
            # Convert enum values to strings if needed
            order_type_str = order_type.value if isinstance(order_type, OrderType) else order_type
            side_str = side.value if isinstance(side, OrderSide) else side
            
            # Prepare request data
            data = {
                "symbol": symbol,
                "order_type": order_type_str,
                "side": side_str,
                "quantity": quantity
            }
            
            if price is not None:
                data["price"] = price
                
            if account_id is not None:
                data["account_id"] = account_id
            
            # Make API call
            response = await self.client.post("/risk/check-order", json=data)
            
            # Convert response to RiskCheckResult
            return RiskCheckResult.from_dict(response)
        except Exception as e:
            logger.error(f"Error checking order with risk management service: {str(e)}")
            # Fallback to local risk check
            return await self._fallback_check_order(
                symbol, order_type_str, side_str, quantity, price, account_id
            )
    
    @retry_with_backoff()
    @circuit_breaker()
    @timeout()
    @bulkhead()
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
        try:
            # Prepare query parameters
            params = {}
            if symbol is not None:
                params["symbol"] = symbol
            if account_id is not None:
                params["account_id"] = account_id
            
            # Make API call
            response = await self.client.get("/risk/position", params=params)
            return response
        except Exception as e:
            logger.error(f"Error getting position risk from risk management service: {str(e)}")
            # Fallback to local position risk
            return self._fallback_get_position_risk(symbol, account_id)
    
    @retry_with_backoff()
    @circuit_breaker()
    @timeout()
    @bulkhead()
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
        try:
            # Prepare query parameters
            params = {}
            if account_id is not None:
                params["account_id"] = account_id
            
            # Make API call
            response = await self.client.get("/risk/portfolio", params=params)
            return response
        except Exception as e:
            logger.error(f"Error getting portfolio risk from risk management service: {str(e)}")
            # Fallback to local portfolio risk
            return self._fallback_get_portfolio_risk(account_id)
    
    # Implement the remaining IRiskManager methods with similar patterns
    # ...
    
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
        
        # Simple check: is quantity <= max_position_size?
        is_valid = quantity <= max_position_size
        message = "Order validated" if is_valid else f"Order exceeds maximum position size of {max_position_size}"
        
        return RiskCheckResult(
            is_valid=is_valid,
            message=message,
            details={"symbol": symbol, "quantity": quantity, "max_position_size": max_position_size},
            breached_limits=[{"type": RiskLimitType.MAX_POSITION_SIZE.value, "value": max_position_size}] if not is_valid else []
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
        total_exposure = 0.0
        for key, position in self._positions.items():
            pos_account_id, pos_symbol = key.split(":")
            if account_id is None or pos_account_id == account_id:
                positions[pos_symbol] = position
                total_exposure += abs(position.get("exposure", 0.0))
        
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
"""
