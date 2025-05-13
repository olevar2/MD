"""
Trading Service Client

This module provides a client for the Trading Service.
"""

import logging
import json
from typing import Dict, Any, Optional, List

import httpx
from httpx import AsyncClient, Response

from common_lib.config.config_manager import ConfigManager
from common_lib.resilience.circuit_breaker import CircuitBreaker
from common_lib.resilience.retry import retry


class TradingService:
    """
    Client for the Trading Service.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the client.
        
        Args:
            logger: Logger to use (if None, creates a new logger)
        """
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config_manager = ConfigManager()
        
        # Get service client configuration
        try:
            service_clients = self.config_manager.get_service_clients_config()
            self.client_config = service_clients.trading_service
        except Exception as e:
            self.logger.warning(f"Error getting service client configuration: {str(e)}")
            self.client_config = None
        
        # Set default values
        self.base_url = getattr(self.client_config, "base_url", "http://localhost:8003") if self.client_config else "http://localhost:8003"
        self.timeout = getattr(self.client_config, "timeout", 30.0) if self.client_config else 30.0
        
        # Create circuit breaker
        self.circuit_breaker = CircuitBreaker(
            name="trading_service",
            failure_threshold=5,
            recovery_timeout=60.0,
            expected_exceptions=[httpx.HTTPError, httpx.TimeoutException],
            logger=self.logger
        )
    
    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Response:
        """
        Send a request to the service.
        
        Args:
            method: HTTP method
            path: Request path
            params: Query parameters
            json_data: JSON data
            headers: HTTP headers
            
        Returns:
            Response
            
        Raises:
            httpx.HTTPError: If the request fails
        """
        url = f"{self.base_url}{path}"
        
        # Add default headers
        headers = headers or {}
        headers["Content-Type"] = "application/json"
        
        # Send request with retry and circuit breaker
        @retry(
            retries=3,
            delay=1.0,
            backoff=2.0,
            exceptions=[httpx.HTTPError, httpx.TimeoutException]
        )
        async def send_request():
            async with AsyncClient(timeout=self.timeout) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_data,
                    headers=headers
                )
                response.raise_for_status()
                return response
        
        return await self.circuit_breaker.execute(send_request)
    
    async def place_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        quantity: float,
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Place order.
        
        Args:
            symbol: Symbol
            order_type: Order type
            side: Side
            quantity: Quantity
            price: Price
            
        Returns:
            Order
        """
        json_data = {
            "symbol": symbol,
            "order_type": order_type,
            "side": side,
            "quantity": quantity
        }
        
        if price is not None:
            json_data["price"] = price
        
        response = await self._request(
            "POST",
            "/api/v1/orders",
            json_data=json_data
        )
        return response.json()
    
    async def get_orders(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get orders.
        
        Args:
            symbol: Symbol
            status: Status
            start: Start timestamp in milliseconds
            end: End timestamp in milliseconds
            limit: Limit
            
        Returns:
            List of orders
        """
        params = {}
        if symbol is not None:
            params["symbol"] = symbol
        if status is not None:
            params["status"] = status
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end
        if limit is not None:
            params["limit"] = limit
        
        response = await self._request(
            "GET",
            "/api/v1/orders",
            params=params
        )
        return response.json()
    
    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order
        """
        response = await self._request(
            "GET",
            f"/api/v1/orders/{order_id}"
        )
        return response.json()
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order
        """
        response = await self._request(
            "DELETE",
            f"/api/v1/orders/{order_id}"
        )
        return response.json()
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get positions.
        
        Args:
            symbol: Symbol
            
        Returns:
            List of positions
        """
        params = {}
        if symbol is not None:
            params["symbol"] = symbol
        
        response = await self._request(
            "GET",
            "/api/v1/positions",
            params=params
        )
        return response.json()
    
    async def get_position(self, position_id: str) -> Dict[str, Any]:
        """
        Get position.
        
        Args:
            position_id: Position ID
            
        Returns:
            Position
        """
        response = await self._request(
            "GET",
            f"/api/v1/positions/{position_id}"
        )
        return response.json()
    
    async def close_position(self, position_id: str) -> Dict[str, Any]:
        """
        Close position.
        
        Args:
            position_id: Position ID
            
        Returns:
            Position
        """
        response = await self._request(
            "DELETE",
            f"/api/v1/positions/{position_id}"
        )
        return response.json()
    
    async def get_account(self) -> Dict[str, Any]:
        """
        Get account.
        
        Returns:
            Account
        """
        response = await self._request(
            "GET",
            "/api/v1/account"
        )
        return response.json()