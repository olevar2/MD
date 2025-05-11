"""
Market Data Service Client

This module provides a client for the Market Data Service.
"""

import logging
import json
from typing import Dict, Any, Optional, List

import httpx
from httpx import AsyncClient, Response

from common_lib.config.config_manager import ConfigManager
from common_lib.resilience.circuit_breaker import CircuitBreaker
from common_lib.resilience.retry import retry


class MarketDataService:
    """
    Client for the Market Data Service.
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
            self.client_config = service_clients.market_data_service
        except Exception as e:
            self.logger.warning(f"Error getting service client configuration: {str(e)}")
            self.client_config = None
        
        # Set default values
        self.base_url = getattr(self.client_config, "base_url", "http://localhost:8001") if self.client_config else "http://localhost:8001"
        self.timeout = getattr(self.client_config, "timeout", 30.0) if self.client_config else 30.0
        
        # Create circuit breaker
        self.circuit_breaker = CircuitBreaker(
            name="market_data_service",
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
    
    async def get_symbols(self) -> List[Dict[str, Any]]:
        """
        Get available symbols.
        
        Returns:
            List of available symbols
        """
        response = await self._request("GET", "/api/v1/symbols")
        return response.json()
    
    async def get_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Get symbol information.
        
        Args:
            symbol: Symbol
            
        Returns:
            Symbol information
        """
        response = await self._request("GET", f"/api/v1/symbols/{symbol}")
        return response.json()
    
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get market data.
        
        Args:
            symbol: Symbol
            timeframe: Timeframe
            start: Start timestamp in milliseconds
            end: End timestamp in milliseconds
            limit: Limit
            
        Returns:
            Market data
        """
        params = {}
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end
        if limit is not None:
            params["limit"] = limit
        
        response = await self._request(
            "GET",
            f"/api/v1/data/{symbol}/{timeframe}",
            params=params
        )
        return response.json()
    
    async def get_latest_market_data(
        self,
        symbol: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """
        Get latest market data.
        
        Args:
            symbol: Symbol
            timeframe: Timeframe
            
        Returns:
            Latest market data
        """
        response = await self._request(
            "GET",
            f"/api/v1/latest/{symbol}/{timeframe}"
        )
        return response.json()