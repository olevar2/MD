"""
Trading Gateway Client for Strategy Execution Engine

This module provides a standardized client for interacting with the Trading Gateway Service.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from common_lib.clients.base_client import BaseServiceClient, ClientConfig
from common_lib.clients.exceptions import (
    ClientError,
    ClientConnectionError,
    ClientTimeoutError,
    ClientValidationError,
    ClientAuthenticationError
)
from common_lib.correlation import get_correlation_id

from config.config_1 import get_settings
from core.error import (
    ServiceError,
    DataFetchError,
    async_with_error_handling
)

logger = logging.getLogger(__name__)

# Client instance for singleton pattern
_trading_gateway_client = None


class TradingGatewayClient(BaseServiceClient):
    """
    Client for interacting with the Trading Gateway Service.

    This client follows the platform's standardized patterns for service communication,
    including resilience patterns, error handling, metrics collection, and logging.

    Features:
    1. Built-in resilience patterns (circuit breaker, retry, timeout, bulkhead)
    2. Standardized error handling
    3. Metrics collection
    4. Structured logging with correlation IDs
    """

    def __init__(self, config: Union[ClientConfig, Dict[str, Any]]):
        """
        Initialize the client.

        Args:
            config: Client configuration
        """
        super().__init__(config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Trading Gateway Client initialized with base URL: {self.base_url}")

    @async_with_error_handling
    async def execute_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trading order.

        Args:
            order: Order details

        Returns:
            Dict: Order execution result

        Raises:
            ClientError: If the request fails
            ClientConnectionError: If connection to the service fails
            ClientTimeoutError: If the request times out
            ClientValidationError: If the order data is invalid
            ClientAuthenticationError: If authentication fails
            ServiceError: If service returns an error
        """
        self.logger.debug(f"Executing order: {order.get('order_id', 'unknown')}")
        try:
            return await self.post("orders", data=order)
        except ClientError as e:
            self.logger.error(f"Failed to execute order: {str(e)}")
            # Map to domain-specific exception for backward compatibility
            raise ServiceError(f"Trading Gateway Service error: {str(e)}") from e

    @async_with_error_handling
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get the status of an order.

        Args:
            order_id: Order ID

        Returns:
            Dict: Order status

        Raises:
            ClientError: If the request fails
            ClientConnectionError: If connection to the service fails
            ClientTimeoutError: If the request times out
            ClientValidationError: If the order ID is invalid
            ClientAuthenticationError: If authentication fails
            DataFetchError: If order not found
            ServiceError: If service returns an error
        """
        self.logger.debug(f"Getting order status: {order_id}")
        try:
            return await self.get(f"orders/{order_id}")
        except ClientError as e:
            self.logger.error(f"Failed to get order status for {order_id}: {str(e)}")
            # Map to domain-specific exceptions for backward compatibility
            if "not found" in str(e).lower() or getattr(e, "status_code", 0) == 404:
                raise DataFetchError(f"Order not found: {order_id}") from e
            raise ServiceError(f"Trading Gateway Service error: {str(e)}") from e

    @async_with_error_handling
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.

        Returns:
            Dict: Account information

        Raises:
            ClientError: If the request fails
            ClientConnectionError: If connection to the service fails
            ClientTimeoutError: If the request times out
            ClientAuthenticationError: If authentication fails
            ServiceError: If service returns an error
        """
        self.logger.debug("Getting account information")
        try:
            return await self.get("account")
        except ClientError as e:
            self.logger.error(f"Failed to get account info: {str(e)}")
            # Map to domain-specific exception for backward compatibility
            raise ServiceError(f"Trading Gateway Service error: {str(e)}") from e

    @async_with_error_handling
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get open positions.

        Returns:
            List: Open positions

        Raises:
            ClientError: If the request fails
            ClientConnectionError: If connection to the service fails
            ClientTimeoutError: If the request times out
            ClientAuthenticationError: If authentication fails
            ServiceError: If service returns an error
        """
        self.logger.debug("Getting open positions")
        try:
            response = await self.get("positions")
            # Handle different response formats
            if isinstance(response, list):
                return response
            elif isinstance(response, dict) and "positions" in response:
                return response["positions"]
            else:
                self.logger.warning(f"Unexpected response format: {response}")
                return []
        except ClientError as e:
            self.logger.error(f"Failed to get positions: {str(e)}")
            # Map to domain-specific exception for backward compatibility
            raise ServiceError(f"Trading Gateway Service error: {str(e)}") from e

    @async_with_error_handling
    async def get_market_data(
        self,
        instrument: str,
        timeframe: str = "1m",
        count: int = 100
    ) -> Dict[str, Any]:
        """
        Get market data for an instrument.

        Args:
            instrument: Instrument symbol
            timeframe: Timeframe (e.g., "1m", "5m", "1h")
            count: Number of candles to retrieve

        Returns:
            Dict: Market data

        Raises:
            ClientError: If the request fails
            ClientConnectionError: If connection to the service fails
            ClientTimeoutError: If the request times out
            ClientValidationError: If the parameters are invalid
            ClientAuthenticationError: If authentication fails
            DataFetchError: If data not found
            ServiceError: If service returns an error
        """
        self.logger.debug(f"Getting market data for {instrument} on {timeframe} timeframe")
        try:
            params = {
                "instrument": instrument,
                "timeframe": timeframe,
                "count": count
            }
            return await self.get("market-data", params=params)
        except ClientError as e:
            self.logger.error(f"Failed to get market data for {instrument}: {str(e)}")
            # Map to domain-specific exceptions for backward compatibility
            if "not found" in str(e).lower() or getattr(e, "status_code", 0) == 404:
                raise DataFetchError(f"Market data not found for {instrument}") from e
            raise ServiceError(f"Trading Gateway Service error: {str(e)}") from e

    @async_with_error_handling
    async def check_health(self) -> Dict[str, Any]:
        """
        Check the health of the Trading Gateway Service.

        Returns:
            Dict: Health check result

        Raises:
            ServiceError: If service is unhealthy or unreachable
        """
        self.logger.debug("Checking trading gateway service health")
        try:
            health_data = await self.get("health")
            return {
                "status": "healthy",
                "message": "Connection successful",
                "details": health_data
            }
        except Exception as e:
            self.logger.error(f"Failed to check trading gateway service health: {str(e)}")
            return {
                "status": "unhealthy",
                "message": f"Health check failed: {str(e)}",
                "details": None
            }

    # Alias methods for backward compatibility
    async def submit_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for execute_order for backward compatibility."""
        return await self.execute_order(order)

    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """Alias for get_positions for backward compatibility."""
        return await self.get_positions()

    async def close(self) -> None:
        """
        Close the client session.

        This method is provided for backward compatibility.
        The BaseServiceClient handles session management automatically.
        """
        # BaseServiceClient handles session management, so this is a no-op
        pass


def get_trading_gateway_client() -> TradingGatewayClient:
    """
    Get the Trading Gateway client.

    Returns:
        TradingGatewayClient: Trading Gateway client instance
    """
    global _trading_gateway_client

    if _trading_gateway_client is None:
        settings = get_settings()

        # Create client configuration
        config = ClientConfig(
            base_url=settings.trading_gateway_url,
            service_name="trading-gateway-service",
            api_key=settings.trading_gateway_key,
            timeout_seconds=settings.client_timeout_seconds,
            max_retries=settings.client_max_retries,
            retry_base_delay=settings.client_retry_base_delay,
            retry_backoff_factor=settings.client_retry_backoff_factor,
            circuit_breaker_failure_threshold=settings.client_circuit_breaker_threshold,
            circuit_breaker_reset_timeout_seconds=settings.client_circuit_breaker_reset_timeout
        )

        # Create client instance
        _trading_gateway_client = TradingGatewayClient(config)

    return _trading_gateway_client


def get_trading_gateway_client_with_config(config: Dict[str, Any]) -> TradingGatewayClient:
    """
    Get a Trading Gateway client with custom configuration.

    Args:
        config: Custom client configuration

    Returns:
        TradingGatewayClient: Trading Gateway client instance
    """
    # Create client instance with custom configuration
    return TradingGatewayClient(config)
