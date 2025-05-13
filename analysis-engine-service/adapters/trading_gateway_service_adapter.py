"""
Adapter implementation for trading-gateway-service service.
"""

from typing import Dict, List, Optional, Any
import requests
from common_lib.exceptions import ServiceException
from common_lib.resilience import circuit_breaker, retry_with_backoff
from ..interfaces.trading_gateway_service_interface import TradingGatewayServiceInterface


class TradingGatewayServiceAdapter(TradingGatewayServiceInterface):
    """
    Adapter for trading-gateway-service service.
    """

    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize the adapter.

        Args:
            base_url: Base URL of the service
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.timeout = timeout

    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the service.

        Returns:
            Service status information
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/status",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ServiceException(f"Error getting service status: {str(e)}")

    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    def execute_trade(trade_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trade.

        Args:
            trade_request: Trade request details
        Returns:
            Trade execution result
        """
        try:
            # Implement the method using requests to call the service API
            # This is a placeholder implementation
            response = requests.get(
                f"{self.base_url}/api/trades",
                params={},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ServiceException(f"Error calling execute_trade: {str(e)}")

    @retry_with_backoff(max_retries=3, backoff_factor=1.5)
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    def get_trade_status(trade_id: str) -> Dict[str, Any]:
        """
        Get the status of a trade.

        Args:
            trade_id: Trade identifier
        Returns:
            Trade status information
        """
        try:
            # Implement the method using requests to call the service API
            # This is a placeholder implementation
            response = requests.get(
                f"{self.base_url}/api/trades/" + str(trade_id) + "",
                params={},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ServiceException(f"Error calling get_trade_status: {str(e)}")

