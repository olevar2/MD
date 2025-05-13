"""
Interface definition for trading-gateway-service service.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class TradingGatewayServiceInterface(ABC):
    """
    Interface for trading-gateway-service service.
    """

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the service.

        Returns:
            Service status information
        """
        pass

    @abstractmethod
    def execute_trade(trade_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trade.

        Args:
            trade_request: Trade request details
        Returns:
            Trade execution result
        """
        pass

    @abstractmethod
    def get_trade_status(trade_id: str) -> Dict[str, Any]:
        """
        Get the status of a trade.

        Args:
            trade_id: Trade identifier
        Returns:
            Trade status information
        """
        pass

