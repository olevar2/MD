"""
Mock broker adapter interface for testing.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
from trading_gateway_service.interfaces.broker_adapter_interface import OrderRequest, ExecutionReport


from trading_gateway_service.resilience.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class MockBrokerAdapterInterface(ABC):
    """
    Interface that all mock broker adapters must implement.
    """

    @abstractmethod
    def connect(self, credentials: Optional[Dict[str, str]]=None) ->bool:
        """
        Connect to the broker using the provided credentials.
        
        Args:
            credentials: Dictionary containing authentication details
            
        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def disconnect(self) ->bool:
        """
        Disconnect from the broker.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        pass

    @abstractmethod
    def is_connected(self) ->bool:
        """
        Check if the adapter is currently connected to the broker.
        
        Returns:
            True if connected, False otherwise
        """
        pass

    @with_broker_api_resilience('get_broker_info')
    @abstractmethod
    def get_broker_info(self) ->Dict[str, Any]:
        """
        Get information about the broker's capabilities and limitations.
        
        Returns:
            Dictionary containing broker details
        """
        pass

    @with_broker_api_resilience('get_account_info')
    @abstractmethod
    def get_account_info(self) ->Dict[str, Any]:
        """
        Get information about the trading account.
        
        Returns:
            Dictionary containing account details (balance, equity, margin, etc.)
        """
        pass

    @with_broker_api_resilience('get_positions')
    @abstractmethod
    def get_positions(self) ->List[Dict[str, Any]]:
        """
        Get all open positions.
        
        Returns:
            List of dictionaries containing position details
        """
        pass

    @with_broker_api_resilience('get_orders')
    @abstractmethod
    def get_orders(self) ->List[Dict[str, Any]]:
        """
        Get all pending orders.
        
        Returns:
            List of dictionaries containing order details
        """
        pass

    @abstractmethod
    def place_order(self, order: OrderRequest) ->ExecutionReport:
        """
        Place a new order with the broker.
        
        Args:
            order: OrderRequest object containing order details
            
        Returns:
            ExecutionReport with the result of the order placement
        """
        pass

    @abstractmethod
    def modify_order(self, order_id: str, modifications: Dict[str, Any]
        ) ->ExecutionReport:
        """
        Modify an existing order.
        
        Args:
            order_id: ID of the order to modify
            modifications: Dictionary of parameters to modify
            
        Returns:
            ExecutionReport with the result of the order modification
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) ->ExecutionReport:
        """
        Cancel an existing order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            ExecutionReport with the result of the cancellation
        """
        pass

    @with_market_data_resilience('get_market_data')
    @abstractmethod
    def get_market_data(self, instrument: str, data_type: Optional[str]=None
        ) ->Dict[str, Any]:
        """
        Get current market data for an instrument.
        
        Args:
            instrument: The instrument to get data for (e.g., "EURUSD")
            data_type: Optional type of data to retrieve
            
        Returns:
            Dictionary containing market data (bid, ask, spread, etc.)
        """
        pass
