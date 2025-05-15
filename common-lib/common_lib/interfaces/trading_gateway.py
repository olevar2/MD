"""
Trading Gateway Service Interface.

This module defines the interfaces for interacting with the Trading Gateway Service.
These interfaces allow other services to use Trading Gateway functionality without
direct dependencies, breaking circular dependencies.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

import pandas as pd

from common_lib.models.trading import (
    Order, Position, OrderStatus, OrderType, OrderSide, 
    ExecutionReport, MarketData, TradingAccount
)


class ITradingGateway(ABC):
    """Interface for trading gateway functionality."""
    
    @abstractmethod
    async def place_order(self, order: Order) -> ExecutionReport:
        """
        Place a trading order.
        
        Args:
            order: Order details
            
        Returns:
            Execution report with order status
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> ExecutionReport:
        """
        Cancel an existing order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            Execution report with cancellation status
        """
        pass
    
    @abstractmethod
    async def modify_order(self, order_id: str, modifications: Dict[str, Any]) -> ExecutionReport:
        """
        Modify an existing order.
        
        Args:
            order_id: ID of the order to modify
            modifications: Dictionary of modifications to apply
            
        Returns:
            Execution report with modification status
        """
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Get the current status of an order.
        
        Args:
            order_id: ID of the order
            
        Returns:
            Current order status
        """
        pass
    
    @abstractmethod
    async def get_open_orders(self, account_id: Optional[str] = None) -> List[Order]:
        """
        Get all open orders.
        
        Args:
            account_id: Optional account ID to filter by
            
        Returns:
            List of open orders
        """
        pass
    
    @abstractmethod
    async def get_positions(self, account_id: Optional[str] = None) -> List[Position]:
        """
        Get all open positions.
        
        Args:
            account_id: Optional account ID to filter by
            
        Returns:
            List of open positions
        """
        pass
    
    @abstractmethod
    async def get_market_data(
        self, 
        symbol: str, 
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> MarketData:
        """
        Get market data for a symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the data
            start_time: Optional start time for the data
            end_time: Optional end time for the data
            limit: Optional limit on the number of data points
            
        Returns:
            Market data
        """
        pass
    
    @abstractmethod
    async def get_account_info(self, account_id: str) -> TradingAccount:
        """
        Get information about a trading account.
        
        Args:
            account_id: ID of the account
            
        Returns:
            Account information
        """
        pass
    
    @abstractmethod
    async def get_available_symbols(self) -> List[Dict[str, Any]]:
        """
        Get a list of available trading symbols.
        
        Returns:
            List of available symbols with metadata
        """
        pass
    
    @abstractmethod
    async def get_trading_hours(self, symbol: str) -> Dict[str, Any]:
        """
        Get trading hours for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with trading hours information
        """
        pass
