"""
Order Book Provider Adapter Module

This module provides adapter implementations for order book provider interfaces,
helping to break circular dependencies between services.
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from common_lib.interfaces.trading import IOrderBookProvider
from core.logging import get_logger
from core.exceptions_bridge_1 import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from utils.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class OrderBookProviderAdapter(IOrderBookProvider):
    """
    Adapter implementation for the Order Book Provider interface.
    
    This adapter implements the Order Book Provider interface using the
    trading gateway service's internal components.
    """

    def __init__(self, logger: Optional[logging.Logger]=None):
        """
        Initialize the Order Book Provider adapter.
        
        Args:
            logger: Logger to use (if None, creates a new logger)
        """
        self.logger = logger or get_logger(
            f'{__name__}.{self.__class__.__name__}')
        from services.market_data_service import MarketDataService
        self.market_data_service = MarketDataService()

    @with_broker_api_resilience('get_order_book')
    @async_with_exception_handling
    async def get_order_book(self, symbol: str, depth: int=10) ->Dict[str, Any
        ]:
        """
        Get the order book for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "EUR/USD")
            depth: Depth of the order book
            
        Returns:
            Dictionary containing the order book data
        """
        try:
            order_book = await self.market_data_service.get_order_book(symbol
                =symbol, depth=depth)
            return order_book
        except Exception as e:
            self.logger.error(f'Error getting order book: {str(e)}')
            return {'symbol': symbol, 'timestamp': datetime.utcnow().
                isoformat(), 'bids': [], 'asks': []}

    @async_with_exception_handling
    async def subscribe_order_book(self, symbol: str, callback: callable,
        depth: int=10) ->str:
        """
        Subscribe to order book updates.
        
        Args:
            symbol: Trading symbol (e.g., "EUR/USD")
            callback: Callback function to receive updates
            depth: Depth of the order book
            
        Returns:
            Subscription ID
        """
        try:
            subscription_id = (await self.market_data_service.
                subscribe_order_book(symbol=symbol, callback=callback,
                depth=depth))
            return subscription_id
        except Exception as e:
            self.logger.error(f'Error subscribing to order book: {str(e)}')
            return ''

    @async_with_exception_handling
    async def unsubscribe_order_book(self, subscription_id: str) ->bool:
        """
        Unsubscribe from order book updates.
        
        Args:
            subscription_id: Subscription ID to unsubscribe
            
        Returns:
            True if unsubscribed successfully, False otherwise
        """
        try:
            success = await self.market_data_service.unsubscribe_order_book(
                subscription_id=subscription_id)
            return success
        except Exception as e:
            self.logger.error(f'Error unsubscribing from order book: {str(e)}')
            return False
