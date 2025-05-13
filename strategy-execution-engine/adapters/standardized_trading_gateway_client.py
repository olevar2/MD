"""
Standardized Trading Gateway Client for Strategy Execution Engine

This module provides a standardized client for interacting with the Trading Gateway Service
following the platform's standardized patterns for service communication.
"""
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from common_lib.clients.base_client import BaseServiceClient, ClientConfig
from common_lib.clients.exceptions import ClientError, ClientConnectionError, ClientTimeoutError, ClientValidationError, ClientAuthenticationError
from common_lib.error import ServiceError, DataFetchError
logger = logging.getLogger(__name__)


from core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class StandardizedTradingGatewayClient(BaseServiceClient):
    """
    Standardized client for interacting with the Trading Gateway Service.
    
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
        Initialize the standardized trading gateway client.
        
        Args:
            config: Client configuration
        """
        super().__init__(config)
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}'
            )
        self.logger.info(
            f'Standardized Trading Gateway Client initialized with base URL: {self.base_url}'
            )

    @async_with_exception_handling
    async def execute_order(self, order: Dict[str, Any]) ->Dict[str, Any]:
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
        """
        self.logger.debug(
            f"Executing order: {order.get('order_id', 'unknown')}")
        try:
            return await self.post('orders', data=order)
        except Exception as e:
            self.logger.error(f'Failed to execute order: {str(e)}')
            raise

    @async_with_exception_handling
    async def get_order_status(self, order_id: str) ->Dict[str, Any]:
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
        """
        self.logger.debug(f'Getting order status: {order_id}')
        try:
            return await self.get(f'orders/{order_id}')
        except Exception as e:
            self.logger.error(
                f'Failed to get order status for {order_id}: {str(e)}')
            if isinstance(e, ClientError) and 'not found' in str(e).lower():
                raise DataFetchError(f'Order not found: {order_id}') from e
            raise

    @async_with_exception_handling
    async def get_account_info(self) ->Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dict: Account information
            
        Raises:
            ClientError: If the request fails
            ClientConnectionError: If connection to the service fails
            ClientTimeoutError: If the request times out
            ClientAuthenticationError: If authentication fails
        """
        self.logger.debug('Getting account information')
        try:
            return await self.get('account')
        except Exception as e:
            self.logger.error(f'Failed to get account info: {str(e)}')
            raise

    @async_with_exception_handling
    async def get_positions(self) ->List[Dict[str, Any]]:
        """
        Get open positions.
        
        Returns:
            List: Open positions
            
        Raises:
            ClientError: If the request fails
            ClientConnectionError: If connection to the service fails
            ClientTimeoutError: If the request times out
            ClientAuthenticationError: If authentication fails
        """
        self.logger.debug('Getting open positions')
        try:
            response = await self.get('positions')
            if isinstance(response, list):
                return response
            elif isinstance(response, dict) and 'positions' in response:
                return response['positions']
            else:
                self.logger.warning(f'Unexpected response format: {response}')
                return []
        except Exception as e:
            self.logger.error(f'Failed to get positions: {str(e)}')
            raise

    @async_with_exception_handling
    async def get_market_data(self, instrument: str, timeframe: str='1m',
        count: int=100) ->Dict[str, Any]:
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
        """
        self.logger.debug(
            f'Getting market data for {instrument} on {timeframe} timeframe')
        try:
            params = {'instrument': instrument, 'timeframe': timeframe,
                'count': count}
            return await self.get('market-data', params=params)
        except Exception as e:
            self.logger.error(
                f'Failed to get market data for {instrument}: {str(e)}')
            if isinstance(e, ClientError) and 'not found' in str(e).lower():
                raise DataFetchError(f'Market data not found for {instrument}'
                    ) from e
            raise

    @async_with_exception_handling
    async def cancel_order(self, order_id: str) ->Dict[str, Any]:
        """
        Cancel an existing order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            Dictionary with cancellation result
            
        Raises:
            ClientError: If the request fails
            ClientConnectionError: If connection to the service fails
            ClientTimeoutError: If the request times out
            ClientValidationError: If the order ID is invalid
            ClientAuthenticationError: If authentication fails
        """
        self.logger.debug(f'Cancelling order: {order_id}')
        try:
            return await self.post(f'orders/{order_id}/cancel')
        except Exception as e:
            self.logger.error(f'Failed to cancel order {order_id}: {str(e)}')
            raise

    @async_with_exception_handling
    async def close_position(self, position_id: str) ->Dict[str, Any]:
        """
        Close an open position.
        
        Args:
            position_id: ID of the position to close
            
        Returns:
            Dictionary with position closure result
            
        Raises:
            ClientError: If the request fails
            ClientConnectionError: If connection to the service fails
            ClientTimeoutError: If the request times out
            ClientValidationError: If the position ID is invalid
            ClientAuthenticationError: If authentication fails
        """
        self.logger.debug(f'Closing position: {position_id}')
        try:
            return await self.post(f'positions/{position_id}/close')
        except Exception as e:
            self.logger.error(
                f'Failed to close position {position_id}: {str(e)}')
            raise

    @async_with_exception_handling
    async def check_health(self) ->Dict[str, Any]:
        """
        Check the health of the Trading Gateway Service.
        
        Returns:
            Dict: Health check result
            
        Raises:
            ClientError: If the request fails
            ClientConnectionError: If connection to the service fails
            ClientTimeoutError: If the request times out
        """
        self.logger.debug('Checking trading gateway service health')
        try:
            health_data = await self.get('health')
            return {'status': 'healthy', 'message': 'Connection successful',
                'details': health_data}
        except Exception as e:
            self.logger.error(
                f'Failed to check trading gateway service health: {str(e)}')
            return {'status': 'unhealthy', 'message':
                f'Health check failed: {str(e)}', 'details': None}
