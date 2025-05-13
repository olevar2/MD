"""
Trading Service Adapter.

This module provides adapter implementations for the Trading Service interfaces.
These adapters allow other services to interact with the Trading Service
without direct dependencies, breaking circular dependencies.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable

import pandas as pd

from common_lib.interfaces.trading import ITradingProvider, IOrderBookProvider, IRiskManager
from common_lib.service_client.base_client import ServiceClientConfig
from common_lib.service_client.http_client import AsyncHTTPServiceClient


class TradingProviderAdapter(ITradingProvider):
    """
    Adapter implementation for the Trading Provider interface.

    This adapter uses the HTTP service client to communicate with the Trading Service.
    """

    def __init__(
        self,
        config: ServiceClientConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Trading Provider adapter.

        Args:
            config: Configuration for the service client
            logger: Logger to use (if None, creates a new logger)
        """
        self.client = AsyncHTTPServiceClient(config)
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def place_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        time_in_force: str = "GTC",
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Place a trading order.

        Args:
            symbol: Trading symbol (e.g., "EUR/USD")
            order_type: Type of order (e.g., "MARKET", "LIMIT", "STOP")
            side: Order side (e.g., "BUY", "SELL")
            quantity: Order quantity
            price: Order price (required for LIMIT and STOP orders)
            stop_loss: Stop loss price
            take_profit: Take profit price
            time_in_force: Time in force (e.g., "GTC", "IOC", "FOK")
            parameters: Optional additional parameters

        Returns:
            Dictionary containing the order details
        """
        try:
            # Prepare request body
            request_body = {
                "symbol": symbol,
                "order_type": order_type,
                "side": side,
                "quantity": quantity,
                "time_in_force": time_in_force
            }

            # Add optional parameters
            if price is not None:
                request_body["price"] = price
            if stop_loss is not None:
                request_body["stop_loss"] = stop_loss
            if take_profit is not None:
                request_body["take_profit"] = take_profit
            if parameters:
                request_body.update(parameters)

            # Send request to the Trading Service
            response = await self.client.send_request({
                "method": "POST",
                "path": "api/v1/orders",
                "json": request_body
            })

            # Return order details
            return response.get("data", {})
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            raise

    async def cancel_order(
        self,
        order_id: str
    ) -> Dict[str, Any]:
        """
        Cancel a trading order.

        Args:
            order_id: ID of the order to cancel

        Returns:
            Dictionary containing the cancellation details
        """
        try:
            # Send request to the Trading Service
            response = await self.client.send_request({
                "method": "DELETE",
                "path": f"api/v1/orders/{order_id}"
            })

            # Return cancellation details
            return response.get("data", {})
        except Exception as e:
            self.logger.error(f"Error cancelling order: {str(e)}")
            raise

    async def modify_order(
        self,
        order_id: str,
        price: Optional[float] = None,
        quantity: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Modify a trading order.

        Args:
            order_id: ID of the order to modify
            price: New order price
            quantity: New order quantity
            stop_loss: New stop loss price
            take_profit: New take profit price
            parameters: Optional additional parameters

        Returns:
            Dictionary containing the modified order details
        """
        try:
            # Prepare request body
            request_body = {}

            # Add optional parameters
            if price is not None:
                request_body["price"] = price
            if quantity is not None:
                request_body["quantity"] = quantity
            if stop_loss is not None:
                request_body["stop_loss"] = stop_loss
            if take_profit is not None:
                request_body["take_profit"] = take_profit
            if parameters:
                request_body.update(parameters)

            # Send request to the Trading Service
            response = await self.client.send_request({
                "method": "PATCH",
                "path": f"api/v1/orders/{order_id}",
                "json": request_body
            })

            # Return modified order details
            return response.get("data", {})
        except Exception as e:
            self.logger.error(f"Error modifying order: {str(e)}")
            raise

    async def get_order(
        self,
        order_id: str
    ) -> Dict[str, Any]:
        """
        Get details of a trading order.

        Args:
            order_id: ID of the order

        Returns:
            Dictionary containing the order details
        """
        try:
            # Send request to the Trading Service
            response = await self.client.send_request({
                "method": "GET",
                "path": f"api/v1/orders/{order_id}"
            })

            # Return order details
            return response.get("data", {})
        except Exception as e:
            self.logger.error(f"Error retrieving order: {str(e)}")
            raise

    async def get_orders(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get a list of trading orders.

        Args:
            symbol: Trading symbol to filter by
            status: Order status to filter by
            start_time: Start time for filtering
            end_time: End time for filtering
            limit: Maximum number of orders to return

        Returns:
            List of dictionaries containing order details
        """
        try:
            # Prepare request parameters
            params = {
                "limit": limit
            }
            if symbol:
                params["symbol"] = symbol
            if status:
                params["status"] = status
            if start_time:
                params["start_time"] = start_time.isoformat()
            if end_time:
                params["end_time"] = end_time.isoformat()

            # Send request to the Trading Service
            response = await self.client.send_request({
                "method": "GET",
                "path": "api/v1/orders",
                "params": params
            })

            # Return orders
            return response.get("data", [])
        except Exception as e:
            self.logger.error(f"Error retrieving orders: {str(e)}")
            raise

    async def get_positions(
        self,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get a list of open positions.

        Args:
            symbol: Trading symbol to filter by

        Returns:
            List of dictionaries containing position details
        """
        try:
            # Prepare request parameters
            params = {}
            if symbol:
                params["symbol"] = symbol

            # Send request to the Trading Service
            response = await self.client.send_request({
                "method": "GET",
                "path": "api/v1/positions",
                "params": params
            })

            # Return positions
            return response.get("data", [])
        except Exception as e:
            self.logger.error(f"Error retrieving positions: {str(e)}")
            raise

    async def close_position(
        self,
        position_id: str,
        quantity: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Close a trading position.

        Args:
            position_id: ID of the position to close
            quantity: Quantity to close (if None, closes the entire position)

        Returns:
            Dictionary containing the closure details
        """
        try:
            # Prepare request body
            request_body = {}
            if quantity is not None:
                request_body["quantity"] = quantity

            # Send request to the Trading Service
            response = await self.client.send_request({
                "method": "POST",
                "path": f"api/v1/positions/{position_id}/close",
                "json": request_body
            })

            # Return closure details
            return response.get("data", {})
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            raise

    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.

        Returns:
            Dictionary containing account information
        """
        try:
            # Send request to the Trading Service
            response = await self.client.send_request({
                "method": "GET",
                "path": "api/v1/account"
            })

            # Return account information
            return response.get("data", {})
        except Exception as e:
            self.logger.error(f"Error retrieving account information: {str(e)}")
            raise

    async def get_trade_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get trade history.

        Args:
            symbol: Trading symbol to filter by
            start_time: Start time for filtering
            end_time: End time for filtering
            limit: Maximum number of trades to return

        Returns:
            List of dictionaries containing trade details
        """
        try:
            # Prepare request parameters
            params = {
                "limit": limit
            }
            if symbol:
                params["symbol"] = symbol
            if start_time:
                params["start_time"] = start_time.isoformat()
            if end_time:
                params["end_time"] = end_time.isoformat()

            # Send request to the Trading Service
            response = await self.client.send_request({
                "method": "GET",
                "path": "api/v1/trades",
                "params": params
            })

            # Return trades
            return response.get("data", [])
        except Exception as e:
            self.logger.error(f"Error retrieving trade history: {str(e)}")
            raise


class OrderBookProviderAdapter(IOrderBookProvider):
    """
    Adapter implementation for the Order Book Provider interface.

    This adapter uses the HTTP service client to communicate with the Trading Service.
    """

    def __init__(
        self,
        config: ServiceClientConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Order Book Provider adapter.

        Args:
            config: Configuration for the service client
            logger: Logger to use (if None, creates a new logger)
        """
        self.client = AsyncHTTPServiceClient(config)
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._subscriptions = {}

    async def get_order_book(
        self,
        symbol: str,
        depth: int = 10
    ) -> Dict[str, Any]:
        """
        Get the order book for a symbol.

        Args:
            symbol: Trading symbol (e.g., "EUR/USD")
            depth: Depth of the order book

        Returns:
            Dictionary containing the order book data
        """
        try:
            # Prepare request parameters
            params = {
                "depth": depth
            }

            # Send request to the Trading Service
            response = await self.client.send_request({
                "method": "GET",
                "path": f"api/v1/orderbook/{symbol}",
                "params": params
            })

            # Return order book data
            return response.get("data", {})
        except Exception as e:
            self.logger.error(f"Error retrieving order book: {str(e)}")
            raise

    async def subscribe_to_order_book(
        self,
        symbol: str,
        callback: Callable,
        depth: int = 10
    ) -> str:
        """
        Subscribe to order book updates.

        Args:
            symbol: Trading symbol (e.g., "EUR/USD")
            callback: Callback function to be called when the order book is updated
            depth: Depth of the order book

        Returns:
            Subscription ID
        """
        try:
            # Send subscription request to the Trading Service
            response = await self.client.send_request({
                "method": "POST",
                "path": "api/v1/orderbook/subscribe",
                "json": {
                    "symbol": symbol,
                    "depth": depth
                }
            })

            # Extract subscription ID
            subscription_id = response.get("subscription_id", "")

            # Store callback
            if subscription_id:
                self._subscriptions[subscription_id] = callback

            return subscription_id
        except Exception as e:
            self.logger.error(f"Error subscribing to order book: {str(e)}")
            raise

    async def unsubscribe_from_order_book(
        self,
        subscription_id: str
    ) -> bool:
        """
        Unsubscribe from order book updates.

        Args:
            subscription_id: Subscription ID returned by subscribe_to_order_book

        Returns:
            True if unsubscribed successfully, False otherwise
        """
        try:
            # Send unsubscription request to the Trading Service
            response = await self.client.send_request({
                "method": "POST",
                "path": "api/v1/orderbook/unsubscribe",
                "json": {
                    "subscription_id": subscription_id
                }
            })

            # Remove callback
            if subscription_id in self._subscriptions:
                del self._subscriptions[subscription_id]

            # Return success status
            return response.get("success", False)
        except Exception as e:
            self.logger.error(f"Error unsubscribing from order book: {str(e)}")
            return False


class RiskManagerAdapter(IRiskManager):
    """
    Adapter implementation for the Risk Manager interface.

    This adapter uses the HTTP service client to communicate with the Trading Service.
    """

    def __init__(
        self,
        config: ServiceClientConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Risk Manager adapter.

        Args:
            config: Configuration for the service client
            logger: Logger to use (if None, creates a new logger)
        """
        self.client = AsyncHTTPServiceClient(config)
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def validate_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        account_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate a trading order against risk rules.

        Args:
            symbol: Trading symbol (e.g., "EUR/USD")
            order_type: Type of order (e.g., "MARKET", "LIMIT", "STOP")
            side: Order side (e.g., "BUY", "SELL")
            quantity: Order quantity
            price: Order price
            account_info: Account information

        Returns:
            Dictionary containing validation results
        """
        try:
            # Prepare request body
            request_body = {
                "symbol": symbol,
                "order_type": order_type,
                "side": side,
                "quantity": quantity
            }

            # Add optional parameters
            if price is not None:
                request_body["price"] = price
            if account_info:
                request_body["account_info"] = account_info

            # Send request to the Trading Service
            response = await self.client.send_request({
                "method": "POST",
                "path": "api/v1/risk/validate-order",
                "json": request_body
            })

            # Return validation results
            return response.get("data", {})
        except Exception as e:
            self.logger.error(f"Error validating order: {str(e)}")
            raise

    async def calculate_position_risk(
        self,
        position: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate risk metrics for a position.

        Args:
            position: Position details

        Returns:
            Dictionary containing risk metrics
        """
        try:
            # Send request to the Trading Service
            response = await self.client.send_request({
                "method": "POST",
                "path": "api/v1/risk/position",
                "json": position
            })

            # Return risk metrics
            return response.get("data", {})
        except Exception as e:
            self.logger.error(f"Error calculating position risk: {str(e)}")
            raise

    async def calculate_portfolio_risk(
        self,
        positions: List[Dict[str, Any]],
        account_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate risk metrics for the entire portfolio.

        Args:
            positions: List of position details
            account_info: Account information

        Returns:
            Dictionary containing risk metrics
        """
        try:
            # Send request to the Trading Service
            response = await self.client.send_request({
                "method": "POST",
                "path": "api/v1/risk/portfolio",
                "json": {
                    "positions": positions,
                    "account_info": account_info
                }
            })

            # Return risk metrics
            return response.get("data", {})
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk: {str(e)}")
            raise

    async def get_risk_limits(
        self,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get risk limits.

        Args:
            symbol: Trading symbol to get limits for, or None for global limits

        Returns:
            Dictionary containing risk limits
        """
        try:
            # Prepare request parameters
            params = {}
            if symbol:
                params["symbol"] = symbol

            # Send request to the Trading Service
            response = await self.client.send_request({
                "method": "GET",
                "path": "api/v1/risk/limits",
                "params": params
            })

            # Return risk limits
            return response.get("data", {})
        except Exception as e:
            self.logger.error(f"Error retrieving risk limits: {str(e)}")
            raise