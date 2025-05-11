"""
Trading Provider Adapter Module

This module provides adapter implementations for trading provider interfaces,
helping to break circular dependencies between services.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from common_lib.interfaces.trading import ITradingProvider, OrderType, OrderSide, OrderStatus
from trading_gateway_service.core.logging import get_logger


class TradingProviderAdapter(ITradingProvider):
    """
    Adapter implementation for the Trading Provider interface.
    
    This adapter implements the Trading Provider interface using the
    trading gateway service's internal components.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the Trading Provider adapter.
        
        Args:
            logger: Logger to use (if None, creates a new logger)
        """
        self.logger = logger or get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Import trading gateway components
        from trading_gateway_service.services.order_service import OrderService
        self.order_service = OrderService()
    
    async def place_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: Optional[str] = "GTC",
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Place a trading order.
        
        Args:
            symbol: Trading symbol (e.g., "EUR/USD")
            order_type: Type of order (e.g., MARKET, LIMIT)
            side: Order side (e.g., BUY, SELL)
            quantity: Order quantity
            price: Order price (required for LIMIT orders)
            stop_price: Stop price (required for STOP orders)
            time_in_force: Time in force (e.g., "GTC", "IOC", "FOK")
            client_order_id: Client-generated order ID
            
        Returns:
            Dictionary containing the order details
        """
        try:
            # Use the order service to place the order
            order_result = await self.order_service.place_order(
                symbol=symbol,
                order_type=order_type,
                side=side,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                time_in_force=time_in_force,
                client_order_id=client_order_id
            )
            
            return order_result
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            # Return error response
            return {
                "status": "error",
                "message": str(e),
                "order_id": None
            }
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        try:
            # Use the order service to cancel the order
            success = await self.order_service.cancel_order(order_id)
            
            return success
        except Exception as e:
            self.logger.error(f"Error cancelling order: {str(e)}")
            # Return False on error
            return False
    
    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get order details.
        
        Args:
            order_id: ID of the order to get
            
        Returns:
            Dictionary containing the order details
        """
        try:
            # Use the order service to get the order
            order = await self.order_service.get_order(order_id)
            
            return order
        except Exception as e:
            self.logger.error(f"Error getting order: {str(e)}")
            # Return empty dictionary on error
            return {}
    
    async def get_orders(
        self,
        symbol: Optional[str] = None,
        status: Optional[OrderStatus] = None
    ) -> List[Dict[str, Any]]:
        """
        Get orders.
        
        Args:
            symbol: Trading symbol to filter by
            status: Order status to filter by
            
        Returns:
            List of dictionaries containing order details
        """
        try:
            # Use the order service to get orders
            orders = await self.order_service.get_orders(
                symbol=symbol,
                status=status
            )
            
            return orders
        except Exception as e:
            self.logger.error(f"Error getting orders: {str(e)}")
            # Return empty list on error
            return []
