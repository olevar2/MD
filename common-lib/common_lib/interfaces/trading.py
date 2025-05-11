
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"

class ITradingProvider(ABC):
    """Interface for trading providers"""

    @abstractmethod
    async def place_order(self,
                         symbol: str,
                         order_type: OrderType,
                         side: OrderSide,
                         quantity: float,
                         price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         time_in_force: Optional[str] = "GTC",
                         client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """Place a trading order"""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass

    @abstractmethod
    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get order details"""
        pass

    @abstractmethod
    async def get_orders(self,
                        symbol: Optional[str] = None,
                        status: Optional[OrderStatus] = None) -> List[Dict[str, Any]]:
        """Get orders"""
        pass

    @abstractmethod
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get current positions"""
        pass

class IOrderBookProvider(ABC):
    """Interface for order book providers"""

    @abstractmethod
    async def get_order_book(self, symbol: str, depth: Optional[int] = 10) -> Dict[str, Any]:
        """Get order book for a symbol"""
        pass

    @abstractmethod
    async def subscribe_order_book(self, symbol: str, callback: callable) -> Any:
        """Subscribe to order book updates"""
        pass

    @abstractmethod
    async def unsubscribe_order_book(self, subscription_id: Any) -> bool:
        """Unsubscribe from order book updates"""
        pass

# Risk Manager interface moved to risk_management.py
