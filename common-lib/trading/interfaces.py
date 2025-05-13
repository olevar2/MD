"""
Shared interfaces for trading and risk management.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

class TradingStatus(Enum):
    """
    TradingStatus class that inherits from Enum.
    
    Attributes:
        Add attributes here
    """

    PENDING = "pending"
    EXECUTED = "executed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"

@dataclass
class OrderRequest:
    """
    OrderRequest class.
    
    Attributes:
        Add attributes here
    """

    symbol: str
    side: str
    quantity: float
    order_type: str
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class ITradingGateway(ABC):
    """
    ITradingGateway class that inherits from ABC.
    
    Attributes:
        Add attributes here
    """

    @abstractmethod
    async def submit_order(self, order: OrderRequest) -> Dict[str, Any]:
    """
    Submit order.
    
    Args:
        order: Description of order
    
    Returns:
        Dict[str, Any]: Description of return value
    
    """

        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
    """
    Cancel order.
    
    Args:
        order_id: Description of order_id
    
    Returns:
        bool: Description of return value
    
    """

        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> TradingStatus:
    """
    Get order status.
    
    Args:
        order_id: Description of order_id
    
    Returns:
        TradingStatus: Description of return value
    
    """

        pass

class IRiskManager(ABC):
    """
    IRiskManager class that inherits from ABC.
    
    Attributes:
        Add attributes here
    """

    @abstractmethod
    async def validate_order(self, order: OrderRequest) -> Dict[str, Any]:
    """
    Validate order.
    
    Args:
        order: Description of order
    
    Returns:
        Dict[str, Any]: Description of return value
    
    """

        pass

    @abstractmethod
    async def get_position_risk(self, symbol: str) -> Dict[str, Any]:
    """
    Get position risk.
    
    Args:
        symbol: Description of symbol
    
    Returns:
        Dict[str, Any]: Description of return value
    
    """

        pass

    @abstractmethod
    async def get_portfolio_risk(self) -> Dict[str, Any]:
    """
    Get portfolio risk.
    
    Returns:
        Dict[str, Any]: Description of return value
    
    """

        pass
