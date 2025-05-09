"""
Shared interfaces for trading and risk management.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

class TradingStatus(Enum):
    PENDING = "pending"
    EXECUTED = "executed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"

@dataclass
class OrderRequest:
    symbol: str
    side: str
    quantity: float
    order_type: str
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class ITradingGateway(ABC):
    @abstractmethod
    async def submit_order(self, order: OrderRequest) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> TradingStatus:
        pass

class IRiskManager(ABC):
    @abstractmethod
    async def validate_order(self, order: OrderRequest) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def get_position_risk(self, symbol: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def get_portfolio_risk(self) -> Dict[str, Any]:
        pass
