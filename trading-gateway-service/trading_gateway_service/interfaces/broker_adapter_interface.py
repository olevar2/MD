"""
Core interfaces for broker adapters in the trading gateway service.

This module defines the standard interfaces that all broker adapters must implement,
ensuring consistent behavior across different brokers.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timezone
import uuid


from trading_gateway_service.resilience.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class OrderType(Enum):
    """Types of orders that can be placed with brokers."""
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    STOP_LIMIT = 'stop_limit'
    OCO = 'one_cancels_other'
    TRAILING_STOP = 'trailing_stop'


class OrderDirection(Enum):
    """Direction of an order (buy/sell)."""
    BUY = 'buy'
    SELL = 'sell'


class OrderStatus(Enum):
    """Possible statuses for an order."""
    PENDING = 'pending'
    OPEN = 'open'
    PARTIALLY_FILLED = 'partially_filled'
    FILLED = 'filled'
    CANCELLED = 'cancelled'
    REJECTED = 'rejected'
    EXPIRED = 'expired'


@dataclass
class OrderRequest:
    """
    Represents a request to place an order with a broker.
    """
    instrument: str
    order_type: OrderType
    direction: OrderDirection
    quantity: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_pips: Optional[float] = None
    expiry_time: Optional[datetime] = None
    client_order_id: str = None

    def __post_init__(self):
        """Initialize default values if not provided."""
        if self.client_order_id is None:
            self.client_order_id = str(uuid.uuid4())


@dataclass
class ExecutionReport:
    """
    Represents a report on the execution of an order.
    """
    order_id: str
    client_order_id: str
    instrument: str
    status: OrderStatus
    direction: OrderDirection
    order_type: OrderType
    quantity: float
    filled_quantity: float
    price: Optional[float]
    executed_price: Optional[float] = None
    rejection_reason: Optional[str] = None
    timestamp: datetime = None
    commission: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def __post_init__(self):
        """Initialize default values if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class BrokerInfo:
    """
    Information about a broker's capabilities and limitations.
    """
    name: str
    supported_instruments: List[str]
    supports_oco: bool
    supports_trailing_stop: bool
    minimum_lot_size: Dict[str, float]
    maximum_lot_size: Dict[str, float]
    pip_value: Dict[str, float]
    max_leverage: float
    commission_structure: Dict[str, Any]


class BrokerAdapterInterface(ABC):
    """
    Interface that all broker adapters must implement.
    """

    @abstractmethod
    def connect(self, credentials: Dict[str, str]) ->bool:
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
    def get_broker_info(self) ->BrokerInfo:
        """
        Get information about the broker's capabilities and limitations.

        Returns:
            BrokerInfo object
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
    def get_market_data(self, instrument: str) ->Dict[str, Any]:
        """
        Get current market data for an instrument.

        Args:
            instrument: The instrument to get data for (e.g., "EURUSD")

        Returns:
            Dictionary containing market data (bid, ask, spread, etc.)
        """
        pass
