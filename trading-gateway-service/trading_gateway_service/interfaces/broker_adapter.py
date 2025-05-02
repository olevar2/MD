"""
Interface definitions for broker adapters.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import uuid

class OrderType(Enum):
    """Types of orders that can be placed with a broker."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    OCO = "one_cancels_other"  # One-cancels-other (OCO) order

class OrderDirection(Enum):
    """Direction of an order (buy or sell)."""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Status of an order in its lifecycle."""
    CREATED = "created"        # Order created but not sent
    PENDING = "pending"        # Order sent but not confirmed
    ACCEPTED = "accepted"      # Order accepted by broker
    REJECTED = "rejected"      # Order rejected by broker
    FILLED = "filled"          # Order fully executed
    PARTIALLY_FILLED = "partially_filled"  # Order partially executed
    CANCELLED = "cancelled"    # Order cancelled
    EXPIRED = "expired"        # Order expired (e.g., day orders)
    ERROR = "error"            # Order processing encountered an error

class OrderRequest:
    """
    Represents a request to place an order with a broker.
    """
    
    def __init__(self,
                instrument: str,
                order_type: OrderType,
                direction: OrderDirection,
                quantity: float,
                price: Optional[float] = None,
                stop_loss: Optional[float] = None,
                take_profit: Optional[float] = None,
                time_in_force: Optional[str] = None,
                good_till_date: Optional[datetime] = None,
                client_order_id: Optional[str] = None,
                **extra_params):
        """
        Initialize an order request.
        
        Args:
            instrument: The trading instrument (e.g., 'EURUSD')
            order_type: The type of order
            direction: Whether to buy or sell
            quantity: The quantity/volume to trade
            price: The limit price (for LIMIT orders) or stop price (for STOP orders)
            stop_loss: Optional stop loss level
            take_profit: Optional take profit level
            time_in_force: How long the order should be active (e.g., 'GTC', 'IOC', 'FOK')
            good_till_date: If time_in_force is 'GTD', when the order should expire
            client_order_id: Optional client-generated order ID for tracking
            **extra_params: Additional broker-specific parameters
        """
        self.instrument = instrument
        self.order_type = order_type
        self.direction = direction
        self.quantity = quantity
        self.price = price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.time_in_force = time_in_force or "GTC"  # Good Till Cancelled (default)
        self.good_till_date = good_till_date
        self.client_order_id = client_order_id or str(uuid.uuid4())
        self.extra_params = extra_params
        self.created_at = datetime.utcnow()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the order request to a dictionary representation."""
        return {
            "instrument": self.instrument,
            "order_type": self.order_type.value,
            "direction": self.direction.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "time_in_force": self.time_in_force,
            "good_till_date": self.good_till_date.isoformat() if self.good_till_date else None,
            "client_order_id": self.client_order_id,
            "created_at": self.created_at.isoformat(),
            "extra_params": self.extra_params
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrderRequest':
        """Create an order request from a dictionary."""
        # Extract and convert enum fields
        order_type = OrderType(data.get("order_type", "market"))
        direction = OrderDirection(data.get("direction", "buy"))
        
        # Extract and convert datetime fields
        good_till_date = None
        if data.get("good_till_date"):
            if isinstance(data["good_till_date"], str):
                good_till_date = datetime.fromisoformat(data["good_till_date"])
            else:
                good_till_date = data["good_till_date"]
        
        # Extract extra params
        extra_params = data.get("extra_params", {})
        
        # Create and return the order request
        return cls(
            instrument=data.get("instrument", ""),
            order_type=order_type,
            direction=direction,
            quantity=data.get("quantity", 0),
            price=data.get("price"),
            stop_loss=data.get("stop_loss"),
            take_profit=data.get("take_profit"),
            time_in_force=data.get("time_in_force", "GTC"),
            good_till_date=good_till_date,
            client_order_id=data.get("client_order_id"),
            **extra_params
        )

class ExecutionReport:
    """
    Represents the result of an order execution or modification.
    """
    
    def __init__(self,
                broker_order_id: str,
                client_order_id: str,
                instrument: str,
                status: OrderStatus,
                filled_quantity: float = 0.0,
                average_price: Optional[float] = None,
                rejection_reason: Optional[str] = None,
                transaction_time: Optional[datetime] = None,
                **extra_data):
        """
        Initialize an execution report.
        
        Args:
            broker_order_id: The broker's ID for this order
            client_order_id: The client-generated order ID
            instrument: The traded instrument
            status: Current status of the order
            filled_quantity: Quantity that has been filled
            average_price: Average fill price
            rejection_reason: If rejected, the reason
            transaction_time: Time of the transaction
            **extra_data: Additional broker-specific data
        """
        self.broker_order_id = broker_order_id
        self.client_order_id = client_order_id
        self.instrument = instrument
        self.status = status
        self.filled_quantity = filled_quantity
        self.average_price = average_price
        self.rejection_reason = rejection_reason
        self.transaction_time = transaction_time or datetime.utcnow()
        self.extra_data = extra_data
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the execution report to a dictionary representation."""
        return {
            "broker_order_id": self.broker_order_id,
            "client_order_id": self.client_order_id,
            "instrument": self.instrument,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "average_price": self.average_price,
            "rejection_reason": self.rejection_reason,
            "transaction_time": self.transaction_time.isoformat(),
            "extra_data": self.extra_data
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionReport':
        """Create an execution report from a dictionary."""
        # Extract and convert enum fields
        status = OrderStatus(data.get("status", "pending"))
        
        # Extract and convert datetime fields
        transaction_time = None
        if data.get("transaction_time"):
            if isinstance(data["transaction_time"], str):
                transaction_time = datetime.fromisoformat(data["transaction_time"])
            else:
                transaction_time = data["transaction_time"]
        
        # Extract extra data
        extra_data = data.get("extra_data", {})
        
        # Create and return the execution report
        return cls(
            broker_order_id=data.get("broker_order_id", ""),
            client_order_id=data.get("client_order_id", ""),
            instrument=data.get("instrument", ""),
            status=status,
            filled_quantity=data.get("filled_quantity", 0.0),
            average_price=data.get("average_price"),
            rejection_reason=data.get("rejection_reason"),
            transaction_time=transaction_time,
            **extra_data
        )

class PositionUpdate:
    """
    Represents an update to a position from the broker.
    """
    
    def __init__(self,
                instrument: str,
                position_id: str,
                quantity: float,
                average_price: float,
                unrealized_pl: float,
                realized_pl: float = 0.0,
                margin_used: Optional[float] = None,
                update_time: Optional[datetime] = None,
                **extra_data):
        """
        Initialize a position update.
        
        Args:
            instrument: The instrument being traded
            position_id: Unique identifier for the position
            quantity: Current position quantity (positive for long, negative for short)
            average_price: Average entry price
            unrealized_pl: Current unrealized profit/loss
            realized_pl: Realized profit/loss from partial closes
            margin_used: Margin allocated to this position
            update_time: Time of the update
            **extra_data: Additional broker-specific data
        """
        self.instrument = instrument
        self.position_id = position_id
        self.quantity = quantity
        self.average_price = average_price
        self.unrealized_pl = unrealized_pl
        self.realized_pl = realized_pl
        self.margin_used = margin_used
        self.update_time = update_time or datetime.utcnow()
        self.extra_data = extra_data
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the position update to a dictionary representation."""
        return {
            "instrument": self.instrument,
            "position_id": self.position_id,
            "quantity": self.quantity,
            "average_price": self.average_price,
            "unrealized_pl": self.unrealized_pl,
            "realized_pl": self.realized_pl,
            "margin_used": self.margin_used,
            "update_time": self.update_time.isoformat(),
            "extra_data": self.extra_data
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PositionUpdate':
        """Create a position update from a dictionary."""
        # Extract and convert datetime fields
        update_time = None
        if data.get("update_time"):
            if isinstance(data["update_time"], str):
                update_time = datetime.fromisoformat(data["update_time"])
            else:
                update_time = data["update_time"]
        
        # Extract extra data
        extra_data = data.get("extra_data", {})
        
        # Create and return the position update
        return cls(
            instrument=data.get("instrument", ""),
            position_id=data.get("position_id", ""),
            quantity=data.get("quantity", 0.0),
            average_price=data.get("average_price", 0.0),
            unrealized_pl=data.get("unrealized_pl", 0.0),
            realized_pl=data.get("realized_pl", 0.0),
            margin_used=data.get("margin_used"),
            update_time=update_time,
            **extra_data
        )

class AccountUpdate:
    """
    Represents an update to an account from the broker.
    """
    
    def __init__(self,
                account_id: str,
                balance: float,
                equity: float,
                margin_used: float,
                margin_available: float,
                currency: str,
                update_time: Optional[datetime] = None,
                **extra_data):
        """
        Initialize an account update.
        
        Args:
            account_id: The account identifier
            balance: Current account balance
            equity: Balance plus unrealized P&L
            margin_used: Amount of margin currently in use
            margin_available: Margin available for new positions
            currency: Currency of the account
            update_time: Time of the update
            **extra_data: Additional broker-specific data
        """
        self.account_id = account_id
        self.balance = balance
        self.equity = equity
        self.margin_used = margin_used
        self.margin_available = margin_available
        self.currency = currency
        self.update_time = update_time or datetime.utcnow()
        self.extra_data = extra_data
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the account update to a dictionary representation."""
        return {
            "account_id": self.account_id,
            "balance": self.balance,
            "equity": self.equity,
            "margin_used": self.margin_used,
            "margin_available": self.margin_available,
            "currency": self.currency,
            "update_time": self.update_time.isoformat(),
            "extra_data": self.extra_data
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AccountUpdate':
        """Create an account update from a dictionary."""
        # Extract and convert datetime fields
        update_time = None
        if data.get("update_time"):
            if isinstance(data["update_time"], str):
                update_time = datetime.fromisoformat(data["update_time"])
            else:
                update_time = data["update_time"]
        
        # Extract extra data
        extra_data = data.get("extra_data", {})
        
        # Create and return the account update
        return cls(
            account_id=data.get("account_id", ""),
            balance=data.get("balance", 0.0),
            equity=data.get("equity", 0.0),
            margin_used=data.get("margin_used", 0.0),
            margin_available=data.get("margin_available", 0.0),
            currency=data.get("currency", "USD"),
            update_time=update_time,
            **extra_data
        )

class BrokerAdapter(ABC):
    """
    Abstract base class for broker adapters, providing a common interface
    for interacting with different forex brokers.
    """
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the broker.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
        
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the broker.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        pass
        
    @abstractmethod
    async def place_order(self, order_request: OrderRequest) -> ExecutionReport:
        """
        Place a new order with the broker.
        
        Args:
            order_request: The order to place
            
        Returns:
            Execution report for the order
        """
        pass
        
    @abstractmethod
    async def cancel_order(self, client_order_id: str) -> ExecutionReport:
        """
        Cancel an existing order.
        
        Args:
            client_order_id: The client order ID to cancel
            
        Returns:
            Execution report indicating the cancellation status
        """
        pass
        
    @abstractmethod
    async def modify_order(self, 
                          client_order_id: str, 
                          modifications: Dict[str, Any]) -> ExecutionReport:
        """
        Modify an existing order.
        
        Args:
            client_order_id: The client order ID to modify
            modifications: Dictionary of fields to modify
            
        Returns:
            Execution report indicating the modification status
        """
        pass
        
    @abstractmethod
    async def get_order_status(self, client_order_id: str) -> ExecutionReport:
        """
        Get the current status of an order.
        
        Args:
            client_order_id: The client order ID to check
            
        Returns:
            Execution report with the current status
        """
        pass
        
    @abstractmethod
    async def get_positions(self) -> List[PositionUpdate]:
        """
        Get all current positions.
        
        Returns:
            List of position updates
        """
        pass
        
    @abstractmethod
    async def close_position(self, 
                           position_id: str, 
                           quantity: Optional[float] = None) -> ExecutionReport:
        """
        Close an existing position.
        
        Args:
            position_id: The position ID to close
            quantity: Optional quantity to close (if None, close entire position)
            
        Returns:
            Execution report for the closing order
        """
        pass
        
    @abstractmethod
    async def get_account_info(self) -> AccountUpdate:
        """
        Get current account information.
        
        Returns:
            Account update with current information
        """
        pass
        
    @abstractmethod
    async def subscribe_to_updates(self, 
                                callback_execution: callable, 
                                callback_position: callable,
                                callback_account: callable) -> bool:
        """
        Subscribe to real-time updates from the broker.
        
        Args:
            callback_execution: Callback function for execution updates
            callback_position: Callback function for position updates
            callback_account: Callback function for account updates
            
        Returns:
            True if subscription successful, False otherwise
        """
        pass
        
    @abstractmethod
    async def unsubscribe_from_updates(self) -> bool:
        """
        Unsubscribe from real-time updates.
        
        Returns:
            True if unsubscription successful, False otherwise
        """
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if the adapter is currently connected to the broker.
        
        Returns:
            True if connected, False otherwise
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the broker.
        
        Returns:
            Broker name
        """
        pass
