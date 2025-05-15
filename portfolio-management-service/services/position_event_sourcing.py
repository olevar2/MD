"""
Position Event Sourcing Service

This service implements event sourcing for position management.
It publishes position events and maintains a position event store.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Set, Callable, Awaitable

from common_lib.events.base import Event, EventType, EventPriority, EventMetadata
from common_lib.events.event_publisher import EventPublisher
from common_lib.events.event_bus_factory import EventBusFactory, EventBusType
from common_lib.exceptions import ServiceError

logger = logging.getLogger(__name__)


class PositionEventStore:
    """
    Event store for position events.
    
    This class stores position events and provides methods to replay events
    to reconstruct position state.
    """
    
    def __init__(self):
        """
        Initialize the position event store.
        """
        # Position events by position ID
        self.position_events: Dict[str, List[Dict[str, Any]]] = {}
        
        # Position state by position ID
        self.position_state: Dict[str, Dict[str, Any]] = {}
    
    def add_event(self, position_id: str, event: Dict[str, Any]) -> None:
        """
        Add an event to the event store.
        
        Args:
            position_id: Position ID
            event: Position event
        """
        if position_id not in self.position_events:
            self.position_events[position_id] = []
        
        self.position_events[position_id].append(event)
        
        # Update position state
        self._update_position_state(position_id, event)
    
    def get_events(self, position_id: str) -> List[Dict[str, Any]]:
        """
        Get all events for a position.
        
        Args:
            position_id: Position ID
            
        Returns:
            List of position events
        """
        return self.position_events.get(position_id, [])
    
    def get_position_state(self, position_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current state of a position.
        
        Args:
            position_id: Position ID
            
        Returns:
            Position state or None if not found
        """
        return self.position_state.get(position_id)
    
    def get_all_position_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the current state of all positions.
        
        Returns:
            Dictionary of position states by position ID
        """
        return self.position_state
    
    def _update_position_state(self, position_id: str, event: Dict[str, Any]) -> None:
        """
        Update position state based on an event.
        
        Args:
            position_id: Position ID
            event: Position event
        """
        event_type = event.get("event_type")
        payload = event.get("payload", {})
        
        if event_type == "position.opened":
            # Create new position state
            self.position_state[position_id] = {
                "id": position_id,
                "symbol": payload.get("symbol"),
                "direction": payload.get("direction"),
                "quantity": payload.get("quantity"),
                "entry_price": payload.get("entry_price"),
                "stop_loss": payload.get("stop_loss"),
                "take_profit": payload.get("take_profit"),
                "account_id": payload.get("account_id"),
                "strategy_id": payload.get("strategy_id"),
                "status": "OPEN",
                "entry_date": payload.get("timestamp"),
                "unrealized_pl": 0.0,
                "realized_pl": 0.0,
                "current_price": payload.get("entry_price"),
                "metadata": payload.get("metadata", {})
            }
        
        elif event_type == "position.updated":
            # Update existing position state
            if position_id in self.position_state:
                position = self.position_state[position_id]
                
                # Update fields
                if "stop_loss" in payload:
                    position["stop_loss"] = payload["stop_loss"]
                if "take_profit" in payload:
                    position["take_profit"] = payload["take_profit"]
                if "current_price" in payload:
                    position["current_price"] = payload["current_price"]
                if "unrealized_pl" in payload:
                    position["unrealized_pl"] = payload["unrealized_pl"]
                if "metadata" in payload:
                    position["metadata"] = {**position.get("metadata", {}), **payload["metadata"]}
        
        elif event_type == "position.closed":
            # Close position
            if position_id in self.position_state:
                position = self.position_state[position_id]
                
                # Update fields
                position["status"] = "CLOSED"
                position["exit_date"] = payload.get("timestamp")
                position["exit_price"] = payload.get("exit_price")
                position["realized_pl"] = payload.get("realized_pl", 0.0)
                position["unrealized_pl"] = 0.0
                
                if "metadata" in payload:
                    position["metadata"] = {**position.get("metadata", {}), **payload["metadata"]}


class PositionEventSourcing:
    """
    Service for position event sourcing.
    
    This service publishes position events and maintains a position event store.
    """
    
    def __init__(
        self,
        service_name: str = "position-event-sourcing",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the position event sourcing service.
        
        Args:
            service_name: Name of the service
            config: Configuration options
        """
        self.service_name = service_name
        self.config = config or {}
        
        # Initialize event bus
        self.event_bus = EventBusFactory.create_event_bus(
            bus_type=EventBusType.IN_MEMORY,  # Use in-memory for development, Kafka for production
            service_name=service_name
        )
        
        # Initialize event publisher
        self.publisher = EventPublisher(
            event_bus=self.event_bus,
            source_service=service_name
        )
        
        # Set a correlation ID for all events published by this service
        self.publisher.set_correlation_id(f"position-events-{int(time.time())}")
        
        # Initialize event store
        self.event_store = PositionEventStore()
        
        # Running flag
        self.running = False
    
    async def start(self) -> None:
        """
        Start the position event sourcing service.
        """
        if self.running:
            logger.warning("Position event sourcing service is already running")
            return
        
        # Start the event bus
        await self.event_bus.start()
        
        # Set running flag
        self.running = True
        
        logger.info("Position event sourcing service started")
    
    async def stop(self) -> None:
        """
        Stop the position event sourcing service.
        """
        if not self.running:
            logger.warning("Position event sourcing service is not running")
            return
        
        # Set running flag
        self.running = False
        
        # Stop the event bus
        await self.event_bus.stop()
        
        logger.info("Position event sourcing service stopped")
    
    async def open_position(
        self,
        symbol: str,
        direction: str,
        quantity: float,
        entry_price: float,
        account_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        strategy_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Open a new position.
        
        Args:
            symbol: Trading symbol
            direction: Position direction (BUY/SELL)
            quantity: Position quantity
            entry_price: Entry price
            account_id: Account ID
            stop_loss: Stop loss price
            take_profit: Take profit price
            strategy_id: Strategy ID
            metadata: Additional metadata
            
        Returns:
            Position ID
        """
        # Generate position ID
        position_id = str(uuid.uuid4())
        
        # Create payload
        payload = {
            "position_id": position_id,
            "symbol": symbol,
            "direction": direction,
            "quantity": quantity,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "account_id": account_id,
            "strategy_id": strategy_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}
        }
        
        # Publish event
        await self.publisher.publish(
            event_type=EventType.POSITION_OPENED,
            payload=payload,
            priority=EventPriority.HIGH
        )
        
        # Store event
        self.event_store.add_event(position_id, {
            "event_type": "position.opened",
            "payload": payload,
            "timestamp": payload["timestamp"]
        })
        
        logger.info(f"Opened {direction} position for {symbol} with quantity {quantity} at {entry_price}")
        
        return position_id
    
    async def update_position(
        self,
        position_id: str,
        current_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update a position.
        
        Args:
            position_id: Position ID
            current_price: Current price
            stop_loss: Stop loss price
            take_profit: Take profit price
            metadata: Additional metadata
        """
        # Get position state
        position = self.event_store.get_position_state(position_id)
        if not position:
            raise ServiceError(f"Position {position_id} not found")
        
        # Create payload
        payload = {
            "position_id": position_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}
        }
        
        # Add optional fields
        if current_price is not None:
            payload["current_price"] = current_price
            
            # Calculate unrealized P&L
            if position["direction"] == "BUY":
                payload["unrealized_pl"] = (current_price - position["entry_price"]) * position["quantity"]
            else:  # SELL
                payload["unrealized_pl"] = (position["entry_price"] - current_price) * position["quantity"]
        
        if stop_loss is not None:
            payload["stop_loss"] = stop_loss
        
        if take_profit is not None:
            payload["take_profit"] = take_profit
        
        # Publish event
        await self.publisher.publish(
            event_type=EventType.POSITION_UPDATED,
            payload=payload,
            priority=EventPriority.MEDIUM
        )
        
        # Store event
        self.event_store.add_event(position_id, {
            "event_type": "position.updated",
            "payload": payload,
            "timestamp": payload["timestamp"]
        })
        
        logger.info(f"Updated position {position_id}")
    
    async def close_position(
        self,
        position_id: str,
        exit_price: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Close a position.
        
        Args:
            position_id: Position ID
            exit_price: Exit price
            metadata: Additional metadata
        """
        # Get position state
        position = self.event_store.get_position_state(position_id)
        if not position:
            raise ServiceError(f"Position {position_id} not found")
        
        if position["status"] != "OPEN":
            raise ServiceError(f"Position {position_id} is already closed")
        
        # Calculate realized P&L
        if position["direction"] == "BUY":
            realized_pl = (exit_price - position["entry_price"]) * position["quantity"]
        else:  # SELL
            realized_pl = (position["entry_price"] - exit_price) * position["quantity"]
        
        # Create payload
        payload = {
            "position_id": position_id,
            "exit_price": exit_price,
            "realized_pl": realized_pl,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}
        }
        
        # Publish event
        await self.publisher.publish(
            event_type=EventType.POSITION_CLOSED,
            payload=payload,
            priority=EventPriority.HIGH
        )
        
        # Store event
        self.event_store.add_event(position_id, {
            "event_type": "position.closed",
            "payload": payload,
            "timestamp": payload["timestamp"]
        })
        
        logger.info(f"Closed position {position_id} with realized P&L {realized_pl}")
    
    def get_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a position by ID.
        
        Args:
            position_id: Position ID
            
        Returns:
            Position state or None if not found
        """
        return self.event_store.get_position_state(position_id)
    
    def get_position_events(self, position_id: str) -> List[Dict[str, Any]]:
        """
        Get all events for a position.
        
        Args:
            position_id: Position ID
            
        Returns:
            List of position events
        """
        return self.event_store.get_events(position_id)
    
    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all positions.
        
        Returns:
            Dictionary of position states by position ID
        """
        return self.event_store.get_all_position_states()
    
    def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all open positions.
        
        Returns:
            Dictionary of open position states by position ID
        """
        return {
            position_id: position
            for position_id, position in self.event_store.get_all_position_states().items()
            if position["status"] == "OPEN"
        }
    
    def get_closed_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all closed positions.
        
        Returns:
            Dictionary of closed position states by position ID
        """
        return {
            position_id: position
            for position_id, position in self.event_store.get_all_position_states().items()
            if position["status"] == "CLOSED"
        }


# Singleton instance
_position_event_sourcing = None


def get_position_event_sourcing(
    service_name: str = "position-event-sourcing",
    config: Optional[Dict[str, Any]] = None
) -> PositionEventSourcing:
    """
    Get the singleton position event sourcing instance.
    
    Args:
        service_name: Name of the service
        config: Configuration options
        
    Returns:
        Position event sourcing instance
    """
    global _position_event_sourcing
    
    if _position_event_sourcing is None:
        _position_event_sourcing = PositionEventSourcing(
            service_name=service_name,
            config=config
        )
    
    return _position_event_sourcing
