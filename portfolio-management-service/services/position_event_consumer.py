"""
Position Event Consumer Service

This service is responsible for consuming position events from the event bus.
It provides an interface for other services to access position events and state.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable, Awaitable, Set

from common_lib.events.base import Event, EventType, EventPriority, EventMetadata, IEventBus
from common_lib.events.event_bus_factory import EventBusFactory, EventBusType
from common_lib.exceptions import ServiceError

logger = logging.getLogger(__name__)


class PositionEventConsumer:
    """
    Service for consuming position events.
    
    This service subscribes to position events from the event bus and provides
    an interface for other services to access position state.
    """
    
    def __init__(
        self,
        service_name: str = "position-event-consumer",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the position event consumer.
        
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
        
        # Position state by position ID
        self.position_state: Dict[str, Dict[str, Any]] = {}
        
        # Position events by position ID
        self.position_events: Dict[str, List[Dict[str, Any]]] = {}
        
        # Account to positions mapping
        self.account_positions: Dict[str, Set[str]] = {}  # account_id -> set of position_ids
        
        # Symbol to positions mapping
        self.symbol_positions: Dict[str, Set[str]] = {}  # symbol -> set of position_ids
        
        # Callbacks for position events
        self.event_callbacks: Dict[str, List[Callable[[Dict[str, Any]], Awaitable[None]]]] = {
            "position.opened": [],
            "position.updated": [],
            "position.closed": []
        }
        
        # Running flag
        self.running = False
    
    async def start(self) -> None:
        """
        Start the position event consumer.
        """
        if self.running:
            logger.warning("Position event consumer is already running")
            return
        
        # Start the event bus
        await self.event_bus.start()
        
        # Subscribe to position events
        self.event_bus.subscribe(
            event_types=[
                EventType.POSITION_OPENED,
                EventType.POSITION_UPDATED,
                EventType.POSITION_CLOSED
            ],
            handler=self._handle_position_event
        )
        
        # Set running flag
        self.running = True
        
        logger.info("Position event consumer started")
    
    async def stop(self) -> None:
        """
        Stop the position event consumer.
        """
        if not self.running:
            logger.warning("Position event consumer is not running")
            return
        
        # Set running flag
        self.running = False
        
        # Stop the event bus
        await self.event_bus.stop()
        
        logger.info("Position event consumer stopped")
    
    async def _handle_position_event(self, event: Event) -> None:
        """
        Handle position event.
        
        Args:
            event: Position event
        """
        try:
            # Extract event type and payload
            event_type = str(event.event_type)
            payload = event.payload
            
            # Map event type to internal event type
            internal_event_type = None
            if event_type == EventType.POSITION_OPENED:
                internal_event_type = "position.opened"
            elif event_type == EventType.POSITION_UPDATED:
                internal_event_type = "position.updated"
            elif event_type == EventType.POSITION_CLOSED:
                internal_event_type = "position.closed"
            else:
                logger.warning(f"Unknown event type: {event_type}")
                return
            
            # Extract position ID
            position_id = payload.get("position_id")
            if not position_id:
                logger.warning(f"Received {internal_event_type} event without position_id")
                return
            
            # Store event
            if position_id not in self.position_events:
                self.position_events[position_id] = []
            
            self.position_events[position_id].append({
                "event_type": internal_event_type,
                "payload": payload,
                "timestamp": payload.get("timestamp") or datetime.now(timezone.utc).isoformat()
            })
            
            # Update position state
            self._update_position_state(position_id, internal_event_type, payload)
            
            # Call event callbacks
            for callback in self.event_callbacks.get(internal_event_type, []):
                try:
                    await callback(payload)
                except Exception as e:
                    logger.error(f"Error in position event callback: {str(e)}")
            
            logger.info(f"Processed {internal_event_type} event for position {position_id}")
            
        except Exception as e:
            logger.error(f"Error handling position event: {str(e)}")
    
    def _update_position_state(
        self,
        position_id: str,
        event_type: str,
        payload: Dict[str, Any]
    ) -> None:
        """
        Update position state based on an event.
        
        Args:
            position_id: Position ID
            event_type: Event type
            payload: Event payload
        """
        if event_type == "position.opened":
            # Extract account ID and symbol
            account_id = payload.get("account_id")
            symbol = payload.get("symbol")
            
            # Create new position state
            self.position_state[position_id] = {
                "id": position_id,
                "symbol": symbol,
                "direction": payload.get("direction"),
                "quantity": payload.get("quantity"),
                "entry_price": payload.get("entry_price"),
                "stop_loss": payload.get("stop_loss"),
                "take_profit": payload.get("take_profit"),
                "account_id": account_id,
                "strategy_id": payload.get("strategy_id"),
                "status": "OPEN",
                "entry_date": payload.get("timestamp"),
                "unrealized_pl": 0.0,
                "realized_pl": 0.0,
                "current_price": payload.get("entry_price"),
                "metadata": payload.get("metadata", {})
            }
            
            # Update account to positions mapping
            if account_id:
                if account_id not in self.account_positions:
                    self.account_positions[account_id] = set()
                self.account_positions[account_id].add(position_id)
            
            # Update symbol to positions mapping
            if symbol:
                if symbol not in self.symbol_positions:
                    self.symbol_positions[symbol] = set()
                self.symbol_positions[symbol].add(position_id)
        
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
    
    def get_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a position by ID.
        
        Args:
            position_id: Position ID
            
        Returns:
            Position state or None if not found
        """
        return self.position_state.get(position_id)
    
    def get_position_events(self, position_id: str) -> List[Dict[str, Any]]:
        """
        Get all events for a position.
        
        Args:
            position_id: Position ID
            
        Returns:
            List of position events
        """
        return self.position_events.get(position_id, [])
    
    def get_positions_by_account(self, account_id: str) -> List[Dict[str, Any]]:
        """
        Get all positions for an account.
        
        Args:
            account_id: Account ID
            
        Returns:
            List of positions
        """
        position_ids = self.account_positions.get(account_id, set())
        return [self.position_state[position_id] for position_id in position_ids if position_id in self.position_state]
    
    def get_positions_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get all positions for a symbol.
        
        Args:
            symbol: Symbol
            
        Returns:
            List of positions
        """
        position_ids = self.symbol_positions.get(symbol, set())
        return [self.position_state[position_id] for position_id in position_ids if position_id in self.position_state]
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Get all open positions.
        
        Returns:
            List of open positions
        """
        return [position for position in self.position_state.values() if position["status"] == "OPEN"]
    
    def get_closed_positions(self) -> List[Dict[str, Any]]:
        """
        Get all closed positions.
        
        Returns:
            List of closed positions
        """
        return [position for position in self.position_state.values() if position["status"] == "CLOSED"]
    
    def register_event_callback(
        self,
        event_type: str,
        callback: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> None:
        """
        Register a callback for position events.
        
        Args:
            event_type: Event type (position.opened, position.updated, position.closed)
            callback: Callback function that takes an event payload
        """
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)
    
    def unregister_event_callback(
        self,
        event_type: str,
        callback: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> None:
        """
        Unregister a callback for position events.
        
        Args:
            event_type: Event type (position.opened, position.updated, position.closed)
            callback: Callback function to unregister
        """
        if event_type in self.event_callbacks and callback in self.event_callbacks[event_type]:
            self.event_callbacks[event_type].remove(callback)


# Singleton instance
_position_event_consumer = None


def get_position_event_consumer(
    service_name: str = "position-event-consumer",
    config: Optional[Dict[str, Any]] = None
) -> PositionEventConsumer:
    """
    Get the singleton position event consumer instance.
    
    Args:
        service_name: Name of the service
        config: Configuration options
        
    Returns:
        Position event consumer instance
    """
    global _position_event_consumer
    
    if _position_event_consumer is None:
        _position_event_consumer = PositionEventConsumer(
            service_name=service_name,
            config=config
        )
    
    return _position_event_consumer
