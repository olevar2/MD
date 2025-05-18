"""
Standalone test script for event-driven architecture.
"""

import asyncio
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Union, Awaitable
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Event types
class EventType(str, Enum):
    """Standard event types."""
    
    MARKET_DATA_UPDATED = "market.data.updated"
    SIGNAL_GENERATED = "signal.generated"
    ORDER_CREATED = "order.created"
    ORDER_FILLED = "order.filled"
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"

# Event priority
class EventPriority(str, Enum):
    """Priority levels for events."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Event metadata
class EventMetadata:
    """Metadata for events."""
    
    def __init__(
        self,
        source_service: str,
        correlation_id: Optional[str] = None,
        causation_id: Optional[str] = None,
        priority: EventPriority = EventPriority.MEDIUM
    ):
        self.event_id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow().isoformat()
        self.source_service = source_service
        self.correlation_id = correlation_id
        self.causation_id = causation_id
        self.priority = priority

# Event
class Event:
    """Standard event schema."""
    
    def __init__(
        self,
        event_type: Union[EventType, str],
        payload: Dict[str, Any],
        metadata: EventMetadata
    ):
        self.event_type = event_type
        self.payload = payload
        self.metadata = metadata
    
    def get_routing_key(self) -> str:
        """Get the routing key for this event."""
        return str(self.event_type)

# Event handler type
EventHandler = Callable[[Event], Awaitable[None]]
EventFilter = Callable[[Event], bool]

# Event bus interface
class IEventBus(ABC):
    """Interface for event buses."""
    
    @abstractmethod
    async def publish(self, event: Event) -> None:
        """Publish an event to the event bus."""
        pass
    
    @abstractmethod
    def subscribe(
        self,
        event_types: Union[str, EventType, List[Union[str, EventType]]],
        handler: EventHandler,
        filter_func: Optional[EventFilter] = None
    ) -> Callable[[], None]:
        """Subscribe to events of the specified types."""
        pass
    
    @abstractmethod
    def unsubscribe(
        self,
        event_type: Union[str, EventType],
        handler: EventHandler
    ) -> None:
        """Unsubscribe from events of a specific type."""
        pass
    
    @abstractmethod
    def subscribe_to_all(
        self,
        handler: EventHandler,
        filter_func: Optional[EventFilter] = None
    ) -> Callable[[], None]:
        """Subscribe to all events."""
        pass
    
    @abstractmethod
    def unsubscribe_from_all(
        self,
        handler: EventHandler
    ) -> None:
        """Unsubscribe from all events."""
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """Start the event bus."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the event bus."""
        pass

# In-memory event bus
class InMemoryEventBus(IEventBus):
    """In-memory implementation of the event bus."""
    
    def __init__(self):
        self.subscribers: Dict[str, Set[tuple[EventHandler, Optional[EventFilter]]]] = {}
        self.global_subscribers: Set[tuple[EventHandler, Optional[EventFilter]]] = set()
        self._running = False
    
    async def publish(self, event: Event) -> None:
        """Publish an event to the event bus."""
        if not self._running:
            logger.warning("Event bus is not running. Event will not be published.")
            return
            
        event_type = str(event.event_type)
        routing_key = event.get_routing_key()
        
        logger.debug(f"Publishing event: {event_type} (routing key: {routing_key})")
        
        # Get subscribers for this event type
        handlers = set()
        handlers.update(self.subscribers.get(event_type, set()))
        handlers.update(self.subscribers.get(routing_key, set()))
        handlers.update(self.global_subscribers)
        
        # Publish event to subscribers
        tasks = []
        for handler, filter_func in handlers:
            # Apply filter if provided
            if filter_func is None or filter_func(event):
                tasks.append(self._call_handler(handler, event))
        
        # Wait for all handlers to complete
        if tasks:
            await asyncio.gather(*tasks)
    
    async def _call_handler(self, handler: EventHandler, event: Event) -> None:
        """Call an event handler with an event."""
        try:
            await handler(event)
        except Exception as e:
            handler_name = getattr(handler, "__name__", str(handler))
            logger.error(f"Error in event handler {handler_name}: {e}")
    
    def subscribe(
        self,
        event_types: Union[str, EventType, List[Union[str, EventType]]],
        handler: EventHandler,
        filter_func: Optional[EventFilter] = None
    ) -> Callable[[], None]:
        """Subscribe to events of the specified types."""
        # Convert single event type to list
        if not isinstance(event_types, list):
            event_types = [event_types]
        
        # Convert EventType enum values to strings
        event_types = [str(et) for et in event_types]
        
        # Subscribe to each event type
        for event_type in event_types:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = set()
            
            self.subscribers[event_type].add((handler, filter_func))
            handler_name = getattr(handler, "__name__", str(handler))
            logger.debug(f"Subscribed {handler_name} to event type: {event_type}")
        
        # Return unsubscribe function
        def unsubscribe():
            for event_type in event_types:
                self.unsubscribe(event_type, handler)
        
        return unsubscribe
    
    def unsubscribe(
        self,
        event_type: Union[str, EventType],
        handler: EventHandler
    ) -> None:
        """Unsubscribe from events of a specific type."""
        event_type = str(event_type)
        
        if event_type in self.subscribers:
            # Find and remove the handler
            to_remove = None
            for h, f in self.subscribers[event_type]:
                if h == handler:
                    to_remove = (h, f)
                    break
            
            if to_remove:
                self.subscribers[event_type].remove(to_remove)
                handler_name = getattr(handler, "__name__", str(handler))
                logger.debug(f"Unsubscribed {handler_name} from event type: {event_type}")
    
    def subscribe_to_all(
        self,
        handler: EventHandler,
        filter_func: Optional[EventFilter] = None
    ) -> Callable[[], None]:
        """Subscribe to all events."""
        self.global_subscribers.add((handler, filter_func))
        handler_name = getattr(handler, "__name__", str(handler))
        logger.debug(f"Subscribed {handler_name} to all events")
        
        # Return unsubscribe function
        def unsubscribe():
            self.unsubscribe_from_all(handler)
        
        return unsubscribe
    
    def unsubscribe_from_all(
        self,
        handler: EventHandler
    ) -> None:
        """Unsubscribe from all events."""
        # Find and remove the handler
        to_remove = None
        for h, f in self.global_subscribers:
            if h == handler:
                to_remove = (h, f)
                break
        
        if to_remove:
            self.global_subscribers.remove(to_remove)
            handler_name = getattr(handler, "__name__", str(handler))
            logger.debug(f"Unsubscribed {handler_name} from all events")
    
    async def start(self) -> None:
        """Start the event bus."""
        self._running = True
        logger.info("In-memory event bus started")
    
    async def stop(self) -> None:
        """Stop the event bus."""
        self._running = False
        logger.info("In-memory event bus stopped")

# Event publisher
class EventPublisher:
    """Helper class for publishing events."""
    
    def __init__(
        self,
        event_bus: IEventBus,
        source_service: str
    ):
        """Initialize the event publisher."""
        self.event_bus = event_bus
        self.source_service = source_service
        self._correlation_id = None
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """Set the correlation ID for all events published by this publisher."""
        self._correlation_id = correlation_id
    
    async def publish(
        self,
        event_type: Union[EventType, str],
        payload: Dict[str, Any],
        causation_id: Optional[str] = None,
        priority: EventPriority = EventPriority.MEDIUM
    ) -> Event:
        """Publish an event."""
        # Create event metadata
        metadata = EventMetadata(
            source_service=self.source_service,
            correlation_id=self._correlation_id,
            causation_id=causation_id,
            priority=priority
        )
        
        # Create event
        event = Event(
            event_type=event_type,
            payload=payload,
            metadata=metadata
        )
        
        # Publish event
        await self.event_bus.publish(event)
        
        return event

# Example service
class ExampleService:
    """Example service that demonstrates event-driven architecture."""
    
    def __init__(self, service_name: str):
        """Initialize the example service."""
        self.service_name = service_name
        self.event_bus = InMemoryEventBus()
        self.publisher = EventPublisher(
            event_bus=self.event_bus,
            source_service=service_name
        )
        
        # Set correlation ID for all events
        self.publisher.set_correlation_id(str(uuid.uuid4()))
    
    async def start(self) -> None:
        """Start the service."""
        # Start the event bus
        await self.event_bus.start()
        
        # Subscribe to events
        self.event_bus.subscribe(
            event_types=[EventType.MARKET_DATA_UPDATED],
            handler=self.handle_market_data
        )
        
        self.event_bus.subscribe(
            event_types=[EventType.SIGNAL_GENERATED],
            handler=self.handle_trading_signal
        )
        
        logger.info(f"Service {self.service_name} started")
    
    async def stop(self) -> None:
        """Stop the service."""
        # Stop the event bus
        await self.event_bus.stop()
        
        logger.info(f"Service {self.service_name} stopped")
    
    async def handle_market_data(self, event: Event) -> None:
        """Handle market data events."""
        logger.info(f"Received market data event: {event.event_type}")
        logger.info(f"  Symbol: {event.payload.get('symbol')}")
        logger.info(f"  Price: {event.payload.get('price')}")
        logger.info(f"  Timestamp: {event.payload.get('timestamp')}")
    
    async def handle_trading_signal(self, event: Event) -> None:
        """Handle trading signal events."""
        logger.info(f"Received trading signal event: {event.event_type}")
        logger.info(f"  Symbol: {event.payload.get('symbol')}")
        logger.info(f"  Signal: {event.payload.get('signal')}")
        logger.info(f"  Confidence: {event.payload.get('confidence')}")
    
    async def publish_market_data(
        self,
        symbol: str,
        price: float,
        timestamp: str
    ) -> None:
        """Publish market data."""
        payload = {
            "symbol": symbol,
            "price": price,
            "timestamp": timestamp
        }
        
        await self.publisher.publish(
            event_type=EventType.MARKET_DATA_UPDATED,
            payload=payload,
            priority=EventPriority.HIGH
        )
    
    async def publish_trading_signal(
        self,
        symbol: str,
        signal: str,
        confidence: float
    ) -> None:
        """Publish trading signal."""
        payload = {
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence
        }
        
        await self.publisher.publish(
            event_type=EventType.SIGNAL_GENERATED,
            payload=payload
        )

# Main function
async def main():
    """Main function."""
    # Create example service
    service = ExampleService("example-service")
    
    # Start the service
    await service.start()
    
    try:
        # Publish some events
        logger.info("Publishing market data event...")
        await service.publish_market_data(
            symbol="EUR/USD",
            price=1.1234,
            timestamp="2023-01-01T12:00:00Z"
        )
        
        await asyncio.sleep(1)
        
        logger.info("Publishing trading signal event...")
        await service.publish_trading_signal(
            symbol="EUR/USD",
            signal="buy",
            confidence=0.85
        )
        
        await asyncio.sleep(1)
        
    finally:
        # Stop the service
        await service.stop()

# Run the main function
if __name__ == "__main__":
    logger.info("Starting test...")
    asyncio.run(main())
    logger.info("Test completed.")
