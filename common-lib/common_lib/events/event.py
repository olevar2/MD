"""
Event Module

This module defines the base event class and event types for the event-driven architecture.
"""

import uuid
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, ClassVar, Type, TypeVar, Generic

from pydantic import BaseModel, Field, validator

T = TypeVar('T', bound=BaseModel)


class Event(BaseModel, Generic[T]):
    """
    Base event class for all events in the system.
    
    All events should inherit from this class and define their own payload type.
    """
    
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for the event."""
    
    event_type: str
    """Type of the event."""
    
    event_version: str = "1.0"
    """Version of the event schema."""
    
    source: str
    """Source of the event (service or component that generated it)."""
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    """Timestamp when the event was generated."""
    
    correlation_id: Optional[str] = None
    """Correlation ID for tracking related events."""
    
    causation_id: Optional[str] = None
    """Causation ID for tracking event causality."""
    
    payload: T
    """Payload of the event."""
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    """Additional metadata for the event."""
    
    @validator('correlation_id', always=True)
    def set_correlation_id(cls, v, values):
        """Set correlation ID if not provided."""
        if v is None:
            return values.get('event_id')
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the event to a dictionary.
        
        Returns:
            Dictionary representation of the event
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "event_version": self.event_version,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "payload": self.payload.dict(),
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """
        Convert the event to a JSON string.
        
        Returns:
            JSON string representation of the event
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """
        Create an event from a dictionary.
        
        Args:
            data: Dictionary representation of the event
            
        Returns:
            Event instance
        """
        # Convert timestamp string to datetime
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Event':
        """
        Create an event from a JSON string.
        
        Args:
            json_str: JSON string representation of the event
            
        Returns:
            Event instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_routing_key(self) -> str:
        """
        Get the routing key for the event.
        
        The routing key is used by the message broker to route the event to the appropriate queues.
        
        Returns:
            Routing key for the event
        """
        return f"{self.source}.{self.event_type}"
    
    def with_causation(self, cause_event: 'Event') -> 'Event':
        """
        Create a new event with causation information from another event.
        
        Args:
            cause_event: Event that caused this event
            
        Returns:
            New event with causation information
        """
        return self.copy(update={
            "correlation_id": cause_event.correlation_id,
            "causation_id": cause_event.event_id
        })


class EventRegistry:
    """
    Registry for event types.
    
    This class maintains a registry of event types and their payload types.
    """
    
    _registry: ClassVar[Dict[str, Type[Event]]] = {}
    
    @classmethod
    def register(cls, event_type: str, event_class: Type[Event]) -> None:
        """
        Register an event type.
        
        Args:
            event_type: Type of the event
            event_class: Event class
        """
        cls._registry[event_type] = event_class
    
    @classmethod
    def get(cls, event_type: str) -> Optional[Type[Event]]:
        """
        Get an event class by type.
        
        Args:
            event_type: Type of the event
            
        Returns:
            Event class, or None if not found
        """
        return cls._registry.get(event_type)
    
    @classmethod
    def list_types(cls) -> List[str]:
        """
        Get a list of all registered event types.
        
        Returns:
            List of event types
        """
        return list(cls._registry.keys())


# Common event types
class MarketDataUpdatedPayload(BaseModel):
    """Payload for market data updated events."""
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataUpdatedEvent(Event[MarketDataUpdatedPayload]):
    """Event for market data updates."""
    event_type: str = "market.data.updated"


class IndicatorCalculatedPayload(BaseModel):
    """Payload for indicator calculated events."""
    indicator_name: str
    symbol: str
    timeframe: str
    timestamp: datetime
    values: Dict[str, float]


class IndicatorCalculatedEvent(Event[IndicatorCalculatedPayload]):
    """Event for indicator calculation results."""
    event_type: str = "indicator.calculated"


class OrderPlacedPayload(BaseModel):
    """Payload for order placed events."""
    order_id: str
    symbol: str
    order_type: str
    side: str
    quantity: float
    price: Optional[float] = None
    status: str
    timestamp: datetime


class OrderPlacedEvent(Event[OrderPlacedPayload]):
    """Event for order placement."""
    event_type: str = "order.placed"


class OrderExecutedPayload(BaseModel):
    """Payload for order executed events."""
    order_id: str
    symbol: str
    order_type: str
    side: str
    quantity: float
    price: float
    timestamp: datetime


class OrderExecutedEvent(Event[OrderExecutedPayload]):
    """Event for order execution."""
    event_type: str = "order.executed"


# Register common event types
EventRegistry.register(MarketDataUpdatedEvent.event_type, MarketDataUpdatedEvent)
EventRegistry.register(IndicatorCalculatedEvent.event_type, IndicatorCalculatedEvent)
EventRegistry.register(OrderPlacedEvent.event_type, OrderPlacedEvent)
EventRegistry.register(OrderExecutedEvent.event_type, OrderExecutedEvent)