"""
Base Event Module

This module defines the base types and interfaces for the event-driven architecture.
"""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Union, TypeVar

# Type definitions
EventHandler = Callable[['Event'], Awaitable[None]]
EventFilter = Callable[['Event'], bool]
T = TypeVar('T')


class EventType(str, Enum):
    """Standard event types for the Forex Trading Platform."""
    
    # Trading events
    ORDER_CREATED = "order.created"
    ORDER_UPDATED = "order.updated"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_FILLED = "order.filled"
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"
    POSITION_UPDATED = "position.updated"
    
    # Market data events
    MARKET_DATA_UPDATED = "market.data.updated"
    MARKET_VOLATILITY_CHANGE = "market.volatility.change"
    MARKET_REGIME_CHANGE = "market.regime.change"
    
    # Analysis events
    ANALYSIS_COMPLETED = "analysis.completed"
    SIGNAL_GENERATED = "signal.generated"
    PATTERN_DETECTED = "pattern.detected"
    
    # ML events
    MODEL_TRAINED = "model.trained"
    MODEL_EVALUATED = "model.evaluated"
    PREDICTION_GENERATED = "prediction.generated"
    
    # Risk events
    RISK_LIMIT_BREACH = "risk.limit.breach"
    RISK_PARAMETERS_UPDATED = "risk.parameters.updated"
    
    # System events
    SERVICE_STARTED = "service.started"
    SERVICE_STOPPED = "service.stopped"
    SERVICE_COMMAND = "service.command"
    SERVICE_STATUS_CHANGED = "service.status.changed"
    SERVICE_DEGRADED = "service.degraded"
    
    # Feedback events
    FEEDBACK_COLLECTED = "feedback.collected"
    FEEDBACK_PROCESSED = "feedback.processed"
    FEEDBACK_ANALYSIS = "feedback.analysis"
    
    # Error events
    ERROR_OCCURRED = "error.occurred"
    
    # Custom event type
    CUSTOM = "custom"


class EventPriority(str, Enum):
    """Priority levels for events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventMetadata:
    """Metadata for events."""
    
    def __init__(
        self,
        source_service: str,
        correlation_id: Optional[str] = None,
        causation_id: Optional[str] = None,
        priority: EventPriority = EventPriority.MEDIUM,
        additional_metadata: Optional[Dict[str, Any]] = None
    ):
        self.event_id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow().isoformat()
        self.source_service = source_service
        self.correlation_id = correlation_id or self.event_id
        self.causation_id = causation_id
        self.priority = priority
        self.additional_metadata = additional_metadata or {}


class Event:
    """Base event class."""
    
    def __init__(
        self,
        event_type: Union[EventType, str],
        payload: Dict[str, Any],
        metadata: Optional[EventMetadata] = None,
        source_service: Optional[str] = None
    ):
        self.event_type = event_type
        self.payload = payload
        
        if metadata is None and source_service is not None:
            metadata = EventMetadata(source_service=source_service)
        
        if metadata is None:
            raise ValueError("Either metadata or source_service must be provided")
            
        self.metadata = metadata
    
    def get_routing_key(self) -> str:
        """Get the routing key for this event."""
        return str(self.event_type)


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
