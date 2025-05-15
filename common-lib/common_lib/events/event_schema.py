"""
Event Schema Module

This module defines the standard event schema for the Forex Trading Platform.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic

from pydantic import BaseModel, Field, root_validator


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


class EventMetadata(BaseModel):
    """Metadata for events."""

    # Unique identifier for the event
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Timestamp when the event was created
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Service that created the event
    source_service: str

    # Optional correlation ID for tracing related events
    correlation_id: Optional[str] = None

    # Optional ID of the event that caused this event
    causation_id: Optional[str] = None

    # Optional list of specific target services
    target_services: Optional[List[str]] = None

    # Optional version of the event schema
    schema_version: str = "1.0"

    # Event priority
    priority: EventPriority = EventPriority.MEDIUM

    # Optional additional metadata
    additional_metadata: Dict[str, Any] = Field(default_factory=dict)


# Type variable for generic event payload
T = TypeVar('T', bound=BaseModel)


class Event(BaseModel, Generic[T]):
    """Standard event schema for the Forex Trading Platform."""

    # Type of the event
    event_type: Union[EventType, str]

    # Event metadata
    metadata: EventMetadata

    # Event payload data - can be either a Dict or a specific payload model
    payload: Union[Dict[str, Any], T]

    def get_routing_key(self) -> str:
        """
        Get the routing key for this event.

        The routing key is used to route the event to the appropriate handlers.
        By default, it's the event type, but it can be overridden by subclasses.

        Returns:
            Routing key for the event
        """
        return str(self.event_type)

    @root_validator(pre=True)
    def handle_legacy_data_field(cls, values):
        """
        Handle legacy 'data' field for backward compatibility.

        If 'data' is present but 'payload' is not, use 'data' as 'payload'.
        """
        if 'data' in values and 'payload' not in values:
            values['payload'] = values.pop('data')
        return values


def create_event(
    event_type: Union[EventType, str],
    payload: Union[Dict[str, Any], BaseModel],
    source_service: str,
    correlation_id: Optional[str] = None,
    causation_id: Optional[str] = None,
    target_services: Optional[List[str]] = None,
    priority: EventPriority = EventPriority.MEDIUM,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> Event:
    """
    Create a new event.

    Args:
        event_type: Type of the event
        payload: Event payload data or model
        source_service: Service that created the event
        correlation_id: Optional ID to correlate related events
        causation_id: Optional ID of the event that caused this event
        target_services: Optional list of specific target services
        priority: Priority of the event
        additional_metadata: Optional additional metadata

    Returns:
        A new Event instance
    """
    metadata = EventMetadata(
        source_service=source_service,
        correlation_id=correlation_id,
        causation_id=causation_id,
        target_services=target_services,
        priority=priority,
        additional_metadata=additional_metadata or {}
    )

    # If payload is a BaseModel, convert it to dict for backward compatibility
    if isinstance(payload, BaseModel):
        payload_dict = payload.dict()
    else:
        payload_dict = payload

    return Event(
        event_type=event_type,
        metadata=metadata,
        payload=payload_dict
    )


# Event Registry for dynamic event type resolution
class EventRegistry:
    """Registry for event types and their corresponding event classes."""

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, event_type: str, event_class: type) -> None:
        """
        Register an event class for a specific event type.

        Args:
            event_type: Type of the event
            event_class: Event class to register
        """
        cls._registry[event_type] = event_class

    @classmethod
    def get(cls, event_type: str) -> Optional[type]:
        """
        Get the event class for a specific event type.

        Args:
            event_type: Type of the event

        Returns:
            Event class or None if not found
        """
        return cls._registry.get(event_type)
