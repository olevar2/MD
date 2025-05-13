"""
Event Schema Module

This module defines the standard event schema for the Forex Trading Platform.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


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
    
    # Feedback events
    FEEDBACK_COLLECTED = "feedback.collected"
    FEEDBACK_PROCESSED = "feedback.processed"
    
    # Error events
    ERROR_OCCURRED = "error.occurred"
    
    # Custom event type
    CUSTOM = "custom"


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
    
    # Optional additional metadata
    additional_metadata: Dict[str, Any] = Field(default_factory=dict)


class Event(BaseModel):
    """Standard event schema for the Forex Trading Platform."""
    
    # Type of the event
    event_type: Union[EventType, str]
    
    # Event metadata
    metadata: EventMetadata
    
    # Event payload data
    data: Dict[str, Any]


def create_event(
    event_type: Union[EventType, str],
    data: Dict[str, Any],
    source_service: str,
    correlation_id: Optional[str] = None,
    causation_id: Optional[str] = None,
    target_services: Optional[List[str]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> Event:
    """
    Create a new event.
    
    Args:
        event_type: Type of the event
        data: Event payload data
        source_service: Service that created the event
        correlation_id: Optional ID to correlate related events
        causation_id: Optional ID of the event that caused this event
        target_services: Optional list of specific target services
        additional_metadata: Optional additional metadata
        
    Returns:
        A new Event instance
    """
    metadata = EventMetadata(
        source_service=source_service,
        correlation_id=correlation_id,
        causation_id=causation_id,
        target_services=target_services,
        additional_metadata=additional_metadata or {}
    )
    
    return Event(
        event_type=event_type,
        metadata=metadata,
        data=data
    )
