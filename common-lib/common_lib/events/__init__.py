"""
Events Module

This module provides standardized event handling for the Forex Trading Platform.
It includes event schemas, topic naming conventions, and event bus abstractions.
"""

# Base types and interfaces
from .base import (
    Event, EventType, EventPriority, EventMetadata,
    IEventBus, EventHandler, EventFilter
)

# Event bus implementations
from .in_memory_event_bus import InMemoryEventBus
from .kafka_event_bus_v2 import KafkaEventBusV2, KafkaConfig

# Event bus factory
from .event_bus_factory import EventBusFactory, EventBusType, EventBusManager

# Event publisher
from .event_publisher import EventPublisher, publish_event

# Legacy imports for backward compatibility
from .event_bus import EventBus

__all__ = [
    # Base types and interfaces
    'Event',
    'EventType',
    'EventPriority',
    'EventMetadata',
    'IEventBus',
    'EventHandler',
    'EventFilter',

    # Event bus implementations
    'InMemoryEventBus',
    'KafkaEventBusV2',
    'KafkaConfig',

    # Event bus factory
    'EventBusFactory',
    'EventBusType',
    'EventBusManager',

    # Event publisher
    'EventPublisher',
    'publish_event',

    # Legacy imports for backward compatibility
    'EventBus'
]
