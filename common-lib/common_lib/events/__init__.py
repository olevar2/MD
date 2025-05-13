"""
Events Module

This module provides standardized event handling for the Forex Trading Platform.
It includes event schemas, topic naming conventions, and event bus abstractions.
"""

from .event_schema import Event, EventType, EventMetadata, create_event
from .event_bus import EventBus, EventHandler, EventFilter
from .kafka_event_bus import KafkaEventBus, KafkaConfig

__all__ = [
    'Event',
    'EventType',
    'EventMetadata',
    'create_event',
    'EventBus',
    'EventHandler',
    'EventFilter',
    'KafkaEventBus',
    'KafkaConfig'
]
