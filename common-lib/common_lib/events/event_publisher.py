"""
Event Publisher Module

This module provides helper functions for publishing events.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from .base import IEventBus, Event, EventType, EventPriority, EventMetadata


async def publish_event(
    event_bus: IEventBus,
    event_type: Union[EventType, str],
    payload: Dict[str, Any],
    source_service: str,
    correlation_id: Optional[str] = None,
    causation_id: Optional[str] = None,
    target_services: Optional[List[str]] = None,
    priority: EventPriority = EventPriority.MEDIUM,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> Event:
    """
    Create and publish an event in one operation.

    Args:
        event_bus: The event bus to publish to
        event_type: Type of the event
        payload: Event payload data
        source_service: Service that created the event
        correlation_id: Optional ID to correlate related events
        causation_id: Optional ID of the event that caused this event
        target_services: Optional list of specific target services
        priority: Priority of the event
        additional_metadata: Optional additional metadata

    Returns:
        The created event
    """
    # Create event metadata
    metadata = EventMetadata(
        source_service=source_service,
        correlation_id=correlation_id,
        causation_id=causation_id,
        priority=priority,
        additional_metadata=additional_metadata or {}
    )

    # Create event
    event = Event(
        event_type=event_type,
        payload=payload,
        metadata=metadata
    )

    await event_bus.publish(event)

    return event


class EventPublisher:
    """
    Helper class for publishing events.

    This class provides a convenient way to publish events with a consistent source service.
    """

    def __init__(
        self,
        event_bus: IEventBus,
        source_service: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the event publisher.

        Args:
            event_bus: The event bus to publish to
            source_service: The source service name
            logger: Optional logger
        """
        self.event_bus = event_bus
        self.source_service = source_service
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._correlation_id = None

    def set_correlation_id(self, correlation_id: str) -> None:
        """
        Set the correlation ID for all events published by this publisher.

        Args:
            correlation_id: The correlation ID
        """
        self._correlation_id = correlation_id

    async def publish(
        self,
        event_type: Union[EventType, str],
        payload: Dict[str, Any],
        causation_id: Optional[str] = None,
        target_services: Optional[List[str]] = None,
        priority: EventPriority = EventPriority.MEDIUM,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Event:
        """
        Publish an event.

        Args:
            event_type: Type of the event
            payload: Event payload data
            causation_id: Optional ID of the event that caused this event
            target_services: Optional list of specific target services
            priority: Priority of the event
            additional_metadata: Optional additional metadata

        Returns:
            The published event
        """
        try:
            event = await publish_event(
                event_bus=self.event_bus,
                event_type=event_type,
                payload=payload,
                source_service=self.source_service,
                correlation_id=self._correlation_id,
                causation_id=causation_id,
                target_services=target_services,
                priority=priority,
                additional_metadata=additional_metadata
            )

            self.logger.debug(f"Published event: {event_type}")

            return event

        except Exception as e:
            self.logger.error(f"Failed to publish event {event_type}: {e}", exc_info=True)
            raise
