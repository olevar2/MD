"""
Event Bus Module

This module provides an event bus for publishing and subscribing to events.
"""

import asyncio
import logging
import traceback
from typing import Dict, Any, Optional, List, Callable, Set, Awaitable, Union, TypeVar

from .base import Event, EventHandler


class EventBus:
    """
    Event bus for publishing and subscribing to events.

    This class provides a simple in-memory event bus for publishing and subscribing to events.
    For production use, it should be replaced with a more robust solution like RabbitMQ or Kafka.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the event bus.

        Args:
            logger: Logger to use (if None, creates a new logger)
        """
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.subscribers: Dict[str, Set[EventHandler]] = {}
        self.global_subscribers: Set[EventHandler] = set()

    async def publish(self, event: Event) -> None:
        """
        Publish an event to the event bus.

        Args:
            event: Event to publish
        """
        event_type = event.event_type
        routing_key = event.get_routing_key()

        self.logger.debug(f"Publishing event: {event_type} (routing key: {routing_key})")

        # Get subscribers for this event type
        handlers = set()
        handlers.update(self.subscribers.get(event_type, set()))
        handlers.update(self.subscribers.get(routing_key, set()))
        handlers.update(self.global_subscribers)

        # Publish event to subscribers
        tasks = []
        for handler in handlers:
            tasks.append(self._call_handler(handler, event))

        # Wait for all handlers to complete
        if tasks:
            await asyncio.gather(*tasks)

    async def _call_handler(self, handler: EventHandler, event: Event) -> None:
        """
        Call an event handler with error handling.

        Args:
            handler: Event handler to call
            event: Event to pass to the handler
        """
        try:
            await handler(event)
        except Exception as e:
            self.logger.error(
                f"Error in event handler {handler.__name__} for event {event.event_type}: {str(e)}"
            )
            self.logger.debug(traceback.format_exc())

    def subscribe(
        self,
        event_type: str,
        handler: EventHandler
    ) -> Callable[[], None]:
        """
        Subscribe to events of a specific type.

        Args:
            event_type: Type of events to subscribe to
            handler: Event handler to call when an event of this type is published

        Returns:
            Function to unsubscribe from events
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = set()

        self.subscribers[event_type].add(handler)
        self.logger.debug(f"Subscribed {handler.__name__} to event type: {event_type}")

        # Return unsubscribe function
        def unsubscribe():
            self.unsubscribe(event_type, handler)

        return unsubscribe

    def subscribe_all(self, handler: EventHandler) -> Callable[[], None]:
        """
        Subscribe to all events.

        Args:
            handler: Event handler to call when any event is published

        Returns:
            Function to unsubscribe from all events
        """
        self.global_subscribers.add(handler)
        self.logger.debug(f"Subscribed {handler.__name__} to all events")

        # Return unsubscribe function
        def unsubscribe():
            self.unsubscribe_all(handler)

        return unsubscribe

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """
        Unsubscribe from events of a specific type.

        Args:
            event_type: Type of events to unsubscribe from
            handler: Event handler to unsubscribe
        """
        if event_type in self.subscribers and handler in self.subscribers[event_type]:
            self.subscribers[event_type].remove(handler)
            self.logger.debug(f"Unsubscribed {handler.__name__} from event type: {event_type}")

            # Remove empty subscriber sets
            if not self.subscribers[event_type]:
                del self.subscribers[event_type]

    def unsubscribe_all(self, handler: EventHandler) -> None:
        """
        Unsubscribe from all events.

        Args:
            handler: Event handler to unsubscribe
        """
        if handler in self.global_subscribers:
            self.global_subscribers.remove(handler)
            self.logger.debug(f"Unsubscribed {handler.__name__} from all events")

        # Also remove from specific event types
        for event_type in list(self.subscribers.keys()):
            if handler in self.subscribers[event_type]:
                self.subscribers[event_type].remove(handler)

                # Remove empty subscriber sets
                if not self.subscribers[event_type]:
                    del self.subscribers[event_type]


class EventBusManager:
    """
    Manager for event buses.

    This class provides a singleton manager for event buses.
    """

    _instance = None
    _event_buses: Dict[str, EventBus] = {}

    def __new__(cls, *args, **kwargs):
        """
        Create a new instance of the event bus manager.

        Returns:
            Singleton instance of the event bus manager
        """
        if cls._instance is None:
            cls._instance = super(EventBusManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the event bus manager.

        Args:
            logger: Logger to use (if None, creates a new logger)
        """
        # Skip initialization if already initialized
        if getattr(self, "_initialized", False):
            return

        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._event_buses = {}
        self._initialized = True

    def get_event_bus(self, name: str = "default") -> EventBus:
        """
        Get an event bus by name.

        Args:
            name: Name of the event bus

        Returns:
            Event bus
        """
        if name not in self._event_buses:
            self._event_buses[name] = EventBus(logger=self.logger)

        return self._event_buses[name]

    def remove_event_bus(self, name: str) -> None:
        """
        Remove an event bus.

        Args:
            name: Name of the event bus to remove
        """
        if name in self._event_buses:
            del self._event_buses[name]