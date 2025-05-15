"""
In-Memory Event Bus Module

This module provides an in-memory implementation of the event bus interface.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Union, Awaitable

from .base import IEventBus, Event, EventType, EventHandler, EventFilter


class InMemoryEventBus(IEventBus):
    """
    In-memory implementation of the event bus.

    This class provides a simple in-memory event bus for publishing and subscribing to events.
    It's suitable for development, testing, and small-scale deployments.
    For production use, consider using a more robust solution like RabbitMQ or Kafka.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the event bus.

        Args:
            logger: Logger to use (if None, creates a new logger)
        """
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.subscribers: Dict[str, Set[tuple[EventHandler, Optional[EventFilter]]]] = {}
        self.global_subscribers: Set[tuple[EventHandler, Optional[EventFilter]]] = set()
        self._running = False

    async def publish(self, event: Event) -> None:
        """
        Publish an event to the event bus.

        Args:
            event: Event to publish
        """
        if not self._running:
            self.logger.warning("Event bus is not running. Event will not be published.")
            return

        event_type = str(event.event_type)
        routing_key = event.get_routing_key()

        self.logger.debug(f"Publishing event: {event_type} (routing key: {routing_key})")

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
        """
        Call an event handler with an event.

        Args:
            handler: Event handler to call
            event: Event to pass to the handler
        """
        try:
            await handler(event)
        except Exception as e:
            handler_name = getattr(handler, "__name__", str(handler))
            self.logger.error(f"Error in event handler {handler_name}: {e}", exc_info=True)

    def subscribe(
        self,
        event_types: Union[str, EventType, List[Union[str, EventType]]],
        handler: EventHandler,
        filter_func: Optional[EventFilter] = None
    ) -> Callable[[], None]:
        """
        Subscribe to events of the specified types.

        Args:
            event_types: Type(s) of events to subscribe to
            handler: Function to handle events
            filter_func: Optional function to filter events

        Returns:
            Function to unsubscribe from events
        """
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
            self.logger.debug(f"Subscribed {handler_name} to event type: {event_type}")

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
        """
        Unsubscribe from events of a specific type.

        Args:
            event_type: Type of events to unsubscribe from
            handler: Event handler to unsubscribe
        """
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
                self.logger.debug(f"Unsubscribed {handler_name} from event type: {event_type}")

                # Remove empty subscriber sets
                if not self.subscribers[event_type]:
                    del self.subscribers[event_type]

    def subscribe_to_all(
        self,
        handler: EventHandler,
        filter_func: Optional[EventFilter] = None
    ) -> Callable[[], None]:
        """
        Subscribe to all events.

        Args:
            handler: Function to handle events
            filter_func: Optional function to filter events

        Returns:
            Function to unsubscribe from events
        """
        self.global_subscribers.add((handler, filter_func))
        handler_name = getattr(handler, "__name__", str(handler))
        self.logger.debug(f"Subscribed {handler_name} to all events")

        # Return unsubscribe function
        def unsubscribe():
            self.unsubscribe_from_all(handler)

        return unsubscribe

    def unsubscribe_from_all(
        self,
        handler: EventHandler
    ) -> None:
        """
        Unsubscribe from all events.

        Args:
            handler: Event handler to unsubscribe
        """
        # Find and remove the handler from global subscribers
        to_remove = None
        for h, f in self.global_subscribers:
            if h == handler:
                to_remove = (h, f)
                break

        if to_remove:
            self.global_subscribers.remove(to_remove)
            handler_name = getattr(handler, "__name__", str(handler))
            self.logger.debug(f"Unsubscribed {handler_name} from all events")

        # Also remove from specific event types
        for event_type in list(self.subscribers.keys()):
            to_remove = None
            for h, f in self.subscribers[event_type]:
                if h == handler:
                    to_remove = (h, f)
                    break

            if to_remove:
                self.subscribers[event_type].remove(to_remove)

                # Remove empty subscriber sets
                if not self.subscribers[event_type]:
                    del self.subscribers[event_type]

    async def start(self) -> None:
        """
        Start the event bus.

        This method should be called before using the event bus.
        """
        self._running = True
        self.logger.info("In-memory event bus started")

    async def stop(self) -> None:
        """
        Stop the event bus.

        This method should be called when the event bus is no longer needed.
        """
        self._running = False
        self.logger.info("In-memory event bus stopped")
