"""
Event Bus Interface

This module defines the interface for event buses in the Forex Trading Platform.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Union

from .event_schema import Event, EventType

# Type definitions
EventHandler = Callable[[Event], None]
EventFilter = Callable[[Event], bool]


class EventBus(ABC):
    """
    Interface for event buses in the Forex Trading Platform.
    
    This abstract class defines the interface that all event bus implementations must follow.
    """
    
    @abstractmethod
    def publish(self, event: Event) -> None:
        """
        Publish an event to the event bus.
        
        Args:
            event: The event to publish
        """
        pass
    
    @abstractmethod
    def subscribe(
        self,
        event_types: List[Union[EventType, str]],
        handler: EventHandler,
        filter_func: Optional[EventFilter] = None
    ) -> None:
        """
        Subscribe to events of the specified types.
        
        Args:
            event_types: List of event types to subscribe to
            handler: Function to handle events
            filter_func: Optional function to filter events
        """
        pass
    
    @abstractmethod
    def unsubscribe(
        self,
        event_types: List[Union[EventType, str]],
        handler: EventHandler
    ) -> None:
        """
        Unsubscribe from events of the specified types.
        
        Args:
            event_types: List of event types to unsubscribe from
            handler: Function to unsubscribe
        """
        pass
    
    @abstractmethod
    def start_consuming(self, blocking: bool = True) -> None:
        """
        Start consuming events.
        
        Args:
            blocking: Whether to block the current thread
        """
        pass
    
    @abstractmethod
    def stop_consuming(self) -> None:
        """Stop consuming events."""
        pass
