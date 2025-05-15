"""
Event Bus Interface Module

This module defines the interface for event buses in the Forex Trading Platform.
All event bus implementations must implement this interface.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set, Union, Awaitable

from .event_schema import Event, EventType

# Type definitions
EventHandler = Callable[[Event], Awaitable[None]]
EventFilter = Callable[[Event], bool]


class IEventBus(ABC):
    """
    Interface for event buses.
    
    All event bus implementations must implement this interface.
    """
    
    @abstractmethod
    async def publish(self, event: Event) -> None:
        """
        Publish an event to the event bus.
        
        Args:
            event: Event to publish
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def unsubscribe_from_all(
        self,
        handler: EventHandler
    ) -> None:
        """
        Unsubscribe from all events.
        
        Args:
            handler: Event handler to unsubscribe
        """
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """
        Start the event bus.
        
        This method should be called before using the event bus.
        """
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """
        Stop the event bus.
        
        This method should be called when the event bus is no longer needed.
        """
        pass
