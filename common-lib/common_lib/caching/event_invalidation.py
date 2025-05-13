"""
Event-based Cache Invalidation Strategy for Forex Trading Platform.

This module provides an event-based cache invalidation strategy.
"""

import logging
import threading
from typing import Dict, Any, Optional, Union, Callable, List, Set, TypeVar
from datetime import datetime, timedelta

# Local imports
from common_lib.caching.cache_service import get_cache_service, CacheService
from common_lib.caching.invalidation import CacheInvalidationStrategy

# Configure logging
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')

class EventInvalidationStrategy(CacheInvalidationStrategy[Any]):
    """
    Event-based cache invalidation strategy.
    
    This strategy invalidates cache entries based on events.
    """
    
    def __init__(
        self,
        cache_service: Optional[CacheService] = None
    ):
        """
        Initialize the event invalidation strategy.
        
        Args:
            cache_service: Cache service
        """
        self.cache_service = cache_service or get_cache_service()
        self.event_handlers: Dict[str, Set[Callable[[str, Any], None]]] = {}
        self.key_events: Dict[str, Set[str]] = {}
        self.lock = threading.RLock()
    
    def should_invalidate(self, key: str, value: Any) -> bool:
        """
        Check if a cache entry should be invalidated based on events.
        
        This is a no-op for event-based invalidation, as invalidation is
        triggered by events.
        
        Args:
            key: Cache key
            value: Cached value
            
        Returns:
            False (invalidation is triggered by events)
        """
        return False
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate a cache entry.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        return self.cache_service.delete(key)
    
    def register_dependency(self, key: str, dependency: str) -> None:
        """
        Register a dependency for a cache key.
        
        This is a no-op for event-based invalidation.
        
        Args:
            key: Cache key
            dependency: Dependency key
        """
        pass
    
    def register_event_handler(self, event: str, handler: Callable[[str, Any], None]) -> None:
        """
        Register an event handler.
        
        Args:
            event: Event name
            handler: Event handler
        """
        with self.lock:
            if event not in self.event_handlers:
                self.event_handlers[event] = set()
            self.event_handlers[event].add(handler)
    
    def register_key_for_event(self, key: str, event: str) -> None:
        """
        Register a key to be invalidated when an event occurs.
        
        Args:
            key: Cache key
            event: Event name
        """
        with self.lock:
            if key not in self.key_events:
                self.key_events[key] = set()
            self.key_events[key].add(event)
    
    def trigger_event(self, event: str, data: Any = None) -> None:
        """
        Trigger an event.
        
        Args:
            event: Event name
            data: Event data
        """
        with self.lock:
            # Get all handlers for this event
            handlers = self.event_handlers.get(event, set())
            
            # Call all handlers
            for handler in handlers:
                try:
                    handler(event, data)
                except Exception as e:
                    logger.warning(f"Error in event handler for event '{event}': {e}")
            
            # Invalidate all keys registered for this event
            keys_to_invalidate = [key for key, events in self.key_events.items() if event in events]
            
            for key in keys_to_invalidate:
                self.invalidate(key)
