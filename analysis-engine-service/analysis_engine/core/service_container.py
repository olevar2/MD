"""
Service Container Module

This module provides dependency injection functionality for the Analysis Engine Service.
"""

from typing import Any, Dict, Optional, Type, TypeVar, Callable, Awaitable
import logging
from functools import wraps

T = TypeVar('T')

class ServiceContainer:
    """Service container for dependency injection."""
    
    def __init__(self):
        """Initialize the service container."""
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable[['ServiceContainer'], Awaitable[Any]]] = {}
        self._logger = logging.getLogger(__name__)

    def register_factory(self, service_type: Type[T], factory: Callable[['ServiceContainer'], Awaitable[T]]) -> None:
        """
        Register a factory function for a service type.
        
        Args:
            service_type: The type of service to register
            factory: Async factory function that creates the service
        """
        self._factories[service_type] = factory
        self._logger.debug(f"Registered factory for {service_type.__name__}")

    async def resolve(self, service_type: Type[T]) -> T:
        """
        Resolve a service instance from the container.
        
        Args:
            service_type: The type of service to resolve
            
        Returns:
            The resolved service instance
            
        Raises:
            KeyError: If no factory is registered for the service type
        """
        if service_type not in self._services:
            if service_type not in self._factories:
                raise KeyError(f"No factory registered for {service_type.__name__}")
            
            factory = self._factories[service_type]
            self._services[service_type] = await factory(self)
            
        return self._services[service_type]

    async def resolve_optional(self, service_type: Type[T]) -> Optional[T]:
        """
        Resolve a service instance if available, otherwise return None.
        
        Args:
            service_type: The type of service to resolve
            
        Returns:
            The resolved service instance or None if not available
        """
        try:
            return await self.resolve(service_type)
        except KeyError:
            return None

    async def cleanup(self) -> None:
        """Clean up all services in the container."""
        for service in self._services.values():
            if hasattr(service, 'cleanup'):
                await service.cleanup()
        self._services.clear()
        self._factories.clear()
        self._logger.info("Service container cleaned up") 