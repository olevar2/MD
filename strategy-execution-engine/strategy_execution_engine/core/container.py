"""
Service Container for Strategy Execution Engine

This module provides a service container for dependency injection.
"""
import logging
from typing import Dict, Any, Optional, Type, TypeVar, Generic, cast
from strategy_execution_engine.core.config import get_settings
logger = logging.getLogger(__name__)
T = TypeVar('T')


from strategy_execution_engine.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ServiceContainer:
    """
    Service container for dependency injection.
    """

    def __init__(self):
        """Initialize service container."""
        self._services: Dict[str, Any] = {}
        self._initialized = False
        self._settings = get_settings()
        logger.debug('Service container created')

    @async_with_exception_handling
    async def initialize(self) ->None:
        """
        Initialize services.
        
        This method should be called during application startup.
        """
        if self._initialized:
            logger.warning('Service container already initialized')
            return
        logger.info('Initializing service container')
        try:
            self._initialized = True
            logger.info('Service container initialized successfully')
        except Exception as e:
            logger.error(f'Failed to initialize service container: {e}',
                exc_info=True)
            raise

    @async_with_exception_handling
    async def shutdown(self) ->None:
        """
        Shutdown services.
        
        This method should be called during application shutdown.
        """
        if not self._initialized:
            logger.warning(
                'Service container not initialized, nothing to shut down')
            return
        logger.info('Shutting down service container')
        try:
            self._initialized = False
            self._services.clear()
            logger.info('Service container shutdown successfully')
        except Exception as e:
            logger.error(f'Failed to shutdown service container: {e}',
                exc_info=True)
            raise

    def register(self, name: str, service: Any) ->None:
        """
        Register a service.
        
        Args:
            name: Service name
            service: Service instance
        """
        self._services[name] = service
        logger.debug(f'Registered service: {name}')

    def get(self, name: str) ->Any:
        """
        Get a service by name.
        
        Args:
            name: Service name
            
        Returns:
            Any: Service instance
            
        Raises:
            KeyError: If service not found
        """
        if name not in self._services:
            raise KeyError(f'Service not found: {name}')
        return self._services[name]

    def get_typed(self, name: str, service_type: Type[T]) ->T:
        """
        Get a service by name with type checking.
        
        Args:
            name: Service name
            service_type: Expected service type
            
        Returns:
            T: Service instance
            
        Raises:
            KeyError: If service not found
            TypeError: If service is not of expected type
        """
        service = self.get(name)
        if not isinstance(service, service_type):
            raise TypeError(
                f'Service {name} is not of type {service_type.__name__}')
        return cast(T, service)

    def has(self, name: str) ->bool:
        """
        Check if a service exists.
        
        Args:
            name: Service name
            
        Returns:
            bool: True if service exists, False otherwise
        """
        return name in self._services

    def remove(self, name: str) ->None:
        """
        Remove a service.
        
        Args:
            name: Service name
            
        Raises:
            KeyError: If service not found
        """
        if name not in self._services:
            raise KeyError(f'Service not found: {name}')
        del self._services[name]
        logger.debug(f'Removed service: {name}')

    @property
    def is_initialized(self) ->bool:
        """
        Check if service container is initialized.
        
        Returns:
            bool: True if initialized, False otherwise
        """
        return self._initialized
