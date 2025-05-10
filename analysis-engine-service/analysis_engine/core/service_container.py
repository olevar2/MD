"""
Service Container Module

This module provides dependency injection functionality for the Analysis Engine Service.
"""

from typing import Any, Dict, Optional, Type, TypeVar, Callable, Awaitable
import logging
import traceback
from functools import wraps

from analysis_engine.core.exceptions_bridge import (
    ServiceError,
    ConfigurationError,
    ServiceInitializationError,
    ServiceResolutionError,
    ServiceCleanupError,
    async_with_exception_handling,
    with_exception_handling,
    generate_correlation_id
)

T = TypeVar('T')

class ServiceContainer:
    """Service container for dependency injection."""

    def __init__(self):
        """Initialize the service container."""
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable[['ServiceContainer'], Awaitable[Any]]] = {}
        self._logger = logging.getLogger(__name__)

    @with_exception_handling
    def register_factory(self, service_type: Type[T], factory: Callable[['ServiceContainer'], Awaitable[T]]) -> None:
        """
        Register a factory function for a service type.

        Args:
            service_type: The type of service to register
            factory: Async factory function that creates the service

        Raises:
            ConfigurationError: If the factory is invalid or already registered
        """
        # Generate a correlation ID for this operation
        correlation_id = generate_correlation_id()

        # Validate inputs
        if service_type is None:
            raise ConfigurationError(
                message="Cannot register factory for None service type",
                error_code="INVALID_SERVICE_TYPE",
                details={"service_type": str(service_type)},
                correlation_id=correlation_id
            )

        if factory is None or not callable(factory):
            raise ConfigurationError(
                message=f"Factory for {service_type.__name__} must be a callable",
                error_code="INVALID_FACTORY",
                details={"service_type": service_type.__name__},
                correlation_id=correlation_id
            )

        # Check if factory is already registered
        if service_type in self._factories:
            self._logger.warning(
                f"Overwriting existing factory for {service_type.__name__}",
                extra={"correlation_id": correlation_id}
            )

        self._factories[service_type] = factory
        self._logger.debug(
            f"Registered factory for {service_type.__name__}",
            extra={"correlation_id": correlation_id}
        )

    @async_with_exception_handling
    async def resolve(self, service_type: Type[T]) -> T:
        """
        Resolve a service instance from the container.

        Args:
            service_type: The type of service to resolve

        Returns:
            The resolved service instance

        Raises:
            ServiceResolutionError: If the service cannot be resolved
        """
        # Generate a correlation ID for this operation
        correlation_id = generate_correlation_id()

        # Validate input
        if service_type is None:
            raise ServiceResolutionError(
                message="Cannot resolve None service type",
                error_code="INVALID_SERVICE_TYPE",
                details={"service_type": str(service_type)},
                correlation_id=correlation_id
            )

        try:
            # Check if service is already instantiated
            if service_type in self._services:
                return self._services[service_type]

            # Check if factory is registered
            if service_type not in self._factories:
                available_services = [t.__name__ for t in self._factories.keys()]
                raise ServiceResolutionError(
                    message=f"No factory registered for {service_type.__name__}",
                    error_code="FACTORY_NOT_FOUND",
                    details={
                        "service_type": service_type.__name__,
                        "available_services": available_services
                    },
                    correlation_id=correlation_id
                )

            # Get factory and create service
            factory = self._factories[service_type]

            try:
                self._logger.debug(
                    f"Creating service instance for {service_type.__name__}",
                    extra={"correlation_id": correlation_id}
                )
                self._services[service_type] = await factory(self)

                return self._services[service_type]
            except Exception as e:
                # Wrap factory exceptions in ServiceInitializationError
                raise ServiceInitializationError(
                    message=f"Failed to initialize service {service_type.__name__}",
                    error_code="SERVICE_INITIALIZATION_FAILED",
                    details={
                        "service_type": service_type.__name__,
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    },
                    correlation_id=correlation_id
                ) from e

        except ServiceError:
            # Re-raise ServiceError exceptions
            raise
        except Exception as e:
            # Wrap other exceptions
            raise ServiceResolutionError(
                message=f"Unexpected error resolving service {service_type.__name__}",
                error_code="SERVICE_RESOLUTION_FAILED",
                details={
                    "service_type": service_type.__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                },
                correlation_id=correlation_id
            ) from e

    @async_with_exception_handling
    async def resolve_optional(self, service_type: Type[T]) -> Optional[T]:
        """
        Resolve a service instance if available, otherwise return None.

        Args:
            service_type: The type of service to resolve

        Returns:
            The resolved service instance or None if not available
        """
        # Generate a correlation ID for this operation
        correlation_id = generate_correlation_id()

        try:
            return await self.resolve(service_type)
        except ServiceResolutionError as e:
            if e.error_code == "FACTORY_NOT_FOUND":
                self._logger.debug(
                    f"Optional service {service_type.__name__} not found, returning None",
                    extra={"correlation_id": correlation_id}
                )
                return None
            # Re-raise other service resolution errors
            raise
        except Exception as e:
            self._logger.warning(
                f"Unexpected error resolving optional service {service_type.__name__}: {str(e)}",
                extra={"correlation_id": correlation_id}
            )
            return None

    @async_with_exception_handling
    async def cleanup(self) -> None:
        """
        Clean up all services in the container.

        Raises:
            ServiceCleanupError: If cleanup fails for any service
        """
        # Generate a correlation ID for this operation
        correlation_id = generate_correlation_id()

        cleanup_errors = []

        # Clean up each service
        for service_type, service in self._services.items():
            if hasattr(service, 'cleanup') and callable(service.cleanup):
                try:
                    self._logger.debug(
                        f"Cleaning up service {service_type.__name__}",
                        extra={"correlation_id": correlation_id}
                    )
                    await service.cleanup()
                except Exception as e:
                    error_msg = f"Error cleaning up service {service_type.__name__}: {str(e)}"
                    self._logger.error(
                        error_msg,
                        extra={"correlation_id": correlation_id},
                        exc_info=True
                    )
                    cleanup_errors.append({
                        "service_type": service_type.__name__,
                        "error": str(e)
                    })

        # Clear services and factories
        self._services.clear()
        self._factories.clear()

        # Log cleanup completion
        self._logger.info(
            f"Service container cleaned up ({len(cleanup_errors)} errors)",
            extra={"correlation_id": correlation_id}
        )

        # Raise exception if there were cleanup errors
        if cleanup_errors:
            raise ServiceCleanupError(
                message=f"Errors occurred during service cleanup ({len(cleanup_errors)} services affected)",
                error_code="SERVICE_CLEANUP_FAILED",
                details={"cleanup_errors": cleanup_errors},
                correlation_id=correlation_id
            )