"""
Service Container Implementation

This module provides a standardized service container for dependency injection
across all services in the Forex Trading Platform.
"""

import asyncio
import inspect
import logging
from enum import Enum
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar, Union, cast, get_type_hints

T = TypeVar('T')
TService = TypeVar('TService')


class ServiceLifetime(Enum):
    """Service lifetime options."""
    SINGLETON = "singleton"  # One instance per container
    TRANSIENT = "transient"  # New instance each time
    SCOPED = "scoped"        # One instance per scope


class ServiceDescriptor(Generic[T]):
    """Describes a service registration."""
    
    def __init__(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type] = None,
        factory: Optional[Callable[..., T]] = None,
        instance: Optional[T] = None,
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON
    ):
        """
        Initialize a service descriptor.
        
        Args:
            service_type: The type of service being registered
            implementation_type: The concrete implementation type
            factory: Factory function to create the service
            instance: Existing instance of the service
            lifetime: The service lifetime
        """
        self.service_type = service_type
        self.implementation_type = implementation_type
        self.factory = factory
        self.instance = instance
        self.lifetime = lifetime
        
        # Validate that exactly one of implementation_type, factory, or instance is provided
        provided_count = sum(1 for x in [implementation_type, factory, instance] if x is not None)
        if provided_count != 1:
            raise ValueError(
                "Exactly one of implementation_type, factory, or instance must be provided"
            )


class ServiceProvider:
    """
    Provides access to services from a container.
    
    This class is used to resolve services from a container and manage their lifecycle.
    """
    
    def __init__(self, container: 'ServiceContainer'):
        """
        Initialize a service provider.
        
        Args:
            container: The service container to use
        """
        self._container = container
        self._instances: Dict[Type, Any] = {}
        self._logger = logging.getLogger(__name__)
    
    def get_service(self, service_type: Type[T]) -> T:
        """
        Get a service of the specified type.
        
        Args:
            service_type: The type of service to get
            
        Returns:
            An instance of the service
            
        Raises:
            KeyError: If the service type is not registered
        """
        descriptor = self._container.get_descriptor(service_type)
        
        # For singletons, check if we already have an instance
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            if descriptor.instance is not None:
                return descriptor.instance
            
            if service_type in self._instances:
                return self._instances[service_type]
        
        # Create a new instance
        instance = self._create_instance(descriptor)
        
        # Store singleton instances
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            self._instances[service_type] = instance
        
        return instance
    
    async def get_service_async(self, service_type: Type[T]) -> T:
        """
        Get a service of the specified type asynchronously.
        
        Args:
            service_type: The type of service to get
            
        Returns:
            An instance of the service
            
        Raises:
            KeyError: If the service type is not registered
        """
        descriptor = self._container.get_descriptor(service_type)
        
        # For singletons, check if we already have an instance
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            if descriptor.instance is not None:
                return descriptor.instance
            
            if service_type in self._instances:
                return self._instances[service_type]
        
        # Create a new instance
        instance = await self._create_instance_async(descriptor)
        
        # Store singleton instances
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            self._instances[service_type] = instance
        
        return instance
    
    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """
        Create an instance from a service descriptor.
        
        Args:
            descriptor: The service descriptor
            
        Returns:
            An instance of the service
        """
        if descriptor.instance is not None:
            return descriptor.instance
        
        if descriptor.factory is not None:
            # Resolve factory dependencies
            kwargs = self._resolve_dependencies(descriptor.factory)
            return descriptor.factory(**kwargs)
        
        if descriptor.implementation_type is not None:
            # Resolve constructor dependencies
            kwargs = self._resolve_dependencies(descriptor.implementation_type.__init__)
            return descriptor.implementation_type(**kwargs)
        
        raise ValueError("Cannot create instance: no implementation, factory, or instance provided")
    
    async def _create_instance_async(self, descriptor: ServiceDescriptor) -> Any:
        """
        Create an instance from a service descriptor asynchronously.
        
        Args:
            descriptor: The service descriptor
            
        Returns:
            An instance of the service
        """
        if descriptor.instance is not None:
            return descriptor.instance
        
        if descriptor.factory is not None:
            # Check if the factory is async
            if asyncio.iscoroutinefunction(descriptor.factory):
                # Resolve factory dependencies
                kwargs = await self._resolve_dependencies_async(descriptor.factory)
                return await descriptor.factory(**kwargs)
            else:
                # Resolve factory dependencies
                kwargs = self._resolve_dependencies(descriptor.factory)
                return descriptor.factory(**kwargs)
        
        if descriptor.implementation_type is not None:
            # Resolve constructor dependencies
            kwargs = self._resolve_dependencies(descriptor.implementation_type.__init__)
            return descriptor.implementation_type(**kwargs)
        
        raise ValueError("Cannot create instance: no implementation, factory, or instance provided")
    
    def _resolve_dependencies(self, func: Callable) -> Dict[str, Any]:
        """
        Resolve dependencies for a function.
        
        Args:
            func: The function to resolve dependencies for
            
        Returns:
            A dictionary of parameter names to resolved services
        """
        if func is object.__init__:
            return {}
        
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        kwargs = {}
        for param_name, param in signature.parameters.items():
            # Skip self parameter
            if param_name == 'self':
                continue
            
            # Skip parameters with default values
            if param.default is not inspect.Parameter.empty:
                continue
            
            # Get the parameter type
            param_type = type_hints.get(param_name, Any)
            
            # Try to resolve the parameter
            try:
                kwargs[param_name] = self.get_service(param_type)
            except KeyError:
                self._logger.warning(f"Could not resolve parameter {param_name} of type {param_type}")
                # Let it fail when the function is called
        
        return kwargs
    
    async def _resolve_dependencies_async(self, func: Callable) -> Dict[str, Any]:
        """
        Resolve dependencies for a function asynchronously.
        
        Args:
            func: The function to resolve dependencies for
            
        Returns:
            A dictionary of parameter names to resolved services
        """
        if func is object.__init__:
            return {}
        
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        kwargs = {}
        for param_name, param in signature.parameters.items():
            # Skip self parameter
            if param_name == 'self':
                continue
            
            # Skip parameters with default values
            if param.default is not inspect.Parameter.empty:
                continue
            
            # Get the parameter type
            param_type = type_hints.get(param_name, Any)
            
            # Try to resolve the parameter
            try:
                kwargs[param_name] = await self.get_service_async(param_type)
            except KeyError:
                self._logger.warning(f"Could not resolve parameter {param_name} of type {param_type}")
                # Let it fail when the function is called
        
        return kwargs


class ServiceContainer:
    """
    Service container for dependency injection.
    
    This class is used to register and resolve services in a dependency injection container.
    """
    
    def __init__(self):
        """Initialize the service container."""
        self._descriptors: Dict[Type, ServiceDescriptor] = {}
        self._logger = logging.getLogger(__name__)
    
    def register(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type] = None,
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON
    ) -> None:
        """
        Register a service with its implementation.
        
        Args:
            service_type: The type of service to register
            implementation_type: The implementation type (defaults to service_type)
            lifetime: The service lifetime
        """
        if implementation_type is None:
            implementation_type = service_type
        
        self._descriptors[service_type] = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            lifetime=lifetime
        )
        
        self._logger.debug(f"Registered {service_type.__name__} with implementation {implementation_type.__name__}")
    
    def register_factory(
        self,
        service_type: Type[T],
        factory: Callable[..., T],
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON
    ) -> None:
        """
        Register a service with a factory function.
        
        Args:
            service_type: The type of service to register
            factory: Factory function to create the service
            lifetime: The service lifetime
        """
        self._descriptors[service_type] = ServiceDescriptor(
            service_type=service_type,
            factory=factory,
            lifetime=lifetime
        )
        
        self._logger.debug(f"Registered {service_type.__name__} with factory function")
    
    def register_instance(self, service_type: Type[T], instance: T) -> None:
        """
        Register an existing instance.
        
        Args:
            service_type: The type of service to register
            instance: The instance to register
        """
        self._descriptors[service_type] = ServiceDescriptor(
            service_type=service_type,
            instance=instance,
            lifetime=ServiceLifetime.SINGLETON
        )
        
        self._logger.debug(f"Registered {service_type.__name__} with existing instance")
    
    def get_descriptor(self, service_type: Type[T]) -> ServiceDescriptor[T]:
        """
        Get the service descriptor for a type.
        
        Args:
            service_type: The type of service to get
            
        Returns:
            The service descriptor
            
        Raises:
            KeyError: If the service type is not registered
        """
        if service_type not in self._descriptors:
            raise KeyError(f"Service {service_type.__name__} is not registered")
        
        return cast(ServiceDescriptor[T], self._descriptors[service_type])
    
    def create_provider(self) -> ServiceProvider:
        """
        Create a service provider.
        
        Returns:
            A new service provider
        """
        return ServiceProvider(self)
    
    def get_service(self, service_type: Type[T]) -> T:
        """
        Get a service of the specified type.
        
        Args:
            service_type: The type of service to get
            
        Returns:
            An instance of the service
            
        Raises:
            KeyError: If the service type is not registered
        """
        provider = self.create_provider()
        return provider.get_service(service_type)
    
    async def get_service_async(self, service_type: Type[T]) -> T:
        """
        Get a service of the specified type asynchronously.
        
        Args:
            service_type: The type of service to get
            
        Returns:
            An instance of the service
            
        Raises:
            KeyError: If the service type is not registered
        """
        provider = self.create_provider()
        return await provider.get_service_async(service_type)
