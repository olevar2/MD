"""
Resilience Decorators Module

This module provides standardized decorators for applying resilience patterns to functions.
These decorators ensure consistent resilience patterns across the platform.
"""

import asyncio
import functools
import logging
import time
import random
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast

from common_lib.errors import ServiceError, TimeoutError, BaseError
from common_lib.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from common_lib.resilience.retry import RetryPolicy
from common_lib.resilience.bulkhead import Bulkhead
from common_lib.resilience.timeout import Timeout

# Type variables for generic functions
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

# Get logger
logger = logging.getLogger(__name__)


def circuit_breaker(
    service_name: Optional[str] = None,
    resource_name: Optional[str] = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    expected_exceptions: Optional[List[Type[Exception]]] = None
) -> Callable[[F], F]:
    """
    Decorator to apply circuit breaker pattern to a function.

    Args:
        service_name: Name of the service
        resource_name: Name of the resource
        failure_threshold: Number of failures before opening the circuit
        recovery_timeout: Time in seconds before trying again
        expected_exceptions: List of exceptions that should be considered as failures

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        """
        Decorator.
        
        Args:
            func: Description of func
        
        Returns:
            F: Description of return value
        
        """

        # Generate service and resource names if not provided
        nonlocal service_name, resource_name
        if service_name is None:
            service_name = func.__module__.split('.')[0]
        if resource_name is None:
            resource_name = func.__name__

        # Create circuit breaker name
        cb_name = f"{service_name}.{resource_name}"

        # Convert expected exceptions to names for CircuitBreakerConfig
        expected_exception_names = []
        if expected_exceptions:
            expected_exception_names = [exc.__name__ for exc in expected_exceptions]

        # Create circuit breaker
        cb_config = CircuitBreakerConfig(
            name=cb_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception_names=expected_exception_names or ["Exception"]
        )
        cb = CircuitBreaker(
            config=cb_config
        )

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            """
            Async wrapper.
            
            Args:
                args: Description of args
                kwargs: Description of kwargs
            
            Returns:
                Any: Description of return value
            
            """

            async def awaitable_func():
                """
                Awaitable func.
                
                """

                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            return await cb.execute(awaitable_func)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            """
            Sync wrapper.
            
            Args:
                args: Description of args
                kwargs: Description of kwargs
            
            Returns:
                Any: Description of return value
            
            """

            # For synchronous functions, we need to run in an event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            async def awaitable_func():
                """
                Awaitable func.
                
                """

                return func(*args, **kwargs)

            try:
                return loop.run_until_complete(cb.execute(awaitable_func))
            finally:
                # Clean up the event loop if we created it
                if loop != asyncio.get_event_loop():
                    loop.close()

        # Return appropriate wrapper based on whether the function is async or not
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)

    return decorator


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[List[Type[Exception]]] = None
) -> Callable[[F], F]:
    """
    Decorator to apply retry pattern with exponential backoff to a function.

    Args:
        max_retries: Maximum number of retries
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Factor to increase delay with each retry
        jitter: Whether to add jitter to delay
        retryable_exceptions: List of exceptions that should be retried

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        """
        Decorator.
        
        Args:
            func: Description of func
        
        Returns:
            F: Description of return value
        
        """

        # Create retry policy
        retry_policy = RetryPolicy(
            retries=max_retries,
            delay=base_delay,
            max_delay=max_delay,
            backoff=backoff_factor,
            exceptions=retryable_exceptions
        )

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            """
            Async wrapper.
            
            Args:
                args: Description of args
                kwargs: Description of kwargs
            
            Returns:
                Any: Description of return value
            
            """

            async def awaitable_func():
                """
                Awaitable func.
                
                """

                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            return await retry_policy.execute(awaitable_func)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            """
            Sync wrapper.
            
            Args:
                args: Description of args
                kwargs: Description of kwargs
            
            Returns:
                Any: Description of return value
            
            """

            # For synchronous functions, we need to run in an event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            async def awaitable_func():
                """
                Awaitable func.
                
                """

                return func(*args, **kwargs)

            try:
                return loop.run_until_complete(retry_policy.execute(awaitable_func))
            finally:
                # Clean up the event loop if we created it
                if loop != asyncio.get_event_loop():
                    loop.close()

        # Return appropriate wrapper based on whether the function is async or not
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)

    return decorator


def timeout(
    timeout_seconds: float = 30.0,
    operation: str = None
) -> Callable[[F], F]:
    """
    Decorator to apply timeout pattern to a function.

    Args:
        timeout_seconds: Timeout in seconds
        operation: Name of the operation (defaults to function name)

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        """
        Decorator.
        
        Args:
            func: Description of func
        
        Returns:
            F: Description of return value
        
        """

        # Get operation name if not provided
        nonlocal operation
        if operation is None:
            operation = func.__name__

        # Create timeout handler
        timeout_handler = Timeout(timeout_seconds, operation=operation)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            """
            Async wrapper.
            
            Args:
                args: Description of args
                kwargs: Description of kwargs
            
            Returns:
                Any: Description of return value
            
            """

            async def awaitable_func():
                """
                Awaitable func.
                
                """

                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            return await timeout_handler.execute(awaitable_func)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            """
            Sync wrapper.
            
            Args:
                args: Description of args
                kwargs: Description of kwargs
            
            Returns:
                Any: Description of return value
            
            """

            # For synchronous functions, we need to run in an event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            async def awaitable_func():
                """
                Awaitable func.
                
                """

                return func(*args, **kwargs)

            try:
                return loop.run_until_complete(timeout_handler.execute(awaitable_func))
            finally:
                # Clean up the event loop if we created it
                if loop != asyncio.get_event_loop():
                    loop.close()

        # Return appropriate wrapper based on whether the function is async or not
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)

    return decorator


def bulkhead(
    max_concurrent: int = 10,
    max_queue: int = 10,
    name: Optional[str] = None
) -> Callable[[F], F]:
    """
    Decorator to apply bulkhead pattern to a function.

    Args:
        max_concurrent: Maximum number of concurrent executions
        max_queue: Maximum size of the queue
        name: Name of the bulkhead

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        """
        Decorator.
        
        Args:
            func: Description of func
        
        Returns:
            F: Description of return value
        
        """

        # Generate bulkhead name if not provided
        nonlocal name
        if name is None:
            name = f"{func.__module__}.{func.__name__}"

        # Create bulkhead
        bulkhead_handler = Bulkhead(
            name=name,
            max_concurrent_calls=max_concurrent,
            max_queue_size=max_queue
        )

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            """
            Async wrapper.
            
            Args:
                args: Description of args
                kwargs: Description of kwargs
            
            Returns:
                Any: Description of return value
            
            """

            async def awaitable_func():
                """
                Awaitable func.
                
                """

                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            return await bulkhead_handler.execute(awaitable_func)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            """
            Sync wrapper.
            
            Args:
                args: Description of args
                kwargs: Description of kwargs
            
            Returns:
                Any: Description of return value
            
            """

            # For synchronous functions, we need to run in an event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            async def awaitable_func():
                """
                Awaitable func.
                
                """

                return func(*args, **kwargs)

            try:
                return loop.run_until_complete(bulkhead_handler.execute(awaitable_func))
            finally:
                # Clean up the event loop if we created it
                if loop != asyncio.get_event_loop():
                    loop.close()

        # Return appropriate wrapper based on whether the function is async or not
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)

    return decorator


def with_resilience(
    # Circuit breaker config
    enable_circuit_breaker: bool = True,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    # Retry config
    enable_retry: bool = True,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    # Bulkhead config
    enable_bulkhead: bool = True,
    max_concurrent: int = 10,
    max_queue: int = 10,
    # Timeout config
    enable_timeout: bool = True,
    timeout_seconds: float = 30.0,
    # General config
    service_name: Optional[str] = None,
    resource_name: Optional[str] = None,
    expected_exceptions: Optional[List[Type[Exception]]] = None
) -> Callable[[F], F]:
    """
    Decorator to apply all resilience patterns to a function.

    Args:
        enable_circuit_breaker: Whether to enable circuit breaker
        failure_threshold: Number of failures before opening the circuit
        recovery_timeout: Time in seconds before trying again
        enable_retry: Whether to enable retry
        max_retries: Maximum number of retries
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Factor to increase delay with each retry
        jitter: Whether to add jitter to delay
        enable_bulkhead: Whether to enable bulkhead
        max_concurrent: Maximum number of concurrent executions
        max_queue: Maximum size of the queue
        enable_timeout: Whether to enable timeout
        timeout_seconds: Timeout in seconds
        service_name: Name of the service
        resource_name: Name of the resource
        expected_exceptions: List of exceptions that should be considered as failures

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        """
        Decorator.
        
        Args:
            func: Description of func
        
        Returns:
            F: Description of return value
        
        """

        # Generate service and resource names if not provided
        nonlocal service_name, resource_name
        if service_name is None:
            service_name = func.__module__.split('.')[0]
        if resource_name is None:
            resource_name = func.__name__

        # Apply decorators in order: timeout -> retry -> circuit breaker -> bulkhead
        decorated_func = func

        if enable_timeout:
            decorated_func = timeout(
                timeout_seconds=timeout_seconds,
                operation=f"{service_name}.{resource_name}"
            )(decorated_func)

        if enable_retry:
            decorated_func = retry_with_backoff(
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                backoff_factor=backoff_factor,
                jitter=jitter,
                retryable_exceptions=expected_exceptions
            )(decorated_func)

        if enable_circuit_breaker:
            decorated_func = circuit_breaker(
                service_name=service_name,
                resource_name=resource_name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                expected_exceptions=expected_exceptions
            )(decorated_func)

        if enable_bulkhead:
            decorated_func = bulkhead(
                max_concurrent=max_concurrent,
                max_queue=max_queue,
                name=f"{service_name}.{resource_name}"
            )(decorated_func)

        return decorated_func

    return decorator
