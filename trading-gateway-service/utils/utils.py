"""
Resilience utilities for Trading Gateway Service.

This module provides utility functions for applying resilience patterns
in the Trading Gateway Service.
"""

import logging
import functools
import asyncio
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, Coroutine

from common_lib.resilience import (
    create_circuit_breaker, CircuitBreaker, CircuitBreakerConfig,
    retry_with_policy, bulkhead, timeout_handler
)

from trading_gateway_service.resilience import (
    get_circuit_breaker_config,
    get_retry_config,
    get_bulkhead_config,
    get_timeout_config
)

# Type variables for function return types
T = TypeVar('T')
R = TypeVar('R')

# Initialize logger
logger = logging.getLogger(__name__)

# Dictionary to store circuit breakers
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(service_name: str, resource_name: str, service_type: str = "default") -> CircuitBreaker:
    """
    Get or create a circuit breaker for a specific service and resource.

    Args:
        service_name: Name of the service using the circuit breaker
        resource_name: Name of the resource being protected
        service_type: Type of service (broker_api, market_data, etc.)

    Returns:
        CircuitBreaker instance
    """
    key = f"{service_name}.{resource_name}"
    if key not in _circuit_breakers:
        config = get_circuit_breaker_config(service_type)
        _circuit_breakers[key] = create_circuit_breaker(
            service_name=service_name,
            resource_name=resource_name,
            config=config
        )
    return _circuit_breakers[key]


def with_resilience(
    service_name: str,
    operation_name: str,
    service_type: str = "default",
    use_circuit_breaker: bool = True,
    use_retry: bool = True,
    use_bulkhead: bool = True,
    use_timeout: bool = True,
    exceptions: Optional[List[Type[Exception]]] = None
) -> Callable[[Callable[..., Union[T, Coroutine[Any, Any, R]]]],
             Callable[..., Union[T, Coroutine[Any, Any, R]]]]:
    """
    Decorator that applies multiple resilience patterns to a function.

    Args:
        service_name: Name of the service
        operation_name: Name of the operation
        service_type: Type of service (broker_api, market_data, etc.)
        use_circuit_breaker: Whether to use circuit breaker
        use_retry: Whether to use retry policy
        use_bulkhead: Whether to use bulkhead
        use_timeout: Whether to use timeout handler
        exceptions: List of exceptions to retry on (None for default)

    Returns:
        Decorated function with resilience patterns applied
    """
    def decorator(func: Callable[..., Union[T, Coroutine[Any, Any, R]]]) -> Callable[..., Union[T, Coroutine[Any, Any, R]]]:
    """
    Decorator.
    
    Args:
        func: Description of func
        Union[T: Description of Union[T
        Coroutine[Any: Description of Coroutine[Any
        Any: Description of Any
        R]]]: Description of R]]]
    
    Returns:
        Callable[..., Union[T, Coroutine[Any, Any, R]]]: Description of return value
    
    """

        # Start with the original function
        result = func

        # Apply timeout handler
        if use_timeout:
            timeout = get_timeout_config(service_type)
            result = timeout_handler(timeout_seconds=timeout)(result)

        # Apply bulkhead
        if use_bulkhead:
            bulkhead_config = get_bulkhead_config(service_type)
            result = bulkhead(
                name=f"{service_name}.{operation_name}",
                max_concurrent=bulkhead_config["max_concurrent"],
                max_waiting=bulkhead_config["max_waiting"]
            )(result)

        # Apply retry policy
        if use_retry:
            retry_config = get_retry_config(service_type)
            result = retry_with_policy(
                max_attempts=retry_config["max_attempts"],
                base_delay=retry_config["base_delay"],
                max_delay=retry_config["max_delay"],
                backoff_factor=retry_config["backoff_factor"],
                jitter=retry_config["jitter"],
                exceptions=exceptions,
                service_name=service_name,
                operation_name=operation_name
            )(result)

        # Apply circuit breaker
        if use_circuit_breaker:
            # Get or create circuit breaker
            cb = get_circuit_breaker(service_name, operation_name, service_type)

            # Check if the function is async
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def circuit_breaker_wrapper(*args, **kwargs):
    """
    Circuit breaker wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

                    return await cb.execute(lambda: func(*args, **kwargs))
                return circuit_breaker_wrapper
            else:
                @functools.wraps(func)
                def circuit_breaker_wrapper(*args, **kwargs):
    """
    Circuit breaker wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

                    return cb.execute(lambda: func(*args, **kwargs))
                return circuit_breaker_wrapper

        return result

    return decorator


def with_broker_api_resilience(
    operation_name: str,
    exceptions: Optional[List[Type[Exception]]] = None
) -> Callable[[Callable[..., Union[T, Coroutine[Any, Any, R]]]],
             Callable[..., Union[T, Coroutine[Any, Any, R]]]]:
    """
    Decorator that applies resilience patterns for broker API operations.

    Args:
        operation_name: Name of the operation
        exceptions: List of exceptions to retry on (None for default)

    Returns:
        Decorated function with resilience patterns applied
    """
    return with_resilience(
        service_name="trading_gateway",
        operation_name=operation_name,
        service_type="broker_api",
        exceptions=exceptions
    )


def with_market_data_resilience(
    operation_name: str,
    exceptions: Optional[List[Type[Exception]]] = None
) -> Callable[[Callable[..., Union[T, Coroutine[Any, Any, R]]]],
             Callable[..., Union[T, Coroutine[Any, Any, R]]]]:
    """
    Decorator that applies resilience patterns for market data operations.

    Args:
        operation_name: Name of the operation
        exceptions: List of exceptions to retry on (None for default)

    Returns:
        Decorated function with resilience patterns applied
    """
    return with_resilience(
        service_name="trading_gateway",
        operation_name=operation_name,
        service_type="market_data",
        exceptions=exceptions
    )


def with_order_execution_resilience(
    operation_name: str,
    exceptions: Optional[List[Type[Exception]]] = None
) -> Callable[[Callable[..., Union[T, Coroutine[Any, Any, R]]]],
             Callable[..., Union[T, Coroutine[Any, Any, R]]]]:
    """
    Decorator that applies resilience patterns for order execution operations.

    Args:
        operation_name: Name of the operation
        exceptions: List of exceptions to retry on (None for default)

    Returns:
        Decorated function with resilience patterns applied
    """
    return with_resilience(
        service_name="trading_gateway",
        operation_name=operation_name,
        service_type="order_execution",
        exceptions=exceptions
    )


def with_risk_management_resilience(
    operation_name: str,
    exceptions: Optional[List[Type[Exception]]] = None
) -> Callable[[Callable[..., Union[T, Coroutine[Any, Any, R]]]],
             Callable[..., Union[T, Coroutine[Any, Any, R]]]]:
    """
    Decorator that applies resilience patterns for risk management operations.

    Args:
        operation_name: Name of the operation
        exceptions: List of exceptions to retry on (None for default)

    Returns:
        Decorated function with resilience patterns applied
    """
    return with_resilience(
        service_name="trading_gateway",
        operation_name=operation_name,
        service_type="risk_management",
        exceptions=exceptions
    )


def with_database_resilience(
    operation_name: str,
    exceptions: Optional[List[Type[Exception]]] = None
) -> Callable[[Callable[..., Union[T, Coroutine[Any, Any, R]]]],
             Callable[..., Union[T, Coroutine[Any, Any, R]]]]:
    """
    Decorator that applies resilience patterns for database operations.

    Args:
        operation_name: Name of the operation
        exceptions: List of exceptions to retry on (None for default)

    Returns:
        Decorated function with resilience patterns applied
    """
    return with_resilience(
        service_name="trading_gateway",
        operation_name=operation_name,
        service_type="database",
        exceptions=exceptions or register_database_retryable_exceptions()
    )
