"""
Resilience Module

This module provides combined resilience patterns for service calls.
It integrates circuit breaker, retry, bulkhead, and timeout patterns.
"""

import logging
import asyncio
from typing import Callable, Any, Optional, Dict, TypeVar, Generic, Union, List, Type
from functools import wraps

from common_lib.errors import ServiceError, ErrorCode
from common_lib.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from common_lib.resilience.retry import RetryPolicy
from common_lib.resilience.bulkhead import Bulkhead
from common_lib.resilience.timeout import Timeout


# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')


class ResilienceConfig:
    """Configuration for combined resilience patterns."""

    def __init__(
        self,
        service_name: str,
        operation_name: str,
        # Circuit breaker config
        enable_circuit_breaker: bool = True,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        # Retry config
        enable_retry: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff: float = 2.0,
        # Bulkhead config
        enable_bulkhead: bool = True,
        max_concurrent_calls: int = 10,
        max_queue_size: int = 10,
        # Timeout config
        enable_timeout: bool = True,
        timeout: float = 10.0,
        # General config
        expected_exceptions: Optional[List[Type[Exception]]] = None
    ):
        """
        Initialize the resilience configuration.

        Args:
            service_name: Name of the service
            operation_name: Name of the operation
            enable_circuit_breaker: Whether to enable circuit breaker
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Time in seconds before trying to close the circuit
            enable_retry: Whether to enable retry
            max_retries: Maximum number of retries
            retry_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            backoff: Backoff factor for retry delay
            enable_bulkhead: Whether to enable bulkhead
            max_concurrent_calls: Maximum number of concurrent calls
            max_queue_size: Maximum size of the queue for waiting calls
            enable_timeout: Whether to enable timeout
            timeout: Timeout in seconds
            expected_exceptions: Exceptions that should be handled by resilience patterns
        """
        self.service_name = service_name
        self.operation_name = operation_name

        # Circuit breaker config
        self.enable_circuit_breaker = enable_circuit_breaker
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        # Retry config
        self.enable_retry = enable_retry
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_delay = max_delay
        self.backoff = backoff

        # Bulkhead config
        self.enable_bulkhead = enable_bulkhead
        self.max_concurrent_calls = max_concurrent_calls
        self.max_queue_size = max_queue_size

        # Timeout config
        self.enable_timeout = enable_timeout
        self.timeout = timeout

        # General config
        self.expected_exceptions = expected_exceptions or [Exception]


class Resilience:
    """
    Combined resilience patterns for service calls.

    This class integrates circuit breaker, retry, bulkhead, and timeout patterns
    to provide comprehensive resilience for service calls.
    """

    def __init__(
        self,
        config: ResilienceConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the resilience functionality.

        Args:
            config: Configuration for resilience patterns
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Create resilience components
        self.circuit_breaker = None
        self.retry_policy = None
        self.bulkhead = None
        self.timeout = None

        if self.config.enable_circuit_breaker:
            circuit_breaker_config = CircuitBreakerConfig(
                name=f"{config.service_name}.{config.operation_name}",
                failure_threshold=config.failure_threshold,
                recovery_timeout=config.recovery_timeout,
                expected_exception_names=[exc.__name__ for exc in config.expected_exceptions]
            )
            self.circuit_breaker = CircuitBreaker(
                config=circuit_breaker_config,
                logger=self.logger
            )

        if self.config.enable_retry:
            self.retry_policy = RetryPolicy(
                retries=config.max_retries,
                delay=config.retry_delay,
                max_delay=config.max_delay,
                backoff=config.backoff,
                exceptions=config.expected_exceptions,
                logger=self.logger
            )

        if self.config.enable_bulkhead:
            self.bulkhead = Bulkhead(
                f"{config.service_name}.{config.operation_name}",
                max_concurrent_calls=config.max_concurrent_calls,
                max_queue_size=config.max_queue_size,
                logger=self.logger
            )

        if self.config.enable_timeout:
            self.timeout = Timeout(
                config.timeout,
                operation=f"{config.service_name}.{config.operation_name}",
                logger=self.logger
            )

    async def execute_async(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute an async function with resilience patterns.

        Args:
            func: The async function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            The result of the function

        Raises:
            ServiceError: If the operation fails with resilience errors
            Exception: Any exception raised by the function
        """
        # Create a wrapped function that applies all resilience patterns
        async def execute_with_resilience():
            """
            Execute with resilience.
            
            """

            # Apply timeout if enabled
            if self.timeout:
                return await self.timeout.execute(lambda: func(*args, **kwargs))
            else:
                return await func(*args, **kwargs)

        # Apply bulkhead if enabled
        if self.bulkhead:
            execute_with_resilience = lambda: self.bulkhead.execute(execute_with_resilience)

        # Apply circuit breaker if enabled
        if self.circuit_breaker:
            execute_with_resilience = lambda: self.circuit_breaker.execute(execute_with_resilience)

        # Apply retry if enabled
        if self.retry_policy:
            return await self.retry_policy.execute(execute_with_resilience)
        else:
            return await execute_with_resilience()

    def async_call(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Decorator for async functions to apply resilience patterns.

        Args:
            func: The async function to decorate

        Returns:
            The decorated async function
        """
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            """
            Wrapper.
            
            Args:
                args: Description of args
                kwargs: Description of kwargs
            
            Returns:
                Any: Description of return value
            
            """

            return await self.execute_async(func, *args, **kwargs)
        return wrapper


# Resilience registry
_resilience_instances: Dict[str, Resilience] = {}


def get_resilience(
    service_name: str,
    operation_name: str,
    config: Optional[ResilienceConfig] = None,
    logger: Optional[logging.Logger] = None
) -> Resilience:
    """
    Get a resilience instance.

    Args:
        service_name: Name of the service
        operation_name: Name of the operation
        config: Configuration for resilience patterns
        logger: Logger instance

    Returns:
        Resilience instance
    """
    key = f"{service_name}.{operation_name}"
    if key not in _resilience_instances:
        if not config:
            config = ResilienceConfig(service_name, operation_name)
        _resilience_instances[key] = Resilience(config, logger)
    return _resilience_instances[key]


def resilient(
    service_name: str,
    operation_name: str,
    # Circuit breaker config
    enable_circuit_breaker: bool = True,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    # Retry config
    enable_retry: bool = True,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff: float = 2.0,
    # Bulkhead config
    enable_bulkhead: bool = True,
    max_concurrent_calls: int = 10,
    max_queue_size: int = 10,
    # Timeout config
    enable_timeout: bool = True,
    timeout: float = 10.0,
    # General config
    expected_exceptions: Optional[List[Type[Exception]]] = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for async functions to apply resilience patterns.

    Args:
        service_name: Name of the service
        operation_name: Name of the operation
        enable_circuit_breaker: Whether to enable circuit breaker
        failure_threshold: Number of failures before opening the circuit
        recovery_timeout: Time in seconds before trying to close the circuit
        enable_retry: Whether to enable retry
        max_retries: Maximum number of retries
        retry_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff: Backoff factor for retry delay
        enable_bulkhead: Whether to enable bulkhead
        max_concurrent_calls: Maximum number of concurrent calls
        max_queue_size: Maximum size of the queue for waiting calls
        enable_timeout: Whether to enable timeout
        timeout: Timeout in seconds
        expected_exceptions: Exceptions that should be handled by resilience patterns

    Returns:
        Decorator function
    """
    config = ResilienceConfig(
        service_name=service_name,
        operation_name=operation_name,
        enable_circuit_breaker=enable_circuit_breaker,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        enable_retry=enable_retry,
        max_retries=max_retries,
        retry_delay=retry_delay,
        max_delay=max_delay,
        backoff=backoff,
        enable_bulkhead=enable_bulkhead,
        max_concurrent_calls=max_concurrent_calls,
        max_queue_size=max_queue_size,
        enable_timeout=enable_timeout,
        timeout=timeout,
        expected_exceptions=expected_exceptions
    )
    resilience = get_resilience(service_name, operation_name, config)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Decorator.
        
        Args:
            func: Description of func
            Any]: Description of Any]
        
        Returns:
            Callable[..., Any]: Description of return value
        
        """

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            """
            Wrapper.
            
            Args:
                args: Description of args
                kwargs: Description of kwargs
            
            Returns:
                Any: Description of return value
            
            """

            return await resilience.execute_async(func, *args, **kwargs)
        return wrapper

    return decorator
