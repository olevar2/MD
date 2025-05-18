"""
Circuit Breaker Module

This module provides a circuit breaker implementation for resilience.
"""

import logging
import time
import asyncio
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Awaitable, Type, TypeVar, Generic, Union

from pydantic import BaseModel, Field

from common_lib.errors import ServiceError


T = TypeVar('T')


class CircuitState(Enum):
    """
    Circuit breaker state.
    """

    CLOSED = "CLOSED"  # Circuit is closed, requests are allowed
    OPEN = "OPEN"  # Circuit is open, requests are not allowed
    HALF_OPEN = "HALF_OPEN"  # Circuit is half-open, limited requests are allowed


class CircuitBreakerConfig(BaseModel):
    """
    Configuration for circuit breaker.
    """

    name: str = Field(..., description="Name of the circuit breaker")
    failure_threshold: int = Field(5, description="Number of failures before opening the circuit")
    recovery_timeout: float = Field(60.0, description="Time in seconds before attempting to close the circuit")
    expected_exception_names: List[str] = Field(
        ["Exception"],
        description="List of exception class names that should be counted as failures"
    )


class CircuitBreaker:
    """
    Circuit breaker for resilience.

    This class implements the circuit breaker pattern to prevent cascading failures.
    """

    def __init__(
        self,
        config: CircuitBreakerConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the circuit breaker.

        Args:
            config: Circuit breaker configuration
            logger: Logger to use (if None, creates a new logger)
        """
        self.name = config.name
        self.failure_threshold = config.failure_threshold
        self.recovery_timeout = config.recovery_timeout

        # Convert exception names to actual exception classes
        self.expected_exceptions = []
        for exc_name in config.expected_exception_names:
            try:
                # Try to get the exception class from builtins
                exc_class = getattr(__import__('builtins'), exc_name)
                if issubclass(exc_class, Exception):
                    self.expected_exceptions.append(exc_class)
            except (AttributeError, ImportError):
                # If not found in builtins, try to get it from common_lib.errors
                try:
                    exc_class = getattr(__import__('common_lib.errors', fromlist=[exc_name]), exc_name)
                    if issubclass(exc_class, Exception):
                        self.expected_exceptions.append(exc_class)
                except (AttributeError, ImportError):
                    # If not found, log a warning
                    logging.warning(f"Exception class {exc_name} not found")

        # If no valid exceptions were found, use Exception as default
        if not self.expected_exceptions:
            self.expected_exceptions = [Exception]

        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.lock = asyncio.Lock()

    async def execute(self, func: Callable[[], Awaitable[T]]) -> T:
        """
        Execute a function with circuit breaker protection.

        Args:
            func: Function to execute

        Returns:
            Result of the function

        Raises:
            ServiceError: If the circuit is open
            Exception: If the function raises an exception
        """
        async with self.lock:
            # Check if circuit is open
            if self.state == CircuitState.OPEN:
                # Check if recovery timeout has elapsed
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.logger.info(f"Circuit {self.name} is half-open, allowing request")
                    self.state = CircuitState.HALF_OPEN
                else:
                    self.logger.warning(f"Circuit {self.name} is open, rejecting request")
                    raise ServiceError(
                        error_code=2000,
                        message=f"Service {self.name} is unavailable",
                        details=f"Circuit breaker is open"
                    )

        try:
            # Execute function
            result = await func()

            # Reset circuit if successful
            async with self.lock:
                if self.state == CircuitState.HALF_OPEN:
                    self.logger.info(f"Circuit {self.name} is closed")
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0

            return result
        except Exception as e:
            # Check if exception is expected
            is_expected = any(isinstance(e, exc) for exc in self.expected_exceptions)

            if is_expected:
                # Increment failure count
                async with self.lock:
                    self.failure_count += 1
                    self.last_failure_time = time.time()

                    # Open circuit if threshold is reached
                    if (self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold) or self.state == CircuitState.HALF_OPEN:
                        self.logger.warning(f"Circuit {self.name} is open")
                        self.state = CircuitState.OPEN

            # Re-raise exception
            raise


class CircuitBreakerOpen(ServiceError):
    """
    Exception raised when the circuit breaker is open and a request is attempted.
    """
    def __init__(self, service_name: str, resource_name: str):
        """
        Initialize the exception.

        Args:
            service_name: The name of the service.
            resource_name: The name of the resource.
        """
        super().__init__(
            error_code=2001, # Define a specific error code for circuit breaker open
            message=f"Circuit breaker for {service_name}/{resource_name} is open",
            details="Requests are currently being rejected due to excessive failures."
        )
        self.service_name = service_name
        self.resource_name = resource_name


__all__ = [
    'CircuitState',
    'CircuitBreakerConfig',
    'CircuitBreaker',
    'create_circuit_breaker',
    'CircuitBreakerOpen'
]