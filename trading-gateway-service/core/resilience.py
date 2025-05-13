"""
Resilience patterns for the Trading Gateway Service.

This module provides decorators and utilities for implementing resilience patterns
such as circuit breakers, retries, and timeouts for broker API calls.
"""

import asyncio
import functools
import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar, cast

from core.exceptions_bridge_1 import (
    BrokerConnectionError,
    ForexTradingPlatformError,
    ServiceError,
    async_with_exception_handling
)

logger = logging.getLogger(__name__)

# Type variables for function signatures
F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Any])


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"  # Normal operation, requests pass through
    OPEN = "OPEN"      # Failing, requests are blocked
    HALF_OPEN = "HALF_OPEN"  # Testing if service is back online


class CircuitBreaker:
    """
    Circuit breaker implementation for broker API calls.
    
    Tracks failures and prevents calls to failing services to allow them time to recover.
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        timeout: float = 10.0
    ):
        """
        Initialize the circuit breaker.
        
        Args:
            name: Name of the circuit breaker (usually the API endpoint)
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Time in seconds to wait before trying again
            timeout: Timeout for API calls in seconds
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.timeout = timeout
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.last_success_time = time.time()
    
    async def execute(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
            
        Raises:
            BrokerConnectionError: If the circuit is open
            Any exception raised by the function
        """
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                logger.info(f"Circuit {self.name} transitioning from OPEN to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
            else:
                logger.warning(f"Circuit {self.name} is OPEN, rejecting request")
                raise BrokerConnectionError(f"Circuit {self.name} is open")
        
        # Execute the function with timeout
        try:
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.timeout)
            else:
                # For synchronous functions, run in executor with timeout
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, functools.partial(func, *args, **kwargs)),
                    timeout=self.timeout
                )
            
            # Success handling
            self._handle_success()
            return result
            
        except Exception as e:
            # Failure handling
            self._handle_failure(e)
            raise
    
    def _handle_success(self) -> None:
        """Handle successful execution."""
        self.last_success_time = time.time()
        if self.state == CircuitState.HALF_OPEN:
            logger.info(f"Circuit {self.name} recovered, transitioning to CLOSED")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
    
    def _handle_failure(self, exception: Exception) -> None:
        """
        Handle execution failure.
        
        Args:
            exception: The exception that occurred
        """
        self.last_failure_time = time.time()
        
        # Only count certain exceptions toward failure threshold
        if isinstance(exception, (BrokerConnectionError, asyncio.TimeoutError)):
            self.failure_count += 1
            logger.warning(f"Circuit {self.name} recorded failure {self.failure_count}/{self.failure_threshold}")
            
            if self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
                logger.error(f"Circuit {self.name} tripped, transitioning to OPEN")
                self.state = CircuitState.OPEN
            elif self.state == CircuitState.HALF_OPEN:
                logger.warning(f"Circuit {self.name} failed in HALF_OPEN state, returning to OPEN")
                self.state = CircuitState.OPEN


# Global registry of circuit breakers
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str) -> CircuitBreaker:
    """
    Get or create a circuit breaker by name.
    
    Args:
        name: Name of the circuit breaker
        
    Returns:
        CircuitBreaker instance
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name)
    return _circuit_breakers[name]


def with_broker_api_resilience(endpoint_name: str) -> Callable[[F], F]:
    """
    Decorator to add resilience patterns to broker API calls.
    
    Adds circuit breaking, retries, and timeout handling.
    
    Args:
        endpoint_name: Name of the API endpoint
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            circuit = get_circuit_breaker(endpoint_name)
            max_retries = 3
            retry_count = 0
            
            while True:
                try:
                    return await circuit.execute(func, *args, **kwargs)
                except (BrokerConnectionError, asyncio.TimeoutError) as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {endpoint_name}")
                        raise
                    
                    # Exponential backoff
                    wait_time = 0.5 * (2 ** retry_count)
                    logger.warning(f"Retrying {endpoint_name} after {wait_time}s (attempt {retry_count}/{max_retries})")
                    await asyncio.sleep(wait_time)
        
        return cast(F, wrapper)
    
    return decorator
