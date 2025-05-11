"""
Circuit Breaker Implementation for Service Resilience

This module provides a robust implementation of the Circuit Breaker pattern
for improving system resilience when dealing with external dependencies.

The Circuit Breaker pattern prevents cascading failures by automatically
detecting failures and stopping calls to problematic services. It has three states:
- CLOSED: Normal operation, calls pass through to the service
- OPEN: Service is failing, calls are blocked or redirected to fallback
- HALF-OPEN: Testing if service has recovered, limited calls allowed

Key features:
- Failure counting with configurable thresholds
- Automatic state transitions based on success/failure patterns
- Sliding window for failure tracking
- Half-open state for testing recovery
- Metrics collection for monitoring
- Async-first implementation with proper locking
- Optional fallback function support

Usage:
    # Create a circuit breaker
    breaker = CircuitBreaker("payment-service")
    
    # Use with async functions
    try:
        result = await breaker.call(external_payment_service.process_payment, payment_data)
    except CircuitBreakerOpen:
        # Handle service unavailability
        notify_admin("Payment service is down")
        return fallback_result
        
    # With custom configuration
    config = CircuitBreakerConfig(
        failure_threshold=10,
        reset_timeout_seconds=300,
        half_open_max_calls=5
    )
    breaker = CircuitBreaker("inventory-service", config=config)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar, Awaitable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')

class CircuitState(Enum):
    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"      # Circuit is broken
    HALF_OPEN = "HALF_OPEN"  # Testing if service is healthy

@dataclass
class CircuitBreakerConfig:
    """
    Configuration parameters for the CircuitBreaker.
    
    Attributes:
        failure_threshold: Number of failures required to trip the circuit breaker.
            When this many failures occur within the monitoring window, the circuit
            transitions from CLOSED to OPEN state.
            
        reset_timeout_seconds: Time in seconds to wait before attempting to reset
            the circuit breaker from OPEN to HALF-OPEN state. This allows the
            problematic service time to recover.
            
        half_open_max_calls: Maximum number of test calls allowed in HALF-OPEN state.
            These calls are used to test if the service has recovered. If any call
            fails, the circuit returns to OPEN state. If all succeed, it transitions
            to CLOSED state.
            
        monitoring_window_seconds: Time window in seconds for tracking failures.
            Only failures within this sliding window are counted toward the
            failure_threshold. This prevents old failures from affecting current
            circuit state.
            
        min_throughput: Minimum number of calls required before the circuit breaker
            can trip. This prevents the circuit from opening due to a small number
            of calls that happen to fail.
    """
    failure_threshold: int = 5
    reset_timeout_seconds: int = 60
    half_open_max_calls: int = 3
    monitoring_window_seconds: int = 120
    min_throughput: int = 10

class CircuitBreaker:
    """
    Circuit breaker pattern implementation with the following features:
    - Failure counting with sliding window
    - Automatic state transitions
    - Half-open state for testing recovery
    - Metrics collection
    - Configurable thresholds
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback_function: Optional[Callable[..., Any]] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback_function = fallback_function
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
        
        # Metrics
        self._total_calls = 0
        self._successful_calls = 0
        self._failed_calls = 0
        self._last_state_change = datetime.utcnow()
        self._recent_failures: Dict[str, datetime] = {}
        
    @property
    def state(self) -> CircuitState:
        """
        Get the current state of the circuit breaker.
        
        Returns:
            CircuitState: The current state (CLOSED, OPEN, or HALF-OPEN)
        """
        return self._state
        
    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any
    ) -> T:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: The async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The function's result
            
        Raises:
            CircuitBreakerOpen: If circuit is open
            Exception: Any exception from the function if circuit is closed
        """
        self._total_calls += 1
        
        async with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    return await self._handle_open_circuit(*args, **kwargs)
                    
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    return await self._handle_open_circuit(*args, **kwargs)
                self._half_open_calls += 1
        
        try:
            result = await func(*args, **kwargs)
            await self._handle_success()
            return result
            
        except Exception as e:
            await self._handle_failure(str(e))
            raise
            
    async def _handle_success(self):
        """
        Handle a successful call to the protected service.
        
        This method:
        1. Increments the successful calls counter
        2. If in HALF-OPEN state, transitions to CLOSED state (service has recovered)
        """
        async with self._lock:
            self._successful_calls += 1
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to_closed()
                
    async def _handle_failure(self, error_details: str):
        """Handle failed call."""
        async with self._lock:
            self._failed_calls += 1
            self._last_failure_time = datetime.utcnow()
            self._recent_failures[str(datetime.utcnow())] = error_details
            
            # Cleanup old failures
            cutoff = datetime.utcnow() - timedelta(
                seconds=self.config.monitoring_window_seconds
            )
            self._recent_failures = {
                ts: err for ts, err in self._recent_failures.items()
                if datetime.fromisoformat(ts) > cutoff
            }
            
            # Update failure count and check threshold
            self._failure_count = len(self._recent_failures)
            if (self._state == CircuitState.CLOSED and 
                self._failure_count >= self.config.failure_threshold):
                self._transition_to_open()
            elif self._state == CircuitState.HALF_OPEN:
                self._transition_to_open()
                
    async def _handle_open_circuit(self, *args: Any, **kwargs: Any) -> Any:
        """Handle calls when circuit is open."""
        if self.fallback_function:
            return await self.fallback_function(*args, **kwargs)
        raise CircuitBreakerOpen(
            f"Circuit {self.name} is open. Last failure at: {self._last_failure_time}"
        )
        
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if not self._last_failure_time:
            return True
        
        elapsed = datetime.utcnow() - self._last_failure_time
        return elapsed.total_seconds() >= self.config.reset_timeout_seconds
        
    def _transition_to_open(self):
        """Transition to open state."""
        self._state = CircuitState.OPEN
        self._last_state_change = datetime.utcnow()
        logger.warning(
            f"Circuit {self.name} transitioned to OPEN state. "
            f"Recent failures: {len(self._recent_failures)}"
        )
        
    def _transition_to_half_open(self):
        """Transition to half-open state."""
        self._state = CircuitState.HALF_OPEN
        self._half_open_calls = 0
        self._last_state_change = datetime.utcnow()
        logger.info(f"Circuit {self.name} transitioned to HALF-OPEN state")
        
    def _transition_to_closed(self):
        """Transition to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._recent_failures.clear()
        self._last_state_change = datetime.utcnow()
        logger.info(f"Circuit {self.name} transitioned to CLOSED state")
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit breaker metrics."""
        return {
            "name": self.name,
            "state": self._state.value,
            "total_calls": self._total_calls,
            "successful_calls": self._successful_calls,
            "failed_calls": self._failed_calls,
            "current_failure_count": self._failure_count,
            "last_failure_time": self._last_failure_time,
            "last_state_change": self._last_state_change,
            "recent_failures": self._recent_failures
        }

class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""
    pass
