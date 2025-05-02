"""
Circuit breaker implementation for service resilience.
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
    failure_threshold: int = 5  # Number of failures before opening
    reset_timeout_seconds: int = 60  # Time before attempting reset
    half_open_max_calls: int = 3  # Max calls to test in half-open state
    monitoring_window_seconds: int = 120  # Window for failure counting
    min_throughput: int = 10  # Minimum calls before triggering

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
        """Handle successful call."""
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
