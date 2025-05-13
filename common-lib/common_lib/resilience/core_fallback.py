"""
Core Foundations Resilience stub module.

This module provides stub implementations of core resilience components
when the actual core_foundations package is not available. This allows
the common_lib resilience modules to function in a standalone mode
with reduced functionality.
"""

import enum
import logging
from typing import Any, Callable, Dict, Optional, Union, List, Type, TypeVar, Set

# Setup logger
logger = logging.getLogger(__name__)

T = TypeVar('T')

# Circuit breaker stubs
class CircuitState(enum.Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

class CircuitBreakerConfig:
    """Configuration for circuit breakers."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout_seconds: float = 60.0,
        half_open_max_calls: int = 1
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout_seconds = reset_timeout_seconds
        self.half_open_max_calls = half_open_max_calls

class CircuitBreakerOpen(Exception):
    """Exception raised when a circuit is open."""
    
    def __init__(self, name: str):
        self.name = name
        super().__init__(f"Circuit '{name}' is OPEN and rejecting calls")

class CircuitBreaker:
    """Base circuit breaker implementation."""
    
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
        logger.warning(
            "Using stub CircuitBreaker implementation. Install core_foundations "
            "package for full functionality."
        )
        
    @property
    def current_state(self) -> CircuitState:
        return self._state
        
    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        # Simplified stub implementation
        if self._state == CircuitState.OPEN:
            if self.fallback_function:
                return await self.fallback_function(*args, **kwargs)
            raise CircuitBreakerOpen(self.name)
            
        try:
            result = await func(*args, **kwargs)
            # Success resets failure count
            self._failure_count = 0
            self._state = CircuitState.CLOSED
            return result
        except Exception:
            self._failure_count += 1
            if self._failure_count >= self.config.failure_threshold:
                self._state = CircuitState.OPEN
            raise
            
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.config.failure_threshold,
            "reset_timeout_seconds": self.config.reset_timeout_seconds,
            "total_calls": 0,  # Not tracked in stub
            "failed_calls": self._failure_count
        }

# Retry policy stubs
class RetryExhaustedException(Exception):
    """Exception raised when all retry attempts have been exhausted."""
    
    def __init__(self, operation: str, attempts: int, last_exception: Exception):
        self.operation = operation
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"All {attempts} retry attempts exhausted for operation '{operation}': {last_exception}"
        )

class RetryPolicy:
    """Base retry policy implementation."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: Optional[float] = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        exceptions: Optional[List[Type[Exception]]] = None,
        on_retry: Optional[Callable[[Any], None]] = None,
        metric_handler: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.exceptions = exceptions or [Exception]
        self.on_retry = on_retry
        self.metric_handler = metric_handler
        logger.warning(
            "Using stub RetryPolicy implementation. Install core_foundations "
            "package for full functionality."
        )

def register_common_retryable_exceptions() -> Set[Type[Exception]]:
    """
    Register common retryable exceptions.
    
    Returns:
        A set of exception types that are commonly retryable
    """
    return {ConnectionError, TimeoutError}

def retry_with_policy(**kwargs) -> Callable:
    """Stub decorator for retry policy."""
    def decorator(func):
        return func  # Just return the original function in stub mode
    return decorator

# Degraded Mode stubs
class DependencyStatus(enum.Enum):
    """Status of a service dependency."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED" 
    UNAVAILABLE = "UNAVAILABLE"

class DegradedModeStrategy:
    """Strategy for handling service in degraded mode."""
    
    def __init__(self, name: str):
        self.name = name
        logger.warning(
            f"Using stub DegradedModeStrategy '{name}'. Install core_foundations "
            "package for full functionality."
        )
    
    async def execute(self, *args):
        """Execute the degraded mode strategy."""
        logger.warning("Stub degraded mode strategy '{}' called".format(self.name))
        return None

class DegradedModeManager:
    """Manager for handling service degradation."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self._dependencies = {}
        logger.warning(
            f"Using stub DegradedModeManager for service '{service_name}'. "
            "Install core_foundations package for full functionality."
        )
    def register_dependency(self, name: str, criticality: str = "standard"):
        """Register a dependency with the manager."""
        self._dependencies[name] = {"status": DependencyStatus.HEALTHY, "criticality": criticality}
        
    def set_dependency_status(self, name: str, status: DependencyStatus):
        """Set the status of a dependency."""
        if name in self._dependencies:
            self._dependencies[name]["status"] = status
            
    def get_dependency_status(self, name: str) -> DependencyStatus:
        """Get the status of a dependency."""
        if name in self._dependencies:
            return self._dependencies[name]["status"]
        return DependencyStatus.UNAVAILABLE

def with_degraded_mode(dependency_name: Optional[str] = None, strategy: Optional[DegradedModeStrategy] = None):
    """Decorator for functions that should handle degraded dependencies."""
    # Parameters are intentionally unused in this stub implementation
    def decorator(func):
        return func  # Just return the original function in stub mode
    return decorator

# Standard configuration stubs
CRITICAL_SERVICE_CONFIG = {"criticality": "critical"}
STANDARD_SERVICE_CONFIG = {"criticality": "standard"}
