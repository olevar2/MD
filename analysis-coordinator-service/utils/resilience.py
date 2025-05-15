import logging
import asyncio
import functools
import time
from typing import Callable, Any, TypeVar, cast

logger = logging.getLogger(__name__)

T = TypeVar('T')

def with_retry(max_retries: int = 3, backoff_factor: float = 0.5):
    """
    Decorator for retrying a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retries
        backoff_factor: Backoff factor for exponential backoff
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            retries = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Failed after {max_retries} retries: {str(e)}")
                        raise
                    
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    logger.warning(f"Retry {retries}/{max_retries} after {wait_time:.2f}s: {str(e)}")
                    await asyncio.sleep(wait_time)
        return wrapper
    return decorator

class CircuitBreaker:
    """
    Circuit breaker implementation.
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
        
    def can_execute(self) -> bool:
        """
        Check if the circuit breaker allows execution.
        """
        if self.state == "closed":
            return True
            
        if self.state == "open":
            # Check if recovery timeout has elapsed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                logger.info("Circuit breaker transitioning from open to half-open")
                self.state = "half-open"
                return True
            return False
            
        # Half-open state allows one request through
        return True
        
    def record_success(self) -> None:
        """
        Record a successful execution.
        """
        if self.state == "half-open":
            logger.info("Circuit breaker transitioning from half-open to closed")
            self.state = "closed"
            self.failure_count = 0
            
    def record_failure(self) -> None:
        """
        Record a failed execution.
        """
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == "closed" and self.failure_count >= self.failure_threshold:
            logger.warning(f"Circuit breaker transitioning from closed to open after {self.failure_count} failures")
            self.state = "open"
        elif self.state == "half-open":
            logger.warning("Circuit breaker transitioning from half-open to open after failure")
            self.state = "open"

# Global circuit breakers
circuit_breakers = {}

def with_circuit_breaker(failure_threshold: int = 5, recovery_timeout: int = 30):
    """
    Decorator for applying circuit breaker pattern to a function.
    
    Args:
        failure_threshold: Number of failures before opening the circuit
        recovery_timeout: Time in seconds before attempting to close the circuit
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Create a unique key for this function
        circuit_breaker_key = f"{func.__module__}.{func.__qualname__}"
        
        # Create a circuit breaker for this function if it doesn't exist
        if circuit_breaker_key not in circuit_breakers:
            circuit_breakers[circuit_breaker_key] = CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout
            )
            
        circuit_breaker = circuit_breakers[circuit_breaker_key]
        
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not circuit_breaker.can_execute():
                logger.warning(f"Circuit breaker open for {circuit_breaker_key}")
                raise Exception(f"Circuit breaker open for {circuit_breaker_key}")
                
            try:
                result = await func(*args, **kwargs)
                circuit_breaker.record_success()
                return result
            except Exception as e:
                circuit_breaker.record_failure()
                raise
                
        return wrapper
    return decorator