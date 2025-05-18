import asyncio
import functools
import logging
import random
import time
from typing import Any, Callable, TypeVar

from common_lib.exceptions import RetryExhaustedError

# Generic type for the decorated function's return value
ResultT = TypeVar("ResultT")

# Global set to store exception types that should trigger a retry
_RETRYABLE_EXCEPTIONS = set()

def register_common_retryable_exceptions(exceptions: list[type[Exception]]) -> None:
    """Register common exceptions that should trigger a retry."""
    for exc in exceptions:
        _RETRYABLE_EXCEPTIONS.add(exc)

class RetryPolicy:
    """Implements a retry mechanism with exponential backoff and jitter."""

    def __init__(
        self,
        max_retries: int = 3,
        delay_seconds: float = 1.0,
        backoff_factor: float = 2.0,
        jitter_range: float = 0.1,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initializes the RetryPolicy.

        Args:
            max_retries: Maximum number of retry attempts.
            delay_seconds: Initial delay between retries in seconds.
            backoff_factor: Factor by which the delay increases after each retry.
            jitter_range: Percentage of jitter to apply to the delay (0.0 to 1.0).
            logger: Optional logger for logging retry attempts.
        """
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if delay_seconds <= 0:
            raise ValueError("delay_seconds must be positive")
        if backoff_factor < 1:
            raise ValueError("backoff_factor must be at least 1")
        if not 0 <= jitter_range <= 1:
            raise ValueError("jitter_range must be between 0 and 1")

        self.max_retries = max_retries
        self.delay_seconds = delay_seconds
        self.backoff_factor = backoff_factor
        self.jitter_range = jitter_range
        self.logger = logger or logging.getLogger(__name__)

    def _should_retry(self, exception: Exception) -> bool:
        """Determines if a retry should be attempted for the given exception."""
        return any(isinstance(exception, retryable_exc) for retryable_exc in _RETRYABLE_EXCEPTIONS)

    def _calculate_delay(self, attempt: int) -> float:
        """Calculates the delay for the current attempt with jitter."""
        delay = self.delay_seconds * (self.backoff_factor ** (attempt - 1))
        jitter = random.uniform(-self.jitter_range, self.jitter_range) * delay
        return delay + jitter

    async def execute_async(self, func: Callable[..., ResultT], *args: Any, **kwargs: Any) -> ResultT:
        """Executes an async function with retry logic."""
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if not self._should_retry(e) or attempt == self.max_retries:
                    self.logger.error(
                        f"Attempt {attempt + 1} failed with non-retryable error or max retries reached: {e}"
                    )
                    raise
                delay = self._calculate_delay(attempt + 1)
                self.logger.warning(
                    f"Attempt {attempt + 1} failed with {e.__class__.__name__}. Retrying in {delay:.2f} seconds..."
                )
                await asyncio.sleep(delay)
        # This line should ideally not be reached if max_retries is handled correctly
        raise RetryExhaustedError(
            f"All {self.max_retries + 1} attempts failed. Last error: {last_exception}"
        ) from last_exception

    def execute_sync(self, func: Callable[..., ResultT], *args: Any, **kwargs: Any) -> ResultT:
        """Executes a synchronous function with retry logic."""
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if not self._should_retry(e) or attempt == self.max_retries:
                    self.logger.error(
                        f"Attempt {attempt + 1} failed with non-retryable error or max retries reached: {e}"
                    )
                    raise
                delay = self._calculate_delay(attempt + 1)
                self.logger.warning(
                    f"Attempt {attempt + 1} failed with {e.__class__.__name__}. Retrying in {delay:.2f} seconds..."
                )
                time.sleep(delay)
        # This line should ideally not be reached if max_retries is handled correctly
        raise RetryExhaustedError(
            f"All {self.max_retries + 1} attempts failed. Last error: {last_exception}"
        ) from last_exception

def retry(max_retries: int = 3, delay_seconds: float = 1.0, backoff_factor: float = 2.0, jitter_range: float = 0.1):
    """Decorator to apply retry logic to a function.

    Args:
        max_retries (int): Maximum number of retry attempts.
        delay_seconds (float): Initial delay between retries in seconds.
        backoff_factor (float): Factor by which the delay increases after each retry.
        jitter_range (float): Percentage of jitter to apply to the delay (0.0 to 1.0).
    """
    policy = RetryPolicy(max_retries, delay_seconds, backoff_factor, jitter_range)

    def decorator(func: Callable[..., ResultT]) -> Callable[..., ResultT]:
        """Decorator function that wraps the original function with retry logic."""
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> ResultT:
            """
            Async wrapper.
            
            Args:
                args: Description of args
                kwargs: Description of kwargs
            
            Returns:
                T: Description of return value
            
            """
            return await policy.execute_async(func, *args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> ResultT:
            """
            Sync wrapper.
            
            Args:
                args: Description of args
                kwargs: Description of kwargs
            
            Returns:
                T: Description of return value
            
            """
            return policy.execute_sync(func, *args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator