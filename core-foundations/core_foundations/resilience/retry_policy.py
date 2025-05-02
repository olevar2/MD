"""
Retry Policy Implementation using Tenacity for Forex Trading Platform

This module provides a robust retry mechanism with exponential backoff
and jitter for resilient service communication, leveraging the tenacity library.
It allows operations that might fail temporarily to be automatically retried.

Usage:
    @retry_with_policy(max_attempts=3, base_delay=1.0, exceptions=(ConnectionError,))
    def call_external_service():
        # Make external call that might fail temporarily
        return result

    # Or manually for more control
    policy = RetryPolicy(max_attempts=3, base_delay=1.0)
    result = policy.execute(call_external_service) # For sync functions
    # result = await policy.execute_async(async_call_external_service) # For async functions
"""
import functools
import logging
import random
import time
import asyncio # Added for async support
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union, cast, Coroutine

# Import tenacity components
from tenacity import (
    AsyncRetrying, # Added for async
    Retrying,
    RetryError,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential, # Added for jitter
    before_sleep_log, # Added for logging
)
from tenacity.wait import wait_base # Added for type hinting

# Type variable for generic function return type
T = TypeVar('T')
# Type variable for async function return type
R = TypeVar('R')

# Configure logger
logger = logging.getLogger(__name__)


class RetryExhaustedException(Exception):
    """Exception raised when all retry attempts have been exhausted."""

    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"All {attempts} retry attempts exhausted. Last exception: {last_exception}"
        )


# Helper function to check if an exception should be retried
def _is_retryable_exception(exceptions_to_retry: Set[Type[Exception]]) -> Callable[[Exception], bool]:
    def check(exception: Exception) -> bool:
        return any(isinstance(exception, exc_type) for exc_type in exceptions_to_retry)
    return check

# Helper function for tenacity's before_sleep callback
def _log_retry(retry_state):
    logger.warning(
        f"Retrying {retry_state.fn.__name__ if retry_state.fn else 'function'} "
        f"after attempt {retry_state.attempt_number} due to: {retry_state.outcome.exception()}. "
        f"Sleeping for {retry_state.next_action.sleep:.2f}s."
    )


class RetryPolicy:
    """
    Implements a configurable retry policy using Tenacity.

    Attributes:
        max_attempts: Maximum number of attempts before failing.
        base_delay: Base delay time in seconds for exponential backoff.
        max_delay: Maximum delay between retries in seconds.
        backoff_factor: Multiplier for exponential backoff (typically 2).
        jitter: If True, uses wait_random_exponential, otherwise wait_exponential.
        exceptions: Set of exception types that trigger a retry.
        on_retry: Optional callback executed before sleeping on retry.
        metric_handler: Optional callback for reporting retry metrics (called after completion).
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: Optional[float] = 60.0, # Made Optional, Tenacity handles None as no max
        backoff_factor: float = 2.0,
        jitter: bool = True,
        exceptions: Optional[List[Type[Exception]]] = None,
        on_retry: Optional[Callable[[Any], None]] = _log_retry, # Use tenacity's retry_state
        metric_handler: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ):
        """
        Initialize a retry policy with specific parameters.
        Args are similar to the previous version but adapted for Tenacity.
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if base_delay < 0:
            raise ValueError("base_delay must be non-negative")
        if max_delay is not None and max_delay < base_delay:
            raise ValueError("max_delay must be greater than or equal to base_delay")
        if backoff_factor <= 1: # Tenacity uses multiplier > 1
             # If factor is 1, it's constant delay, not exponential.
             # If factor < 1, delay decreases. Let's enforce > 1 for exponential.
            raise ValueError("backoff_factor must be greater than 1 for exponential backoff")

        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        # Default to common retryable exceptions if None provided
        self.exceptions = set(exceptions or register_common_retryable_exceptions())
        self.on_retry = on_retry # Tenacity calls this 'before_sleep'
        self.metric_handler = metric_handler

        # Statistics (can be updated via callbacks if needed, but Tenacity tracks attempts)
        self._total_executions = 0
        self._total_failures = 0

    def _build_wait_strategy(self) -> wait_base:
        """Builds the Tenacity wait strategy based on configuration."""
        if self.jitter:
            # wait_random_exponential uses multiplier=base_delay, max=max_delay
            # It provides jitter around the exponential curve.
            return wait_random_exponential(multiplier=self.base_delay, max=self.max_delay)
        else:
            # wait_exponential uses multiplier=base_delay, max=max_delay, exp_base=backoff_factor
            return wait_exponential(multiplier=self.base_delay, max=self.max_delay, exp_base=self.backoff_factor)

    def _build_retryer(self) -> Retrying:
        """Builds a synchronous Tenacity Retrying instance."""
        return Retrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=self._build_wait_strategy(),
            retry=retry_if_exception(_is_retryable_exception(self.exceptions)),
            before_sleep=self.on_retry,
            reraise=True # Reraise the last exception if all attempts fail
        )

    def _build_async_retryer(self) -> AsyncRetrying:
        """Builds an asynchronous Tenacity AsyncRetrying instance."""
        return AsyncRetrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=self._build_wait_strategy(),
            retry=retry_if_exception(_is_retryable_exception(self.exceptions)),
            before_sleep=self.on_retry,
            reraise=True # Reraise the last exception if all attempts fail
        )

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute a synchronous function with the configured retry policy.

        Args:
            func: The synchronous function to execute.
            *args: Arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The result of the function.

        Raises:
            RetryExhaustedException: If all retries are exhausted.
            Any other exception raised by the function if it's not in self.exceptions.
        """
        retryer = self._build_retryer()
        start_time = time.monotonic()
        self._total_executions += 1
        try:
            result = retryer(func, *args, **kwargs)
            duration = time.monotonic() - start_time
            attempts = retryer.statistics.get('attempt_number', 1) if hasattr(retryer, 'statistics') else 1
            self._report_metrics(attempts, True, duration)
            return result
        except RetryError as e:
            # RetryError wraps the last exception
            duration = time.monotonic() - start_time
            attempts = self.max_attempts # Tenacity raises RetryError after max_attempts
            self._total_failures += 1
            self._report_metrics(attempts, False, duration, e.last_attempt.exception)
            raise RetryExhaustedException(attempts, e.last_attempt.exception) from e
        except Exception as e:
             # If the exception wasn't retryable, it's raised directly
             duration = time.monotonic() - start_time
             self._total_failures += 1
             # Report metrics for non-retryable failures too? Maybe add a flag.
             # self._report_metrics(1, False, duration, e)
             raise # Re-raise the original, non-retryable exception

    async def execute_async(self, func: Callable[..., Coroutine[Any, Any, R]], *args: Any, **kwargs: Any) -> R:
        """
        Execute an asynchronous function with the configured retry policy.

        Args:
            func: The asynchronous function (async def) to execute.
            *args: Arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The result of the function.

        Raises:
            RetryExhaustedException: If all retries are exhausted.
            Any other exception raised by the function if it's not in self.exceptions.
        """
        retryer = self._build_async_retryer()
        start_time = time.monotonic()
        self._total_executions += 1
        try:
            # Use 'await retryer.call(func, *args, **kwargs)' for Tenacity >= 8.1.0
            # For older versions or broader compatibility, use 'await retryer(func, *args, **kwargs)'
            result = await retryer(func, *args, **kwargs)
            duration = time.monotonic() - start_time
            attempts = retryer.statistics.get('attempt_number', 1) if hasattr(retryer, 'statistics') else 1
            self._report_metrics(attempts, True, duration)
            return result
        except RetryError as e:
            duration = time.monotonic() - start_time
            attempts = self.max_attempts
            self._total_failures += 1
            self._report_metrics(attempts, False, duration, e.last_attempt.exception)
            raise RetryExhaustedException(attempts, e.last_attempt.exception) from e
        except Exception as e:
             duration = time.monotonic() - start_time
             self._total_failures += 1
             # self._report_metrics(1, False, duration, e)
             raise

    def _report_metrics(self, attempts: int, successful: bool, duration: float, exception: Optional[Exception] = None) -> None:
        """
        Report retry metrics through the metric handler if provided.

        Args:
            attempts: Number of attempts made.
            successful: Whether the operation eventually succeeded.
            duration: Total duration including all retries in seconds.
            exception: The last exception if the operation failed.
        """
        if self.metric_handler:
            metrics_data = {
                "attempts": attempts,
                "successful": successful,
                "duration_seconds": duration,
                "policy_max_attempts": self.max_attempts,
                "policy_base_delay": self.base_delay,
                "policy_max_delay": self.max_delay,
                "policy_jitter": self.jitter,
                "timestamp": time.time()
            }
            if exception:
                metrics_data["last_exception_type"] = type(exception).__name__
                # metrics_data["last_exception_message"] = str(exception) # Optional: might be too verbose

            self.metric_handler("resilience.retry.execution", metrics_data)


def register_common_retryable_exceptions() -> Set[Type[Exception]]:
    """
    Identifies common network and temporary exceptions that should typically be retried.

    Returns:
        Set of exception types that should typically be retried.
    """
    retryable_exceptions = set()

    # Standard Python network/IO errors
    import socket
    retryable_exceptions.add(socket.error) # Includes subclasses like ConnectionError
    retryable_exceptions.add(socket.timeout)
    retryable_exceptions.add(ConnectionError) # Explicitly add for clarity
    retryable_exceptions.add(ConnectionResetError)
    retryable_exceptions.add(ConnectionRefusedError)
    retryable_exceptions.add(ConnectionAbortedError)
    retryable_exceptions.add(TimeoutError) # General timeout
    retryable_exceptions.add(asyncio.TimeoutError) # Asyncio specific timeout

    # HTTP library exceptions (optional, depends on project usage)
    try:
        import requests
        # Retry on connection errors, timeouts, and potentially 5xx server errors
        retryable_exceptions.add(requests.exceptions.ConnectionError)
        retryable_exceptions.add(requests.exceptions.Timeout)
        # Consider adding requests.exceptions.HTTPError for specific status codes (e.g., 502, 503, 504)
        # This requires a more complex retry= condition in Tenacity.
    except ImportError:
        pass

    try:
        import httpx
        # Retry on connection errors, timeouts, and potentially 5xx server errors
        retryable_exceptions.add(httpx.NetworkError) # Covers connection errors
        retryable_exceptions.add(httpx.TimeoutException) # Covers all timeouts
        # Consider adding httpx.HTTPStatusError for specific status codes (e.g., 502, 503, 504)
    except ImportError:
        pass

    # gRPC library exceptions (optional)
    try:
        import grpc
        # Retry on specific gRPC status codes indicating temporary issues
        # This requires a custom retry condition checking e.code()
        # e.g., grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED
        # For simplicity, we might just retry on RpcError initially, but refine later.
        retryable_exceptions.add(grpc.RpcError)
    except ImportError:
        pass

    # Database library exceptions (optional, specific errors are better)
    # Example for psycopg2 (replace with actual library used if needed)
    # try:
    #     import psycopg2.errors
    #     retryable_exceptions.add(psycopg2.errors.OperationalError) # Often includes connection issues
    # except ImportError:
    #     pass

    # Example for asyncpg
    try:
        import asyncpg
        # Specific connection-related errors
        retryable_exceptions.add(asyncpg.exceptions.CannotConnectNowError)
        retryable_exceptions.add(asyncpg.exceptions.InterfaceError) # Can indicate connection issues
        # Be cautious about retrying all OperationalErrors, some might be persistent
    except ImportError:
        pass


    # Add common SQLAlchemy errors if SQLAlchemy is used
    try:
        from sqlalchemy.exc import DBAPIError, TimeoutError as SATimeoutError, OperationalError as SAOperationalError
        # Retry on DBAPIError which often wraps connection issues, and specific timeouts
        retryable_exceptions.add(DBAPIError)
        retryable_exceptions.add(SATimeoutError)
        # Be cautious with OperationalError - check specific sub-types or messages if possible
        # retryable_exceptions.add(SAOperationalError)
    except ImportError:
        pass


    logger.debug(f"Registered common retryable exceptions: {[e.__name__ for e in retryable_exceptions]}")
    return retryable_exceptions


# Decorator using the RetryPolicy class
# Renamed to avoid conflict with tenacity.retry
def retry_with_policy(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: Optional[float] = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: Optional[List[Type[Exception]]] = None,
    on_retry: Optional[Callable[[Any], None]] = _log_retry,
    metric_handler: Optional[Callable[[str, Dict[str, Any]], None]] = None
) -> Callable[[Callable[..., Union[T, Coroutine[Any, Any, R]]]], Callable[..., Union[T, Coroutine[Any, Any, R]]]]:
    """
    Decorator that applies a RetryPolicy to a function or coroutine.

    Args are passed directly to the RetryPolicy constructor.

    Returns:
        A decorator function.
    """
    policy = RetryPolicy(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_factor=backoff_factor,
        jitter=jitter,
        exceptions=exceptions,
        on_retry=on_retry,
        metric_handler=metric_handler
    )

    def decorator(func: Callable[..., Union[T, Coroutine[Any, Any, R]]]) -> Callable[..., Union[T, Coroutine[Any, Any, R]]]:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> R:
                # Cast func for the async execution path
                async_func = cast(Callable[..., Coroutine[Any, Any, R]], func)
                return await policy.execute_async(async_func, *args, **kwargs)
            # Add policy instance for inspection
            async_wrapper.retry_policy = policy # type: ignore
            return async_wrapper # Return the async wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                 # Cast func for the sync execution path
                sync_func = cast(Callable[..., T], func)
                return policy.execute(sync_func, *args, **kwargs)
            # Add policy instance for inspection
            sync_wrapper.retry_policy = policy # type: ignore
            return sync_wrapper # Return the sync wrapper

    return decorator

# Alias for backward compatibility if needed, though retry_with_policy is clearer
retry = retry_with_policy
