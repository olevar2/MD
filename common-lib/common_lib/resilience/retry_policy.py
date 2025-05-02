"""
Retry Policy Implementation

This module provides enhanced retry functionality by extending
the core implementation from core-foundations with integration to common-lib
monitoring, logging, and configuration.
"""

import functools
import logging
import time
import asyncio
import random
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, Coroutine, Set # Import Set from typing

# Import the base retry policy implementation
from core_foundations.resilience.retry_policy import (
    RetryPolicy as CoreRetryPolicy,
    RetryExhaustedException,
    register_common_retryable_exceptions as core_register_common_retryable_exceptions,
    retry_with_policy as core_retry_with_policy
)

# Setup logger
logger = logging.getLogger(__name__)

# Type variables for function return types
T = TypeVar('T')
R = TypeVar('R')

# Re-export RetryExhaustedException
__all__ = [
    "RetryPolicy", "RetryExhaustedException", "retry_with_policy",
    "register_common_retryable_exceptions", "register_database_retryable_exceptions"
]


class RetryPolicy(CoreRetryPolicy):
    """
    Enhanced RetryPolicy that integrates with common_lib monitoring
    and provides additional utilities specific to the Forex platform.
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: Optional[float] = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        exceptions: Optional[List[Type[Exception]]] = None,
        on_retry: Optional[Callable[[Any], None]] = None,
        metric_handler: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        service_name: Optional[str] = None,
        operation_name: Optional[str] = None
    ):
        """
        Initialize an enhanced retry policy with specific parameters.
        
        Args:
            max_attempts: Maximum number of attempts before failing.
            base_delay: Base delay time in seconds for exponential backoff.
            max_delay: Maximum delay between retries in seconds.
            backoff_factor: Multiplier for exponential backoff (typically 2).
            jitter: If True, adds randomization to delay times.
            exceptions: Set of exception types that trigger a retry.
            on_retry: Optional callback executed before sleeping on retry.
            metric_handler: Optional callback for reporting retry metrics.
            service_name: Name of the service using the retry policy.
            operation_name: Name of the operation being retried.
        """
        super().__init__(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            backoff_factor=backoff_factor,
            jitter=jitter,
            exceptions=exceptions,
            on_retry=on_retry,
            metric_handler=metric_handler
        )
        
        # Additional attributes for enhanced retry policy
        self.service_name = service_name
        self.operation_name = operation_name
        
    def _report_metrics(self, attempts: int, successful: bool, duration: float, exception: Optional[Exception] = None) -> None:
        """
        Enhanced metrics reporting that includes service and operation names.
        
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
            
            # Add service and operation names if available
            if self.service_name:
                metrics_data["service_name"] = self.service_name
            if self.operation_name:
                metrics_data["operation_name"] = self.operation_name
                
            if exception:
                metrics_data["last_exception_type"] = type(exception).__name__
                
            self.metric_handler("resilience.retry.execution", metrics_data)


def retry_with_policy(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: Optional[float] = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: Optional[List[Type[Exception]]] = None,
    on_retry: Optional[Callable[[Any], None]] = None,
    metric_handler: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    service_name: Optional[str] = None,
    operation_name: Optional[str] = None
) -> Callable[[Callable[..., Union[T, Coroutine[Any, Any, R]]]], 
             Callable[..., Union[T, Coroutine[Any, Any, R]]]]:
    """
    Enhanced decorator that applies a RetryPolicy to a function or coroutine.
    
    Args:
        max_attempts: Maximum number of attempts before failing.
        base_delay: Base delay time in seconds for exponential backoff.
        max_delay: Maximum delay between retries in seconds.
        backoff_factor: Multiplier for exponential backoff (typically 2).
        jitter: If True, adds randomization to delay times.
        exceptions: Set of exception types that trigger a retry.
        on_retry: Optional callback executed before sleeping on retry.
        metric_handler: Optional callback for reporting retry metrics.
        service_name: Name of the service using the retry policy.
        operation_name: Name of the operation being retried.
        
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
        metric_handler=metric_handler,
        service_name=service_name,
        operation_name=operation_name
    )

    def decorator(func: Callable[..., Union[T, Coroutine[Any, Any, R]]]) -> Callable[..., Union[T, Coroutine[Any, Any, R]]]:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> R:
                return await policy.execute_async(func, *args, **kwargs)
            return async_wrapper # type: ignore
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                return policy.execute_sync(func, *args, **kwargs)
            return sync_wrapper # type: ignore
    return decorator


def register_common_retryable_exceptions() -> Set[Type[Exception]]:
    """
    Re-exports the core register_common_retryable_exceptions function.
    
    Returns:
        Set of exception types that should typically be retried.
    """
    return core_register_common_retryable_exceptions()


def register_database_retryable_exceptions() -> Set[Type[Exception]]:
    """
    Registers database-specific exceptions that should be retried.
    
    Returns:
        Set of database-specific exception types that should be retried.
    """
    db_retryable_exceptions = set()
    
    # Add SQLAlchemy errors
    try:
        from sqlalchemy.exc import DBAPIError, TimeoutError as SATimeoutError, OperationalError
        db_retryable_exceptions.add(DBAPIError)
        db_retryable_exceptions.add(SATimeoutError)
        db_retryable_exceptions.add(OperationalError)
    except ImportError:
        pass
    
    # Add asyncpg errors
    try:
        import asyncpg
        db_retryable_exceptions.add(asyncpg.exceptions.CannotConnectNowError)
        db_retryable_exceptions.add(asyncpg.exceptions.ConnectionDoesNotExistError)
        db_retryable_exceptions.add(asyncpg.exceptions.InterfaceError)
    except ImportError:
        pass
    
    # Add psycopg2 errors
    try:
        import psycopg2.errors
        db_retryable_exceptions.add(psycopg2.errors.OperationalError)
        db_retryable_exceptions.add(psycopg2.errors.ConnectionException)
    except ImportError:
        pass
    
    return db_retryable_exceptions
