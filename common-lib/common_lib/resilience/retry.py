"""
Retry Module

This module provides retry functionality for resilience.
"""

import logging
import time
import asyncio
import functools
from typing import Dict, Any, Optional, List, Callable, Awaitable, Type, TypeVar, Generic, Union, Tuple, cast

T = TypeVar('T')


def retry(
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Optional[List[Type[Exception]]] = None
):
    """
    Retry decorator for async functions.

    Args:
        retries: Maximum number of retries
        delay: Initial delay between retries in seconds
        backoff: Backoff factor for delay
        exceptions: List of exceptions to retry on (if None, retries on all exceptions)

    Returns:
        Decorated function
    """
    exceptions = exceptions or [Exception]

    def decorator(func):
        """
        Decorator.
        
        Args:
            func: Description of func
        
        """
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            """
            Wrapper.
            
            Args:
                args: Description of args
                kwargs: Description of kwargs
            
            """

            # Initialize retry counter
            retry_count = 0
            current_delay = delay

            # Execute function with retry
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Check if exception should be retried
                    if not any(isinstance(e, exc) for exc in exceptions):
                        raise

                    # Increment retry counter
                    retry_count += 1

                    # Check if maximum retries reached
                    if retry_count > retries:
                        raise

                    # Wait before retrying
                    await asyncio.sleep(current_delay)

                    # Increase delay for next retry
                    current_delay *= backoff

        return wrapper

    return decorator


class RetryPolicy:
    """
    Retry policy for resilience.

    This class provides a configurable retry policy for functions.
    """

    def __init__(
        self,
        retries: int = 3,
        delay: float = 1.0,
        max_delay: float = 60.0,
        backoff: float = 2.0,
        exceptions: Optional[List[Type[Exception]]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the retry policy.

        Args:
            retries: Maximum number of retries
            delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            backoff: Backoff factor for delay
            exceptions: List of exceptions to retry on (if None, retries on all exceptions)
            logger: Logger to use (if None, creates a new logger)
        """
        self.retries = retries
        self.delay = delay
        self.max_delay = max_delay
        self.backoff = backoff
        self.exceptions = exceptions or [Exception]
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(self, func: Callable[[], Awaitable[T]], *args, **kwargs) -> T:
        """
        Execute a function with retry.

        Args:
            func: Function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function

        Raises:
            Exception: If the function raises an exception after all retries
        """
        # Initialize retry counter
        retry_count = 0
        current_delay = self.delay

        # Execute function with retry
        while True:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Check if exception should be retried
                if not any(isinstance(e, exc) for exc in self.exceptions):
                    raise

                # Increment retry counter
                retry_count += 1

                # Check if maximum retries reached
                if retry_count > self.retries:
                    self.logger.error(
                        f"Maximum retries ({self.retries}) reached, giving up: {str(e)}"
                    )
                    raise

                # Log retry
                self.logger.warning(
                    f"Retry {retry_count}/{self.retries} after error: {str(e)}"
                )

                # Wait before retrying
                await asyncio.sleep(current_delay)

                # Increase delay for next retry
                current_delay = min(current_delay * self.backoff, self.max_delay)

    def __call__(self, func):
        """
        Use the retry policy as a decorator.

        Args:
            func: Function to decorate

        Returns:
            Decorated function
        """
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            """
            Wrapper.
            
            Args:
                args: Description of args
                kwargs: Description of kwargs
            
            """

            return await self.execute(lambda: func(*args, **kwargs))

        return wrapper


async def retry_with_policy(
    func: Callable[..., Awaitable[T]],
    policy: RetryPolicy,
    *args: Any,
    **kwargs: Any
) -> T:
    """
    Execute a function with a retry policy.

    Args:
        func: Function to execute
        policy: Retry policy to use
        *args: Arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function

    Raises:
        Exception: If the function raises an exception after all retries
    """
    return await policy.execute(lambda: func(*args, **kwargs))


class RetryExhaustedException(Exception):
    """
    Exception raised when a retry policy has exhausted all attempts.
    """
    def __init__(self, message: str = "Retry attempts exhausted"):
        """
        Initialize the exception.

        Args:
            message: The exception message.
        """
        super().__init__(message)


__all__ = [
    'retry',
    'RetryPolicy',
    'retry_with_policy',
    'RetryExhaustedException'
]