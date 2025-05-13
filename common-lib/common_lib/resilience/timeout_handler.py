"""
Timeout Handler Implementation

This module provides utilities for handling timeouts in various operations,
ensuring that long-running operations do not block indefinitely.
"""

import asyncio
import functools
import logging
from typing import Any, Callable, TypeVar, Awaitable, Optional, cast, Union

# Setup logger
logger = logging.getLogger(__name__)

# Type variables for function return types
T = TypeVar('T')
R = TypeVar('R')

__all__ = ["timeout_handler", "async_timeout", "sync_timeout", "TimeoutError"]


class TimeoutError(Exception):
    """Exception raised when an operation times out."""
    
    def __init__(self, operation_name: str, timeout_seconds: float):
    """
      init  .
    
    Args:
        operation_name: Description of operation_name
        timeout_seconds: Description of timeout_seconds
    
    """

        self.operation_name = operation_name
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Operation '{operation_name}' timed out after {timeout_seconds} seconds")


async def async_timeout(
    coroutine: Awaitable[T],
    timeout_seconds: float,
    operation_name: Optional[str] = None
) -> T:
    """
    Execute an asynchronous operation with a timeout.
    
    Args:
        coroutine: The awaitable to execute
        timeout_seconds: Maximum execution time in seconds
        operation_name: Optional name for the operation (for error messages)
        
    Returns:
        The result of the coroutine
        
    Raises:
        TimeoutError: If the operation exceeds the timeout
    """
    try:
        return await asyncio.wait_for(coroutine, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        op_name = operation_name or "Unknown"
        logger.warning(f"Operation '{op_name}' timed out after {timeout_seconds} seconds")
        raise TimeoutError(op_name, timeout_seconds)


def sync_timeout(
    func: Callable[..., T],
    args: Any = None,
    kwargs: Any = None,
    timeout_seconds: float = 10.0,
    operation_name: Optional[str] = None
) -> T:
    """
    Execute a synchronous function with a timeout by running it in a separate thread.
    
    Args:
        func: The function to execute
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        timeout_seconds: Maximum execution time in seconds
        operation_name: Optional name for the operation (for error messages)
        
    Returns:
        The result of the function
        
    Raises:
        TimeoutError: If the operation exceeds the timeout
    """
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
        
    loop = asyncio.get_event_loop()
    
    # Run synchronous function in a thread pool
    future = loop.run_in_executor(
        None,
        lambda: func(*args, **kwargs)
    )
    
    # Use asyncio.wait_for to apply the timeout
    try:
        return loop.run_until_complete(
            asyncio.wait_for(future, timeout=timeout_seconds)
        )
    except asyncio.TimeoutError:
        op_name = operation_name or func.__name__
        logger.warning(f"Operation '{op_name}' timed out after {timeout_seconds} seconds")
        raise TimeoutError(op_name, timeout_seconds)


def timeout_handler(
    timeout_seconds: float,
    operation_name: Optional[str] = None
) -> Callable[[Callable], Callable]:
    """
    Decorator that applies a timeout to a function or coroutine.
    
    Args:
        timeout_seconds: Maximum execution time in seconds
        operation_name: Optional name for the operation (for error messages)
        
    Returns:
        A decorator function
    """
    def decorator(func: Callable) -> Callable:
    """
    Decorator.
    
    Args:
        func: Description of func
    
    Returns:
        Callable: Description of return value
    
    """

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
    """
    Async wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

                # Handle async function
                coro = func(*args, **kwargs)
                op_name = operation_name or func.__name__
                return await async_timeout(coro, timeout_seconds, op_name)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
    """
    Sync wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

                # Handle sync function
                op_name = operation_name or func.__name__
                return sync_timeout(func, args, kwargs, timeout_seconds, op_name)
            return sync_wrapper
            
    return decorator
