"""
Timeout Module

This module provides timeout functionality for resilience.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable, Awaitable, Type, TypeVar, Generic, Union

from common_lib.errors.base import ServiceError


T = TypeVar('T')


class TimeoutError(ServiceError):
    """
    Timeout error.
    """
    
    def __init__(self, operation: str, timeout: float):
        """
        Initialize the timeout error.
        
        Args:
            operation: Operation that timed out
            timeout: Timeout in seconds
        """
        super().__init__(
            error_code=1008,
            message=f"Operation {operation} timed out after {timeout} seconds",
            details=f"The operation took longer than the specified timeout"
        )


async def with_timeout(
    func: Callable[[], Awaitable[T]],
    timeout: float,
    operation: str = "unknown"
) -> T:
    """
    Execute a function with timeout.
    
    Args:
        func: Function to execute
        timeout: Timeout in seconds
        operation: Name of the operation
        
    Returns:
        Result of the function
        
    Raises:
        TimeoutError: If the function times out
        Exception: If the function raises an exception
    """
    try:
        return await asyncio.wait_for(func(), timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(operation, timeout)


class Timeout:
    """
    Timeout for resilience.
    
    This class provides a configurable timeout for functions.
    """
    
    def __init__(
        self,
        timeout: float,
        operation: str = "unknown",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the timeout.
        
        Args:
            timeout: Timeout in seconds
            operation: Name of the operation
            logger: Logger to use (if None, creates a new logger)
        """
        self.timeout = timeout
        self.operation = operation
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def execute(self, func: Callable[[], Awaitable[T]]) -> T:
        """
        Execute a function with timeout.
        
        Args:
            func: Function to execute
            
        Returns:
            Result of the function
            
        Raises:
            TimeoutError: If the function times out
            Exception: If the function raises an exception
        """
        try:
            return await asyncio.wait_for(func(), timeout=self.timeout)
        except asyncio.TimeoutError:
            self.logger.warning(
                f"Operation {self.operation} timed out after {self.timeout} seconds"
            )
            raise TimeoutError(self.operation, self.timeout)
    
    def __call__(self, func):
        """
        Use the timeout as a decorator.
        
        Args:
            func: Function to decorate
            
        Returns:
            Decorated function
        """
        async def wrapper(*args, **kwargs):
            """
            Wrapper.
            
            Args:
                args: Description of args
                kwargs: Description of kwargs
            
            """

            return await self.execute(lambda: func(*args, **kwargs))
        
        return wrapper