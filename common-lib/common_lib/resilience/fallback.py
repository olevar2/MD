"""
Fallback Module

This module provides fallback functionality for resilience.
"""

import logging
import functools
from typing import Dict, Any, Optional, List, Callable, Awaitable, Type, TypeVar, Generic, Union

T = TypeVar('T')


async def with_fallback(
    func: Callable[[], Awaitable[T]],
    fallback_func: Callable[[], Awaitable[T]],
    exceptions: Optional[List[Type[Exception]]] = None
) -> T:
    """
    Execute a function with fallback.
    
    Args:
        func: Function to execute
        fallback_func: Fallback function to execute if the main function fails
        exceptions: List of exceptions to trigger fallback (if None, fallback on all exceptions)
        
    Returns:
        Result of the function or fallback
        
    Raises:
        Exception: If both the function and fallback raise exceptions
    """
    exceptions = exceptions or [Exception]
    
    try:
        return await func()
    except Exception as e:
        # Check if exception should trigger fallback
        if not any(isinstance(e, exc) for exc in exceptions):
            raise
        
        # Execute fallback
        return await fallback_func()


class Fallback:
    """
    Fallback for resilience.
    
    This class provides a configurable fallback for functions.
    """
    
    def __init__(
        self,
        fallback_func: Callable[..., Awaitable[T]],
        exceptions: Optional[List[Type[Exception]]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the fallback.
        
        Args:
            fallback_func: Fallback function to execute if the main function fails
            exceptions: List of exceptions to trigger fallback (if None, fallback on all exceptions)
            logger: Logger to use (if None, creates a new logger)
        """
        self.fallback_func = fallback_func
        self.exceptions = exceptions or [Exception]
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def execute(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """
        Execute a function with fallback.
        
        Args:
            func: Function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function or fallback
            
        Raises:
            Exception: If both the function and fallback raise exceptions
        """
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Check if exception should trigger fallback
            if not any(isinstance(e, exc) for exc in self.exceptions):
                raise
            
            # Log fallback
            self.logger.warning(
                f"Executing fallback after error: {str(e)}"
            )
            
            # Execute fallback
            return await self.fallback_func(*args, **kwargs)
    
    def __call__(self, func):
        """
        Use the fallback as a decorator.
        
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

            return await self.execute(func, *args, **kwargs)
        
        return wrapper