"""
Bulkhead Pattern Implementation

This module provides implementation of the bulkhead pattern, which isolates
components of an application into pools so that if one fails, the others continue to function.
"""

import asyncio
import functools
import logging
from typing import Any, Callable, Dict, Optional, TypeVar, cast, Union, Set

# Setup logger
logger = logging.getLogger(__name__)

# Type variables for function return types
T = TypeVar('T')
R = TypeVar('R')

__all__ = ["Bulkhead", "bulkhead", "BulkheadFullException"]


class BulkheadFullException(Exception):
    """Exception raised when a bulkhead is at capacity."""
    
    def __init__(self, name: str, max_concurrent: int):
        self.name = name
        self.max_concurrent = max_concurrent
        super().__init__(
            f"Bulkhead '{name}' is at maximum capacity ({max_concurrent} concurrent executions)"
        )


class Bulkhead:
    """
    Implements the bulkhead pattern to limit concurrent executions.
    
    This helps prevent resource exhaustion and cascading failures by
    limiting the number of concurrent executions of a particular operation.
    """
    
    def __init__(
        self,
        name: str,
        max_concurrent: int,
        max_waiting: Optional[int] = None,
        wait_timeout: Optional[float] = None
    ):
        """
        Initialize a bulkhead with specified limits.
        
        Args:
            name: Identifier for the bulkhead
            max_concurrent: Maximum number of concurrent executions
            max_waiting: Maximum number of waiting executions (None for unlimited)
            wait_timeout: Maximum time to wait for execution slot (None for no timeout)
        """
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
            
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_waiting = max_waiting
        self.wait_timeout = wait_timeout
        
        # Use a semaphore to track and limit concurrent executions
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
        # Use a queue to track waiting executions if max_waiting is specified
        self._waiting_queue = asyncio.Queue(maxsize=max_waiting or 0) if max_waiting else None
        
        # Track metrics
        self._total_executions = 0
        self._rejected_executions = 0
        self._timeout_executions = 0
        
    async def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute a function with bulkhead protection.
        
        Args:
            func: The function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The function's result
            
        Raises:
            BulkheadFullException: If bulkhead is at capacity
            asyncio.TimeoutError: If wait_timeout is specified and exceeded
        """
        self._total_executions += 1
        
        # Handle waiting queue if max_waiting is specified
        if self._waiting_queue is not None:
            return await self._execute_with_waiting_queue(func, *args, **kwargs)
        else:
            # No waiting queue, just use the semaphore
            return await self._execute_with_semaphore_only(func, *args, **kwargs)
    
    async def _execute_with_waiting_queue(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute with waiting queue management."""
        try:
            # Try to put a token in the waiting queue (non-blocking)
            self._waiting_queue.put_nowait(True)
        except asyncio.QueueFull:
            self._rejected_executions += 1
            raise BulkheadFullException(self.name, self.max_concurrent)
            
        try:
            # Now try to acquire execution permit
            if self.wait_timeout is not None:
                return await self._execute_with_timeout(func, *args, **kwargs)
            else:
                return await self._execute_without_timeout(func, *args, **kwargs)
        finally:
            # Remove from waiting queue
            await self._waiting_queue.get()
            self._waiting_queue.task_done()
    
    async def _execute_with_semaphore_only(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute with just the semaphore, no waiting queue."""
        if self.wait_timeout is not None:
            return await self._execute_with_timeout(func, *args, **kwargs)
        else:
            return await self._execute_without_timeout(func, *args, **kwargs)
            
    async def _execute_with_timeout(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute with a timeout."""
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(), 
                timeout=self.wait_timeout
            )
            # If we get here, we've acquired the semaphore
            try:
                return await self._execute_func(func, *args, **kwargs)
            finally:
                self._semaphore.release()
        except asyncio.TimeoutError:
            self._timeout_executions += 1
            raise
    
    async def _execute_without_timeout(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute without a timeout."""
        await self._semaphore.acquire()
        try:
            return await self._execute_func(func, *args, **kwargs)
        finally:
            self._semaphore.release()
    
    async def _execute_func(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute the function, handling both sync and async functions."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # Run sync function in a thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: func(*args, **kwargs)
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get bulkhead metrics."""
        return {
            "name": self.name,
            "max_concurrent": self.max_concurrent,
            "max_waiting": self.max_waiting,
            "available_concurrent_slots": self._semaphore._value,  # type: ignore
            "total_executions": self._total_executions,
            "rejected_executions": self._rejected_executions,
            "timeout_executions": self._timeout_executions
        }


def bulkhead(
    name: str,
    max_concurrent: int,
    max_waiting: Optional[int] = None,
    wait_timeout: Optional[float] = None
) -> Callable[[Callable], Callable]:
    """
    Decorator that applies a bulkhead to a function or coroutine.
    
    Args:
        name: Identifier for the bulkhead
        max_concurrent: Maximum number of concurrent executions
        max_waiting: Maximum number of waiting executions (None for unlimited)
        wait_timeout: Maximum time to wait for execution slot (None for no timeout)
        
    Returns:
        A decorator function
    """
    bh = Bulkhead(
        name=name,
        max_concurrent=max_concurrent,
        max_waiting=max_waiting,
        wait_timeout=wait_timeout
    )
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await bh.execute(func, *args, **kwargs)
        
        # Store reference to bulkhead for inspection
        wrapper.bulkhead = bh  # type: ignore
        return wrapper
        
    return decorator
