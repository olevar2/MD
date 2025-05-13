"""
Bulkhead Module

This module provides bulkhead functionality for resilience.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable, Awaitable, Type, TypeVar, Generic, Union

from common_lib.errors.base import ServiceError


T = TypeVar('T')


class Bulkhead:
    """
    Bulkhead for resilience.
    
    This class implements the bulkhead pattern to isolate failures.
    """
    
    def __init__(
        self,
        name: str,
        max_concurrent_calls: int = 10,
        max_queue_size: int = 10,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the bulkhead.
        
        Args:
            name: Name of the bulkhead
            max_concurrent_calls: Maximum number of concurrent calls
            max_queue_size: Maximum size of the queue
            logger: Logger to use (if None, creates a new logger)
        """
        self.name = name
        self.max_concurrent_calls = max_concurrent_calls
        self.max_queue_size = max_queue_size
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self.semaphore = asyncio.Semaphore(max_concurrent_calls)
        self.queue_size = 0
        self.queue_semaphore = asyncio.Semaphore(max_queue_size)
    
    async def execute(self, func: Callable[[], Awaitable[T]]) -> T:
        """
        Execute a function with bulkhead protection.
        
        Args:
            func: Function to execute
            
        Returns:
            Result of the function
            
        Raises:
            ServiceError: If the bulkhead is full
            Exception: If the function raises an exception
        """
        # Try to acquire queue semaphore
        if not self.queue_semaphore.locked():
            try:
                self.queue_semaphore.release()
            except ValueError:
                pass
        
        queue_acquired = self.queue_semaphore.acquire()
        if not asyncio.wait_for(asyncio.shield(queue_acquired), timeout=0.01):
            self.logger.warning(f"Bulkhead {self.name} queue is full, rejecting request")
            raise ServiceError(
                code=2000,
                message=f"Service {self.name} is unavailable",
                details=f"Bulkhead queue is full"
            )
        
        # Increment queue size
        self.queue_size += 1
        
        try:
            # Try to acquire semaphore
            async with self.semaphore:
                # Decrement queue size
                self.queue_size -= 1
                
                # Release queue semaphore
                self.queue_semaphore.release()
                
                # Execute function
                return await func()
        except Exception as e:
            # Decrement queue size if not executed
            if self.queue_size > 0:
                self.queue_size -= 1
                
                # Release queue semaphore
                self.queue_semaphore.release()
            
            # Re-raise exception
            raise