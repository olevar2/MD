"""
Parallel Processing Module

This module provides utilities for parallel processing of tasks.
It supports both synchronous and asynchronous parallel processing.
"""

import asyncio
import concurrent.futures
import enum
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar, Union

from common_lib.utils.platform_compatibility import PlatformCompatibility

# Type variables for generic functions
T = TypeVar('T')
U = TypeVar('U')

logger = logging.getLogger(__name__)


class TaskPriority(enum.Enum):
    """Priority levels for tasks."""
    
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class ParallelizationMethod(enum.Enum):
    """Methods for parallelizing tasks."""
    
    THREAD = 0
    PROCESS = 1
    ASYNC = 2
    AUTO = 3


@dataclass
class Task(Generic[T, U]):
    """
    Task for parallel processing.
    
    Attributes:
        func: Function to execute
        args: Positional arguments
        kwargs: Keyword arguments
        priority: Task priority
        id: Task ID
        result: Task result
        error: Task error
        start_time: Task start time
        end_time: Task end time
        status: Task status
    """
    
    func: Callable[..., U]
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = None
    priority: TaskPriority = TaskPriority.MEDIUM
    id: Optional[str] = None
    result: Optional[U] = None
    error: Optional[Exception] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    status: str = "pending"
    
    def __post_init__(self):
        """Initialize default values."""
        if self.kwargs is None:
            self.kwargs = {}
        if self.id is None:
            self.id = f"task_{id(self)}"
    
    def execute(self) -> U:
        """
        Execute the task.
        
        Returns:
            Task result
            
        Raises:
            Exception: If the task execution fails
        """
        self.start_time = time.time()
        self.status = "running"
        
        try:
            self.result = self.func(*self.args, **(self.kwargs or {}))
            self.status = "completed"
            return self.result
        except Exception as e:
            self.error = e
            self.status = "failed"
            raise
        finally:
            self.end_time = time.time()
    
    async def execute_async(self) -> U:
        """
        Execute the task asynchronously.
        
        Returns:
            Task result
            
        Raises:
            Exception: If the task execution fails
        """
        self.start_time = time.time()
        self.status = "running"
        
        try:
            if asyncio.iscoroutinefunction(self.func):
                self.result = await self.func(*self.args, **(self.kwargs or {}))
            else:
                # Run synchronous function in a thread pool
                loop = asyncio.get_event_loop()
                self.result = await loop.run_in_executor(
                    None, lambda: self.func(*self.args, **(self.kwargs or {}))
                )
            
            self.status = "completed"
            return self.result
        except Exception as e:
            self.error = e
            self.status = "failed"
            raise
        finally:
            self.end_time = time.time()
    
    @property
    def duration(self) -> Optional[float]:
        """Get the task duration in seconds."""
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time


class ParallelProcessor:
    """
    Processor for executing tasks in parallel.
    
    This class provides utilities for executing tasks in parallel using
    threads, processes, or asyncio.
    """
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        default_method: ParallelizationMethod = ParallelizationMethod.AUTO,
        thread_name_prefix: str = "parallel-worker"
    ):
        """
        Initialize the parallel processor.
        
        Args:
            max_workers: Maximum number of workers
            default_method: Default parallelization method
            thread_name_prefix: Prefix for thread names
        """
        self.max_workers = max_workers or PlatformCompatibility.get_optimal_thread_count()
        self.default_method = default_method
        self.thread_name_prefix = thread_name_prefix
        self.thread_executor = None
        self.process_executor = None
        self._lock = threading.RLock()
    
    def _get_executor(self, method: ParallelizationMethod) -> concurrent.futures.Executor:
        """
        Get the appropriate executor for the parallelization method.
        
        Args:
            method: Parallelization method
            
        Returns:
            Executor for the parallelization method
        """
        if method == ParallelizationMethod.THREAD or (
            method == ParallelizationMethod.AUTO and self.default_method in (ParallelizationMethod.THREAD, ParallelizationMethod.AUTO)
        ):
            with self._lock:
                if self.thread_executor is None:
                    self.thread_executor = concurrent.futures.ThreadPoolExecutor(
                        max_workers=self.max_workers,
                        thread_name_prefix=self.thread_name_prefix
                    )
                return self.thread_executor
        
        elif method == ParallelizationMethod.PROCESS or (
            method == ParallelizationMethod.AUTO and self.default_method == ParallelizationMethod.PROCESS
        ):
            with self._lock:
                if self.process_executor is None:
                    self.process_executor = concurrent.futures.ProcessPoolExecutor(
                        max_workers=self.max_workers
                    )
                return self.process_executor
        
        else:
            raise ValueError(f"Unsupported parallelization method: {method}")
    
    def map(
        self,
        func: Callable[[T], U],
        items: List[T],
        method: ParallelizationMethod = ParallelizationMethod.AUTO,
        timeout: Optional[float] = None,
        chunksize: int = 1
    ) -> List[U]:
        """
        Apply a function to each item in parallel.
        
        Args:
            func: Function to apply
            items: Items to process
            method: Parallelization method
            timeout: Timeout in seconds
            chunksize: Size of chunks for process pool
            
        Returns:
            List of results
            
        Raises:
            TimeoutError: If the operation times out
            Exception: If any task fails
        """
        if not items:
            return []
        
        if len(items) == 1:
            # No need for parallelization
            return [func(items[0])]
        
        executor = self._get_executor(method)
        return list(executor.map(func, items, chunksize=chunksize, timeout=timeout))
    
    def execute_tasks(
        self,
        tasks: List[Task[Any, U]],
        method: ParallelizationMethod = ParallelizationMethod.AUTO,
        timeout: Optional[float] = None
    ) -> List[U]:
        """
        Execute tasks in parallel.
        
        Args:
            tasks: Tasks to execute
            method: Parallelization method
            timeout: Timeout in seconds
            
        Returns:
            List of results
            
        Raises:
            TimeoutError: If the operation times out
            Exception: If any task fails
        """
        if not tasks:
            return []
        
        if len(tasks) == 1:
            # No need for parallelization
            return [tasks[0].execute()]
        
        # Sort tasks by priority
        sorted_tasks = sorted(tasks, key=lambda t: t.priority.value, reverse=True)
        
        # Execute tasks
        executor = self._get_executor(method)
        futures = {executor.submit(task.execute): task for task in sorted_tasks}
        
        # Wait for results
        results = []
        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            task = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                task.error = e
                task.status = "failed"
                raise
        
        return results
    
    async def map_async(
        self,
        func: Callable[[T], U],
        items: List[T],
        method: ParallelizationMethod = ParallelizationMethod.ASYNC,
        timeout: Optional[float] = None,
        max_concurrency: Optional[int] = None
    ) -> List[U]:
        """
        Apply a function to each item in parallel asynchronously.
        
        Args:
            func: Function to apply
            items: Items to process
            method: Parallelization method
            timeout: Timeout in seconds
            max_concurrency: Maximum number of concurrent tasks
            
        Returns:
            List of results
            
        Raises:
            TimeoutError: If the operation times out
            Exception: If any task fails
        """
        if not items:
            return []
        
        if len(items) == 1:
            # No need for parallelization
            if asyncio.iscoroutinefunction(func):
                return [await func(items[0])]
            else:
                loop = asyncio.get_event_loop()
                return [await loop.run_in_executor(None, func, items[0])]
        
        # Create tasks
        tasks = []
        semaphore = asyncio.Semaphore(max_concurrency or self.max_workers)
        
        async def wrapped_func(item):
    """
    Wrapped func.
    
    Args:
        item: Description of item
    
    """

            async with semaphore:
                if asyncio.iscoroutinefunction(func):
                    return await func(item)
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, func, item)
        
        for item in items:
            tasks.append(wrapped_func(item))
        
        # Execute tasks
        if timeout is not None:
            return await asyncio.gather(*tasks, timeout=timeout)
        else:
            return await asyncio.gather(*tasks)
    
    async def execute_tasks_async(
        self,
        tasks: List[Task[Any, U]],
        method: ParallelizationMethod = ParallelizationMethod.ASYNC,
        timeout: Optional[float] = None,
        max_concurrency: Optional[int] = None
    ) -> List[U]:
        """
        Execute tasks in parallel asynchronously.
        
        Args:
            tasks: Tasks to execute
            method: Parallelization method
            timeout: Timeout in seconds
            max_concurrency: Maximum number of concurrent tasks
            
        Returns:
            List of results
            
        Raises:
            TimeoutError: If the operation times out
            Exception: If any task fails
        """
        if not tasks:
            return []
        
        if len(tasks) == 1:
            # No need for parallelization
            return [await tasks[0].execute_async()]
        
        # Sort tasks by priority
        sorted_tasks = sorted(tasks, key=lambda t: t.priority.value, reverse=True)
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrency or self.max_workers)
        
        # Create wrapped task function
        async def execute_with_semaphore(task):
    """
    Execute with semaphore.
    
    Args:
        task: Description of task
    
    """

            async with semaphore:
                return await task.execute_async()
        
        # Execute tasks
        if timeout is not None:
            return await asyncio.gather(*[execute_with_semaphore(task) for task in sorted_tasks], timeout=timeout)
        else:
            return await asyncio.gather(*[execute_with_semaphore(task) for task in sorted_tasks])
    
    def close(self):
        """Close the executors."""
        if self.thread_executor is not None:
            self.thread_executor.shutdown()
            self.thread_executor = None
        
        if self.process_executor is not None:
            self.process_executor.shutdown()
            self.process_executor = None
    
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.close()
    
    async def __aenter__(self):
        """Enter async context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager."""
        self.close()
