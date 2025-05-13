"""
Parallel Processor Module

This module provides a comprehensive parallel processing framework for the forex trading platform.
It includes support for thread-based, process-based, and async-based parallelism with
resource-aware worker allocation and task prioritization.

Features:
- Dynamic selection between thread, process, and async-based parallelism
- Resource-aware worker allocation
- Priority-based task scheduling
- Dependency-aware task execution
- Comprehensive error handling and reporting
- Performance monitoring and metrics collection
"""

import asyncio
import concurrent.futures
import logging
import multiprocessing
import os
import queue
import threading
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar, Union, cast, Hashable

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from common_lib.errors.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    BaseError,
    ServiceError
)

from common_lib.monitoring.performance_monitoring import (
    track_operation
)

# Type variables
T = TypeVar('T')  # Input type
R = TypeVar('R')  # Result type

# Create logger
logger = logging.getLogger(__name__)


class ParallelizationMethod(Enum):
    """Method for parallelizing tasks."""
    THREAD = auto()  # Thread-based parallelism
    PROCESS = auto()  # Process-based parallelism
    ASYNC = auto()    # Async-based parallelism
    AUTO = auto()     # Automatically select the best method


class TaskPriority(Enum):
    """Priority for task execution."""
    CRITICAL = 0  # Highest priority
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BACKGROUND = 4  # Lowest priority


@dataclass
class TaskDefinition(Generic[T, R]):
    """Definition of a task to be executed in parallel."""
    id: str  # Unique identifier for the task
    func: Callable[[T], R]  # Function to execute
    input_data: T  # Input data for the function
    priority: TaskPriority = TaskPriority.MEDIUM  # Priority for execution
    parallelization_method: ParallelizationMethod = ParallelizationMethod.AUTO  # Method for parallelization
    timeout: Optional[float] = None  # Timeout in seconds
    dependencies: Optional[List[str]] = None  # IDs of tasks that must complete before this one


@dataclass
class TaskResult(Generic[R]):
    """Result of a task execution."""
    task_id: str  # ID of the task
    result: Optional[R] = None  # Result of the task
    error: Optional[Exception] = None  # Error if the task failed
    execution_time: float = 0.0  # Execution time in seconds
    success: bool = True  # Whether the task succeeded


class ResourceManager:
    """Manages resources for parallel execution."""
    
    def __init__(
        self,
        reserved_cores: int = 1,
        max_cpu_percent: float = 80.0,
        max_memory_percent: float = 80.0
    ):
        """
        Initialize the resource manager.
        
        Args:
            reserved_cores: Number of CPU cores to reserve for other tasks
            max_cpu_percent: Maximum CPU usage percentage
            max_memory_percent: Maximum memory usage percentage
        """
        self.reserved_cores = reserved_cores
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self.total_cores = multiprocessing.cpu_count()
    
    @track_operation("parallel", "get_available_workers")
    def get_available_workers(self, method: ParallelizationMethod) -> int:
        """
        Get the number of available workers for a parallelization method.
        
        Args:
            method: Parallelization method
            
        Returns:
            Number of available workers
        """
        if method == ParallelizationMethod.THREAD:
            return max(1, min(self.total_cores * 2, 32))
        
        if method == ParallelizationMethod.ASYNC:
            return max(1, min(self.total_cores * 4, 64))
        
        # For process-based parallelism
        available_cores = max(1, self.total_cores - self.reserved_cores)
        
        if PSUTIL_AVAILABLE:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent
                
                if cpu_percent > self.max_cpu_percent:
                    available_cores = max(1, available_cores // 2)
                
                if memory_percent > self.max_memory_percent:
                    available_cores = max(1, available_cores - 1)
            except Exception as e:
                logger.warning(f"Error getting system resources: {e}")
        
        return available_cores


class TaskScheduler:
    """Schedules tasks for parallel execution based on priorities and dependencies."""
    
    @track_operation("parallel", "schedule_tasks")
    def schedule_tasks(self, tasks: List[TaskDefinition[T, R]]) -> List[List[TaskDefinition[T, R]]]:
        """
        Schedule tasks for parallel execution.
        
        Args:
            tasks: List of task definitions
            
        Returns:
            List of task groups to execute in sequence
        """
        if not tasks:
            return []
        
        # Create dependency graph
        dependency_graph: Dict[str, Set[str]] = {}
        task_map: Dict[str, TaskDefinition[T, R]] = {}
        
        for task in tasks:
            task_map[task.id] = task
            dependency_graph[task.id] = set(task.dependencies or [])
        
        # Check for missing dependencies
        all_task_ids = set(task_map.keys())
        for task_id, dependencies in dependency_graph.items():
            missing_dependencies = dependencies - all_task_ids
            if missing_dependencies:
                logger.warning(f"Task {task_id} has missing dependencies: {missing_dependencies}")
                dependency_graph[task_id] = dependencies & all_task_ids
        
        # Topologically sort tasks
        execution_groups: List[List[TaskDefinition[T, R]]] = []
        remaining_tasks = set(task_map.keys())
        
        while remaining_tasks:
            # Find tasks with no dependencies
            ready_tasks = {task_id for task_id in remaining_tasks if not dependency_graph[task_id]}
            
            if not ready_tasks:
                # Circular dependency detected
                logger.warning("Circular dependency detected in tasks")
                # Break the cycle by selecting the highest priority task
                ready_tasks = {min(remaining_tasks, key=lambda task_id: task_map[task_id].priority.value)}
            
            # Group tasks by priority
            priority_groups: Dict[TaskPriority, List[TaskDefinition[T, R]]] = {}
            for task_id in ready_tasks:
                task = task_map[task_id]
                if task.priority not in priority_groups:
                    priority_groups[task.priority] = []
                priority_groups[task.priority].append(task)
            
            # Add tasks to execution groups by priority
            for priority in sorted(priority_groups.keys(), key=lambda p: p.value):
                execution_groups.append(priority_groups[priority])
            
            # Update remaining tasks and dependencies
            for task_id in ready_tasks:
                remaining_tasks.remove(task_id)
                for deps in dependency_graph.values():
                    deps.discard(task_id)
        
        return execution_groups


class ParallelProcessor(Generic[T, R]):
    """
    Executes tasks in parallel using thread, process, or async-based parallelism.
    
    Features:
    - Dynamic selection between thread, process, and async-based parallelism
    - Resource-aware worker allocation
    - Priority-based task scheduling
    - Dependency-aware task execution
    - Comprehensive error handling and reporting
    - Performance monitoring and metrics collection
    """
    
    def __init__(
        self,
        resource_manager: Optional[ResourceManager] = None,
        min_workers: int = 2,
        max_workers: int = 8
    ):
        """
        Initialize the parallel processor.
        
        Args:
            resource_manager: Resource manager for worker allocation
            min_workers: Minimum number of worker threads/processes
            max_workers: Maximum number of worker threads/processes
        """
        self.resource_manager = resource_manager or ResourceManager()
        self.scheduler = TaskScheduler()
        self.min_workers = min_workers
        self.max_workers = max_workers
    
    @track_operation("parallel", "execute_tasks")
    async def execute_tasks(self, tasks: List[TaskDefinition[T, R]]) -> Dict[str, TaskResult[R]]:
        """
        Execute tasks in parallel with dependency and priority handling.
        
        Args:
            tasks: List of task definitions
            
        Returns:
            Dictionary mapping task IDs to results
        """
        if not tasks:
            return {}
        
        # Schedule tasks
        execution_groups = self.scheduler.schedule_tasks(tasks)
        results: Dict[str, TaskResult[R]] = {}
        
        # Execute task groups in sequence
        for group_idx, task_group in enumerate(execution_groups):
            logger.debug(
                f"Executing task group {group_idx + 1}/{len(execution_groups)} with {len(task_group)} tasks"
            )
            
            # Group tasks by parallelization method
            thread_tasks = []
            process_tasks = []
            async_tasks = []
            
            for task in task_group:
                method = self._determine_parallelization_method(task)
                
                if method == ParallelizationMethod.THREAD:
                    thread_tasks.append(task)
                elif method == ParallelizationMethod.PROCESS:
                    process_tasks.append(task)
                elif method == ParallelizationMethod.ASYNC:
                    async_tasks.append(task)
            
            # Execute tasks by method
            if thread_tasks:
                thread_results = await self._execute_with_threads(thread_tasks)
                results.update(thread_results)
            
            if process_tasks:
                process_results = await self._execute_with_processes(process_tasks)
                results.update(process_results)
            
            if async_tasks:
                async_results = await self._execute_with_async(async_tasks)
                results.update(async_results)
        
        return results
    
    def _determine_parallelization_method(self, task: TaskDefinition[T, R]) -> ParallelizationMethod:
        """
        Determine the best parallelization method for a task.
        
        Args:
            task: Task definition
            
        Returns:
            Parallelization method
        """
        if task.parallelization_method != ParallelizationMethod.AUTO:
            return task.parallelization_method
        
        # Determine the best method based on the task
        if asyncio.iscoroutinefunction(task.func):
            return ParallelizationMethod.ASYNC
        
        # Check if the function is CPU-bound or IO-bound
        # This is a heuristic and may not be accurate for all cases
        try:
            func_code = task.func.__code__
            if func_code.co_flags & 0x80:  # Check if the function is a coroutine
                return ParallelizationMethod.ASYNC
        except AttributeError:
            pass
        
        # Default to thread-based parallelism
        return ParallelizationMethod.THREAD
    
    @async_with_exception_handling
    async def _execute_with_threads(self, tasks: List[TaskDefinition[T, R]]) -> Dict[str, TaskResult[R]]:
        """
        Execute tasks using a thread pool.
        
        Args:
            tasks: List of task definitions
            
        Returns:
            Dictionary mapping task IDs to results
        """
        results: Dict[str, TaskResult[R]] = {}
        max_workers = min(
            self.max_workers,
            max(
                self.min_workers,
                self.resource_manager.get_available_workers(ParallelizationMethod.THREAD)
            )
        )
        
        loop = asyncio.get_event_loop()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {}
            
            for task in tasks:
                future = loop.run_in_executor(
                    executor,
                    self._execute_task_with_timing,
                    task.func,
                    task.input_data,
                    task.timeout
                )
                future_to_task[future] = task
            
            for future in asyncio.as_completed(future_to_task.keys()):
                task = future_to_task[future]
                try:
                    success, result, error, execution_time = await future
                    results[task.id] = TaskResult(
                        task_id=task.id,
                        result=result,
                        error=error,
                        execution_time=execution_time,
                        success=success
                    )
                except Exception as e:
                    logger.error(f"Error executing task {task.id}: {e}")
                    results[task.id] = TaskResult(
                        task_id=task.id,
                        error=e,
                        success=False
                    )
        
        return results
    
    @async_with_exception_handling
    async def _execute_with_processes(self, tasks: List[TaskDefinition[T, R]]) -> Dict[str, TaskResult[R]]:
        """
        Execute tasks using a process pool.
        
        Args:
            tasks: List of task definitions
            
        Returns:
            Dictionary mapping task IDs to results
        """
        results: Dict[str, TaskResult[R]] = {}
        max_workers = min(
            self.max_workers,
            max(
                self.min_workers,
                self.resource_manager.get_available_workers(ParallelizationMethod.PROCESS)
            )
        )
        
        loop = asyncio.get_event_loop()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {}
            
            for task in tasks:
                future = loop.run_in_executor(
                    executor,
                    self._execute_task_with_timing,
                    task.func,
                    task.input_data,
                    task.timeout
                )
                future_to_task[future] = task
            
            for future in asyncio.as_completed(future_to_task.keys()):
                task = future_to_task[future]
                try:
                    success, result, error, execution_time = await future
                    results[task.id] = TaskResult(
                        task_id=task.id,
                        result=result,
                        error=error,
                        execution_time=execution_time,
                        success=success
                    )
                except Exception as e:
                    logger.error(f"Error executing task {task.id}: {e}")
                    results[task.id] = TaskResult(
                        task_id=task.id,
                        error=e,
                        success=False
                    )
        
        return results
    
    @async_with_exception_handling
    async def _execute_with_async(self, tasks: List[TaskDefinition[T, R]]) -> Dict[str, TaskResult[R]]:
        """
        Execute tasks using async/await.
        
        Args:
            tasks: List of task definitions
            
        Returns:
            Dictionary mapping task IDs to results
        """
        results: Dict[str, TaskResult[R]] = {}
        max_concurrency = min(
            self.max_workers * 2,
            max(
                self.min_workers * 2,
                self.resource_manager.get_available_workers(ParallelizationMethod.ASYNC)
            )
        )
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrency)
        
        # Create tasks
        async_tasks = []
        
        for task in tasks:
            async_tasks.append(self._execute_async_task(task, semaphore))
        
        # Execute tasks
        task_results = await asyncio.gather(*async_tasks, return_exceptions=True)
        
        # Process results
        for task, result in zip(tasks, task_results):
            if isinstance(result, Exception):
                logger.error(f"Error executing task {task.id}: {result}")
                results[task.id] = TaskResult(
                    task_id=task.id,
                    error=result,
                    success=False
                )
            else:
                success, task_result, error, execution_time = result
                results[task.id] = TaskResult(
                    task_id=task.id,
                    result=task_result,
                    error=error,
                    execution_time=execution_time,
                    success=success
                )
        
        return results
    
    async def _execute_async_task(
        self,
        task: TaskDefinition[T, R],
        semaphore: asyncio.Semaphore
    ) -> Tuple[bool, Optional[R], Optional[Exception], float]:
        """
        Execute an async task with timing.
        
        Args:
            task: Task definition
            semaphore: Semaphore for concurrency control
            
        Returns:
            Tuple of (success, result, error, execution_time)
        """
        async with semaphore:
            start_time = time.time()
            success = True
            result = None
            error = None
            
            try:
                if asyncio.iscoroutinefunction(task.func):
                    # Function is already async
                    if task.timeout:
                        result = await asyncio.wait_for(task.func(task.input_data), timeout=task.timeout)
                    else:
                        result = await task.func(task.input_data)
                else:
                    # Run sync function in executor
                    loop = asyncio.get_event_loop()
                    if task.timeout:
                        result = await asyncio.wait_for(
                            loop.run_in_executor(None, task.func, task.input_data),
                            timeout=task.timeout
                        )
                    else:
                        result = await loop.run_in_executor(None, task.func, task.input_data)
            except asyncio.TimeoutError as e:
                success = False
                error = e
                logger.warning(f"Task {task.id} timed out after {task.timeout} seconds")
            except Exception as e:
                success = False
                error = e
                logger.error(f"Error executing task {task.id}: {e}")
            
            execution_time = time.time() - start_time
            return success, result, error, execution_time
    
    @with_exception_handling
    def _execute_task_with_timing(
        self,
        func: Callable[[T], R],
        input_data: T,
        timeout: Optional[float]
    ) -> Tuple[bool, Optional[R], Optional[Exception], float]:
        """
        Execute a task with timing.
        
        Args:
            func: Function to execute
            input_data: Input data for the function
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (success, result, error, execution_time)
        """
        start_time = time.time()
        success = True
        result = None
        error = None
        
        try:
            if timeout:
                # Create a future for the function
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(func, input_data)
                    result = future.result(timeout=timeout)
            else:
                result = func(input_data)
        except concurrent.futures.TimeoutError as e:
            success = False
            error = e
            logger.warning(f"Task timed out after {timeout} seconds")
        except Exception as e:
            success = False
            error = e
            logger.error(f"Error executing task: {e}")
        
        execution_time = time.time() - start_time
        return success, result, error, execution_time
    
    @track_operation("parallel", "process")
    def process(
        self,
        tasks: List[Tuple[int, Callable, Tuple]],
        timeout: Optional[float] = None
    ) -> Dict[int, Any]:
        """
        Process tasks in parallel with priority.
        
        Args:
            tasks: List of (priority, function, args) tuples
                  Lower priority values are processed first
            timeout: Optional timeout in seconds
            
        Returns:
            Dictionary mapping task IDs to results
        """
        if not tasks:
            return {}
        
        # Convert to task definitions
        task_definitions = []
        
        for idx, (priority, func, args) in enumerate(tasks):
            task_id = f"task_{idx}_{id(func)}"
            
            # Add hash of args to task_id if possible
            if args:
                for arg in args:
                    if isinstance(arg, Hashable):
                        task_id += f"_{hash(arg)}"
            
            # Create wrapper function
            def create_wrapper(f, a):
    """
    Create wrapper.
    
    Args:
        f: Description of f
        a: Description of a
    
    """

                return lambda _: f(*a)
            
            wrapper_func = create_wrapper(func, args)
            
            # Map priority to TaskPriority
            task_priority = TaskPriority.MEDIUM
            if priority == 0:
                task_priority = TaskPriority.CRITICAL
            elif priority == 1:
                task_priority = TaskPriority.HIGH
            elif priority == 2:
                task_priority = TaskPriority.MEDIUM
            elif priority == 3:
                task_priority = TaskPriority.LOW
            elif priority >= 4:
                task_priority = TaskPriority.BACKGROUND
            
            task_definitions.append(
                TaskDefinition(
                    id=task_id,
                    func=wrapper_func,
                    input_data=None,
                    priority=task_priority,
                    timeout=timeout
                )
            )
        
        # Execute tasks
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(self.execute_tasks(task_definitions))
        finally:
            loop.close()
        
        # Convert results
        processed_results = {}
        
        for idx, (priority, _, _) in enumerate(tasks):
            task_id = f"task_{idx}"
            for result_id, result in results.items():
                if result_id.startswith(f"task_{idx}_"):
                    if result.success:
                        processed_results[idx] = result.result
                    break
        
        return processed_results


# Create singleton instance
_default_parallel_processor = ParallelProcessor()


def get_parallel_processor() -> ParallelProcessor:
    """
    Get the default parallel processor.
    
    Returns:
        Default parallel processor
    """
    return _default_parallel_processor
