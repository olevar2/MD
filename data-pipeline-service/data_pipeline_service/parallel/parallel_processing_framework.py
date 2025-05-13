"""
Parallel Processing Framework for Data Pipeline Service.

This module provides a comprehensive framework for parallel processing of data
in the data pipeline service, supporting both thread-based and process-based
parallelism with appropriate error handling and resource management.

Features:
- Dynamic selection between thread and process-based parallelism
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
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar, Union
import pandas as pd
from common_lib.exceptions import DataProcessingError
T = TypeVar('T')
R = TypeVar('R')
logger = logging.getLogger(__name__)


from data_pipeline_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ParallelizationMethod(str, Enum):
    """Enum defining parallelization methods."""
    THREAD = 'thread'
    PROCESS = 'process'
    AUTO = 'auto'


class TaskPriority(int, Enum):
    """Enum defining task priorities."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class TaskDefinition(Generic[T, R]):
    """Definition of a task to be executed in parallel."""
    id: str
    func: Callable[[T], R]
    input_data: T
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: Optional[Set[str]] = None
    parallelization_method: ParallelizationMethod = ParallelizationMethod.AUTO
    timeout: Optional[float] = None

    def __post_init__(self):
        """Initialize dependencies if None."""
        if self.dependencies is None:
            self.dependencies = set()


@dataclass
class TaskResult(Generic[R]):
    """Result of a task execution."""
    task_id: str
    result: Optional[R] = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    success: bool = True

    @property
    def failed(self) ->bool:
        """Return True if the task failed."""
        return not self.success


class ResourceManager:
    """Manages system resources for parallel processing."""

    def __init__(self, max_cpu_percent: float=80.0, max_memory_percent:
        float=80.0, reserved_cores: int=1):
        """
        Initialize the resource manager.
        
        Args:
            max_cpu_percent: Maximum CPU utilization percentage
            max_memory_percent: Maximum memory utilization percentage
            reserved_cores: Number of CPU cores to reserve for system operations
        """
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self.reserved_cores = reserved_cores
        self.total_cores = multiprocessing.cpu_count()

    @with_exception_handling
    def get_available_workers(self, method: ParallelizationMethod) ->int:
        """
        Get the number of available worker threads or processes.
        
        Args:
            method: Parallelization method (thread or process)
            
        Returns:
            Number of available workers
        """
        if method == ParallelizationMethod.THREAD:
            return max(1, min(self.total_cores * 2, 32))
        available_cores = max(1, self.total_cores - self.reserved_cores)
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            if cpu_percent > self.max_cpu_percent:
                available_cores = max(1, available_cores // 2)
            if memory_percent > self.max_memory_percent:
                available_cores = max(1, available_cores - 1)
        except ImportError:
            available_cores = max(1, available_cores - 1)
        return available_cores


class TaskScheduler:
    """Schedules tasks for parallel execution based on priorities and dependencies."""

    def schedule_tasks(self, tasks: List[TaskDefinition]) ->List[List[
        TaskDefinition]]:
        """
        Schedule tasks into execution groups based on dependencies and priorities.
        
        Args:
            tasks: List of task definitions
            
        Returns:
            List of task groups to be executed in sequence
        """
        if not tasks:
            return []
        task_map = {task.id: task for task in tasks}
        dependency_graph = {task.id: task.dependencies for task in tasks}
        no_deps = [task_id for task_id, deps in dependency_graph.items() if
            not deps]
        levels = {}
        self._assign_levels(dependency_graph, no_deps, levels, 0)
        max_level = max(levels.values()) if levels else 0
        level_groups = [[] for _ in range(max_level + 1)]
        for task_id, level in levels.items():
            level_groups[level].append(task_map[task_id])
        for i, group in enumerate(level_groups):
            level_groups[i] = sorted(group, key=lambda t: t.priority.value)
        return [group for group in level_groups if group]

    def _assign_levels(self, graph: Dict[str, Set[str]], current_nodes:
        List[str], levels: Dict[str, int], current_level: int):
        """
        Recursively assign levels to tasks based on dependencies.
        
        Args:
            graph: Dependency graph
            current_nodes: Nodes at the current level
            levels: Dictionary mapping task IDs to levels
            current_level: Current level being processed
        """
        if not current_nodes:
            return
        for node in current_nodes:
            levels[node] = current_level
        next_level_nodes = []
        for node, deps in graph.items():
            if node not in levels:
                if all(dep in levels for dep in deps):
                    next_level_nodes.append(node)
        self._assign_levels(graph, next_level_nodes, levels, current_level + 1)


class ParallelExecutor(Generic[T, R]):
    """Executes tasks in parallel using thread or process pools."""

    def __init__(self, resource_manager: Optional[ResourceManager]=None):
        """
        Initialize the parallel executor.
        
        Args:
            resource_manager: Resource manager for worker allocation
        """
        self.resource_manager = resource_manager or ResourceManager()
        self.scheduler = TaskScheduler()

    async def execute_tasks(self, tasks: List[TaskDefinition[T, R]]) ->Dict[
        str, TaskResult[R]]:
        """
        Execute tasks in parallel with dependency and priority handling.
        
        Args:
            tasks: List of task definitions
            
        Returns:
            Dictionary mapping task IDs to results
        """
        if not tasks:
            return {}
        execution_groups = self.scheduler.schedule_tasks(tasks)
        results = {}
        for group_idx, task_group in enumerate(execution_groups):
            logger.debug(
                f'Executing task group {group_idx + 1}/{len(execution_groups)} with {len(task_group)} tasks'
                )
            thread_tasks = []
            process_tasks = []
            for task in task_group:
                if self._should_use_process(task):
                    process_tasks.append(task)
                else:
                    thread_tasks.append(task)
            if thread_tasks:
                thread_results = await self._execute_with_threads(thread_tasks)
                results.update(thread_results)
            if process_tasks:
                process_results = await self._execute_with_processes(
                    process_tasks)
                results.update(process_results)
        return results

    def _should_use_process(self, task: TaskDefinition) ->bool:
        """
        Determine if a task should use process-based parallelism.
        
        Args:
            task: Task definition
            
        Returns:
            True if the task should use process-based parallelism
        """
        if task.parallelization_method == ParallelizationMethod.THREAD:
            return False
        elif task.parallelization_method == ParallelizationMethod.PROCESS:
            return True
        return False

    @async_with_exception_handling
    async def _execute_with_threads(self, tasks: List[TaskDefinition[T, R]]
        ) ->Dict[str, TaskResult[R]]:
        """
        Execute tasks using a thread pool.
        
        Args:
            tasks: List of task definitions
            
        Returns:
            Dictionary mapping task IDs to results
        """
        results = {}
        max_workers = self.resource_manager.get_available_workers(
            ParallelizationMethod.THREAD)
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers
            ) as executor:
            future_to_task = {}
            for task in tasks:
                future = loop.run_in_executor(executor, self.
                    _execute_task_with_timing, task.func, task.input_data,
                    task.timeout)
                future_to_task[future] = task
            for future in asyncio.as_completed(future_to_task.keys()):
                task = future_to_task[future]
                try:
                    success, result, error, execution_time = await future
                    results[task.id] = TaskResult(task_id=task.id, result=
                        result, error=error, execution_time=execution_time,
                        success=success)
                except Exception as e:
                    logger.error(f'Error executing task {task.id}: {str(e)}')
                    results[task.id] = TaskResult(task_id=task.id, error=e,
                        success=False)
        return results

    @async_with_exception_handling
    async def _execute_with_processes(self, tasks: List[TaskDefinition[T, R]]
        ) ->Dict[str, TaskResult[R]]:
        """
        Execute tasks using a process pool.
        
        Args:
            tasks: List of task definitions
            
        Returns:
            Dictionary mapping task IDs to results
        """
        results = {}
        max_workers = self.resource_manager.get_available_workers(
            ParallelizationMethod.PROCESS)
        loop = asyncio.get_event_loop()
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers
            ) as executor:
            future_to_task = {}
            for task in tasks:
                future = loop.run_in_executor(executor, self.
                    _execute_task_with_timing, task.func, task.input_data,
                    task.timeout)
                future_to_task[future] = task
            for future in asyncio.as_completed(future_to_task.keys()):
                task = future_to_task[future]
                try:
                    success, result, error, execution_time = await future
                    results[task.id] = TaskResult(task_id=task.id, result=
                        result, error=error, execution_time=execution_time,
                        success=success)
                except Exception as e:
                    logger.error(f'Error executing task {task.id}: {str(e)}')
                    results[task.id] = TaskResult(task_id=task.id, error=e,
                        success=False)
        return results

    @staticmethod
    @with_exception_handling
    def _execute_task_with_timing(func: Callable[[T], R], data: T, timeout:
        Optional[float]=None) ->Tuple[bool, Optional[R], Optional[Exception
        ], float]:
        """
        Execute a task with timing and error handling.
        
        Args:
            func: Function to execute
            data: Input data
            timeout: Optional timeout in seconds
            
        Returns:
            Tuple of (success, result, error, execution_time)
        """
        start_time = time.time()
        result = None
        error = None
        success = True
        try:
            if timeout:
                with concurrent.futures.ProcessPoolExecutor(max_workers=1
                    ) as executor:
                    future = executor.submit(func, data)
                    result = future.result(timeout=timeout)
            else:
                result = func(data)
        except concurrent.futures.TimeoutError:
            error = TimeoutError(
                f'Task execution timed out after {timeout} seconds')
            success = False
        except Exception as e:
            error = e
            success = False
        execution_time = time.time() - start_time
        return success, result, error, execution_time
