"""
Distributed Computing

This module provides utilities for distributed computing across multiple nodes.
"""

import os
import sys
import time
import uuid
import json
import logging
import asyncio
import threading
import multiprocessing
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, TypeVar, Generic
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')
U = TypeVar('U')

class DistributedTask:
    """Distributed task representation."""
    
    def __init__(
        self,
        task_id: str,
        function_name: str,
        args: List[Any],
        kwargs: Dict[str, Any],
        priority: int = 0,
        timeout: Optional[float] = None
    ):
        """
        Initialize a distributed task.
        
        Args:
            task_id: Task ID
            function_name: Function name to execute
            args: Function arguments
            kwargs: Function keyword arguments
            priority: Task priority (higher values have higher priority)
            timeout: Task timeout in seconds
        """
        self.task_id = task_id
        self.function_name = function_name
        self.args = args
        self.kwargs = kwargs
        self.priority = priority
        self.timeout = timeout
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.error = None
        self.status = "pending"  # pending, running, completed, failed, timeout
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert task to dictionary.
        
        Returns:
            Task as dictionary
        """
        return {
            "task_id": self.task_id,
            "function_name": self.function_name,
            "args": self.args,
            "kwargs": self.kwargs,
            "priority": self.priority,
            "timeout": self.timeout,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistributedTask':
        """
        Create task from dictionary.
        
        Args:
            data: Task data
            
        Returns:
            Distributed task
        """
        task = cls(
            task_id=data["task_id"],
            function_name=data["function_name"],
            args=data["args"],
            kwargs=data["kwargs"],
            priority=data["priority"],
            timeout=data["timeout"]
        )
        
        task.created_at = data["created_at"]
        task.started_at = data.get("started_at")
        task.completed_at = data.get("completed_at")
        task.status = data["status"]
        
        return task

class DistributedWorker:
    """Distributed worker for executing tasks."""
    
    def __init__(
        self,
        worker_id: str,
        functions: Dict[str, Callable],
        max_concurrent_tasks: int = 4,
        use_processes: bool = False
    ):
        """
        Initialize a distributed worker.
        
        Args:
            worker_id: Worker ID
            functions: Dictionary mapping function names to functions
            max_concurrent_tasks: Maximum number of concurrent tasks
            use_processes: Whether to use processes instead of threads
        """
        self.worker_id = worker_id
        self.functions = functions
        self.max_concurrent_tasks = max_concurrent_tasks
        self.use_processes = use_processes
        
        # Create executor
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=max_concurrent_tasks)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        
        # Task queue
        self.task_queue = asyncio.Queue()
        
        # Task storage
        self.tasks = {}
        
        # Worker status
        self.status = "idle"  # idle, running, stopping, stopped
        
        # Worker stats
        self.stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_timeout": 0,
            "total_execution_time": 0
        }
        
        # Worker loop
        self.loop = None
        self.worker_task = None
    
    async def start(self):
        """Start the worker."""
        if self.status != "idle" and self.status != "stopped":
            logger.warning(f"Worker {self.worker_id} is already running")
            return
        
        logger.info(f"Starting worker {self.worker_id}")
        
        # Set status
        self.status = "running"
        
        # Get event loop
        self.loop = asyncio.get_event_loop()
        
        # Start worker task
        self.worker_task = asyncio.create_task(self._worker_loop())
    
    async def stop(self):
        """Stop the worker."""
        if self.status != "running":
            logger.warning(f"Worker {self.worker_id} is not running")
            return
        
        logger.info(f"Stopping worker {self.worker_id}")
        
        # Set status
        self.status = "stopping"
        
        # Wait for worker task to complete
        if self.worker_task:
            await self.worker_task
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Set status
        self.status = "stopped"
    
    async def _worker_loop(self):
        """Worker loop for processing tasks."""
        logger.info(f"Worker {self.worker_id} loop started")
        
        while self.status == "running":
            try:
                # Get task from queue
                task = await self.task_queue.get()
                
                # Process task
                await self._process_task(task)
                
                # Mark task as done
                self.task_queue.task_done()
            except asyncio.CancelledError:
                logger.info(f"Worker {self.worker_id} loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in worker {self.worker_id} loop: {e}")
        
        logger.info(f"Worker {self.worker_id} loop stopped")
    
    async def _process_task(self, task: DistributedTask):
        """
        Process a task.
        
        Args:
            task: Task to process
        """
        logger.debug(f"Processing task {task.task_id}")
        
        # Update task status
        task.status = "running"
        task.started_at = time.time()
        
        # Get function
        function = self.functions.get(task.function_name)
        
        if function is None:
            logger.error(f"Function {task.function_name} not found")
            
            # Update task status
            task.status = "failed"
            task.error = f"Function {task.function_name} not found"
            task.completed_at = time.time()
            
            # Update stats
            self.stats["tasks_failed"] += 1
            
            return
        
        try:
            # Execute function
            if task.timeout:
                # Execute with timeout
                result = await asyncio.wait_for(
                    self.loop.run_in_executor(
                        self.executor,
                        function,
                        *task.args,
                        **task.kwargs
                    ),
                    timeout=task.timeout
                )
            else:
                # Execute without timeout
                result = await self.loop.run_in_executor(
                    self.executor,
                    function,
                    *task.args,
                    **task.kwargs
                )
            
            # Update task status
            task.status = "completed"
            task.result = result
            task.completed_at = time.time()
            
            # Update stats
            self.stats["tasks_completed"] += 1
            self.stats["total_execution_time"] += task.completed_at - task.started_at
            
            logger.debug(f"Task {task.task_id} completed")
        except asyncio.TimeoutError:
            # Update task status
            task.status = "timeout"
            task.error = "Task timed out"
            task.completed_at = time.time()
            
            # Update stats
            self.stats["tasks_timeout"] += 1
            
            logger.warning(f"Task {task.task_id} timed out")
        except Exception as e:
            # Update task status
            task.status = "failed"
            task.error = str(e)
            task.completed_at = time.time()
            
            # Update stats
            self.stats["tasks_failed"] += 1
            
            logger.error(f"Error processing task {task.task_id}: {e}")
    
    async def submit_task(self, task: DistributedTask) -> str:
        """
        Submit a task to the worker.
        
        Args:
            task: Task to submit
            
        Returns:
            Task ID
        """
        # Store task
        self.tasks[task.task_id] = task
        
        # Add task to queue
        await self.task_queue.put(task)
        
        logger.debug(f"Task {task.task_id} submitted to worker {self.worker_id}")
        
        return task.task_id
    
    def get_task(self, task_id: str) -> Optional[DistributedTask]:
        """
        Get a task by ID.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task or None if not found
        """
        return self.tasks.get(task_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get worker statistics.
        
        Returns:
            Worker statistics
        """
        return {
            "worker_id": self.worker_id,
            "status": self.status,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "use_processes": self.use_processes,
            "queue_size": self.task_queue.qsize(),
            "tasks_pending": sum(1 for task in self.tasks.values() if task.status == "pending"),
            "tasks_running": sum(1 for task in self.tasks.values() if task.status == "running"),
            "tasks_completed": self.stats["tasks_completed"],
            "tasks_failed": self.stats["tasks_failed"],
            "tasks_timeout": self.stats["tasks_timeout"],
            "total_execution_time": self.stats["total_execution_time"],
            "avg_execution_time": self.stats["total_execution_time"] / max(1, self.stats["tasks_completed"])
        }

class DistributedTaskManager:
    """Manager for distributed tasks across multiple workers."""
    
    def __init__(
        self,
        functions: Dict[str, Callable],
        num_workers: int = 4,
        max_concurrent_tasks_per_worker: int = 4,
        use_processes: bool = False
    ):
        """
        Initialize a distributed task manager.
        
        Args:
            functions: Dictionary mapping function names to functions
            num_workers: Number of workers
            max_concurrent_tasks_per_worker: Maximum number of concurrent tasks per worker
            use_processes: Whether to use processes instead of threads
        """
        self.functions = functions
        self.num_workers = num_workers
        self.max_concurrent_tasks_per_worker = max_concurrent_tasks_per_worker
        self.use_processes = use_processes
        
        # Create workers
        self.workers = {}
        
        for i in range(num_workers):
            worker_id = f"worker-{i}"
            self.workers[worker_id] = DistributedWorker(
                worker_id=worker_id,
                functions=functions,
                max_concurrent_tasks=max_concurrent_tasks_per_worker,
                use_processes=use_processes
            )
        
        # Task storage
        self.tasks = {}
        
        # Manager status
        self.status = "idle"  # idle, running, stopping, stopped
    
    async def start(self):
        """Start the task manager."""
        if self.status != "idle" and self.status != "stopped":
            logger.warning("Task manager is already running")
            return
        
        logger.info("Starting task manager")
        
        # Set status
        self.status = "running"
        
        # Start workers
        for worker in self.workers.values():
            await worker.start()
    
    async def stop(self):
        """Stop the task manager."""
        if self.status != "running":
            logger.warning("Task manager is not running")
            return
        
        logger.info("Stopping task manager")
        
        # Set status
        self.status = "stopping"
        
        # Stop workers
        for worker in self.workers.values():
            await worker.stop()
        
        # Set status
        self.status = "stopped"
    
    async def submit_task(
        self,
        function_name: str,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        priority: int = 0,
        timeout: Optional[float] = None,
        worker_id: Optional[str] = None
    ) -> str:
        """
        Submit a task to the task manager.
        
        Args:
            function_name: Function name to execute
            args: Function arguments
            kwargs: Function keyword arguments
            priority: Task priority (higher values have higher priority)
            timeout: Task timeout in seconds
            worker_id: Worker ID to submit the task to (if None, select the least busy worker)
            
        Returns:
            Task ID
        """
        # Check if manager is running
        if self.status != "running":
            raise RuntimeError("Task manager is not running")
        
        # Create task
        task_id = str(uuid.uuid4())
        task = DistributedTask(
            task_id=task_id,
            function_name=function_name,
            args=args or [],
            kwargs=kwargs or {},
            priority=priority,
            timeout=timeout
        )
        
        # Store task
        self.tasks[task_id] = task
        
        # Select worker
        if worker_id is None:
            worker_id = self._select_worker()
        
        # Submit task to worker
        worker = self.workers.get(worker_id)
        
        if worker is None:
            raise ValueError(f"Worker {worker_id} not found")
        
        await worker.submit_task(task)
        
        logger.debug(f"Task {task_id} submitted to worker {worker_id}")
        
        return task_id
    
    def _select_worker(self) -> str:
        """
        Select the least busy worker.
        
        Returns:
            Worker ID
        """
        # Get worker stats
        worker_stats = {
            worker_id: worker.get_stats()
            for worker_id, worker in self.workers.items()
        }
        
        # Select worker with the smallest queue size
        return min(
            worker_stats.items(),
            key=lambda x: x[1]["queue_size"]
        )[0]
    
    async def get_task_result(
        self,
        task_id: str,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Get the result of a task.
        
        Args:
            task_id: Task ID
            timeout: Timeout in seconds
            
        Returns:
            Task result
            
        Raises:
            ValueError: If task not found
            TimeoutError: If timeout reached
            RuntimeError: If task failed
        """
        # Get task
        task = self.tasks.get(task_id)
        
        if task is None:
            raise ValueError(f"Task {task_id} not found")
        
        # Wait for task to complete
        start_time = time.time()
        
        while task.status == "pending" or task.status == "running":
            # Check timeout
            if timeout and time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout waiting for task {task_id}")
            
            # Wait a bit
            await asyncio.sleep(0.1)
        
        # Check task status
        if task.status == "failed":
            raise RuntimeError(f"Task {task_id} failed: {task.error}")
        elif task.status == "timeout":
            raise TimeoutError(f"Task {task_id} timed out")
        
        # Return result
        return task.result
    
    def get_task(self, task_id: str) -> Optional[DistributedTask]:
        """
        Get a task by ID.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task or None if not found
        """
        return self.tasks.get(task_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get task manager statistics.
        
        Returns:
            Task manager statistics
        """
        # Get worker stats
        worker_stats = {
            worker_id: worker.get_stats()
            for worker_id, worker in self.workers.items()
        }
        
        # Calculate aggregated stats
        tasks_completed = sum(stats["tasks_completed"] for stats in worker_stats.values())
        tasks_failed = sum(stats["tasks_failed"] for stats in worker_stats.values())
        tasks_timeout = sum(stats["tasks_timeout"] for stats in worker_stats.values())
        total_execution_time = sum(stats["total_execution_time"] for stats in worker_stats.values())
        
        return {
            "status": self.status,
            "num_workers": self.num_workers,
            "max_concurrent_tasks_per_worker": self.max_concurrent_tasks_per_worker,
            "use_processes": self.use_processes,
            "tasks_pending": sum(1 for task in self.tasks.values() if task.status == "pending"),
            "tasks_running": sum(1 for task in self.tasks.values() if task.status == "running"),
            "tasks_completed": tasks_completed,
            "tasks_failed": tasks_failed,
            "tasks_timeout": tasks_timeout,
            "total_execution_time": total_execution_time,
            "avg_execution_time": total_execution_time / max(1, tasks_completed),
            "workers": worker_stats
        }
