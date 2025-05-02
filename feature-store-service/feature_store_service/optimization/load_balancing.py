"""
Load Balancing System for Complex Indicators

This module implements a load balancing system that distributes computation of complex
indicators across available computational resources to optimize performance.
"""

import os
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
from typing import Dict, List, Any, Callable, Optional, Tuple, Union
import logging
import psutil
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


class ComputationPriority(Enum):
    """Priority levels for computations."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class ComputationTask:
    """Represents a computation task to be executed."""
    
    def __init__(self, func: Callable, args: tuple = None, kwargs: dict = None,
                 priority: ComputationPriority = ComputationPriority.NORMAL,
                 task_id: str = None, timeout: float = None):
        """
        Initialize a computation task.
        
        Args:
            func: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            priority: Task priority
            task_id: Unique task identifier
            timeout: Timeout in seconds
        """
        self.func = func
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.priority = priority
        self.task_id = task_id or f"task_{id(self)}"
        self.timeout = timeout
        self.creation_time = time.time()
    
    def __lt__(self, other):
        """Compare tasks based on priority for queue ordering."""
        if not isinstance(other, ComputationTask):
            return NotImplemented
        return self.priority.value > other.priority.value  # Higher priority first


class ResourceMonitor:
    """Monitors system resources and adjusts computation parameters accordingly."""
    
    def __init__(self, update_interval: float = 5.0):
        """
        Initialize the resource monitor.
        
        Args:
            update_interval: Interval in seconds between resource updates
        """
        self.update_interval = update_interval
        self._last_update = 0
        self._cpu_usage = 0
        self._memory_usage = 0
        self._gpu_memory_usage = 0
        self._has_gpu = self._check_gpu_availability()
        self._stop_event = threading.Event()
        self._monitor_thread = None
    
    def start(self):
        """Start the resource monitoring thread."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
            
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self._monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop(self):
        """Stop the resource monitoring thread."""
        if self._monitor_thread is not None:
            self._stop_event.set()
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
            logger.info("Resource monitoring stopped")
    
    def _monitor_resources(self):
        """Continuously monitor system resources."""
        while not self._stop_event.is_set():
            try:
                # Update CPU usage
                self._cpu_usage = psutil.cpu_percent(interval=0.1)
                
                # Update memory usage
                memory = psutil.virtual_memory()
                self._memory_usage = memory.percent
                
                # Update GPU memory usage if available
                if self._has_gpu:
                    self._update_gpu_usage()
                
                self._last_update = time.time()
                
                # Sleep for the specified interval
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
                time.sleep(1.0)  # Sleep briefly before retrying
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            # Try to import GPU monitoring libraries
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                import tensorflow as tf
                return tf.config.list_physical_devices('GPU')
            except ImportError:
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    device_count = pynvml.nvmlDeviceGetCount()
                    return device_count > 0
                except (ImportError, Exception):
                    return False
    
    def _update_gpu_usage(self):
        """Update GPU memory usage metrics."""
        try:
            # Try using torch first
            try:
                import torch
                if torch.cuda.is_available():
                    total = torch.cuda.get_device_properties(0).total_memory
                    reserved = torch.cuda.memory_reserved(0)
                    allocated = torch.cuda.memory_allocated(0)
                    free = total - reserved
                    self._gpu_memory_usage = (allocated / total) * 100
                    return
            except ImportError:
                pass
                
            # Try using tensorflow
            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    # TensorFlow doesn't expose detailed memory stats easily
                    # Just mark as available
                    self._gpu_memory_usage = 50.0  # Assume 50% usage
                    return
            except ImportError:
                pass
                
            # Try using pynvml
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self._gpu_memory_usage = (info.used / info.total) * 100
                return
            except ImportError:
                pass
                
        except Exception as e:
            logger.warning(f"Error updating GPU usage: {e}")
            self._gpu_memory_usage = 0
    
    @property
    def cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return self._cpu_usage
    
    @property
    def memory_usage(self) -> float:
        """Get current memory usage percentage."""
        return self._memory_usage
    
    @property
    def gpu_memory_usage(self) -> float:
        """Get current GPU memory usage percentage."""
        return self._gpu_memory_usage if self._has_gpu else 0
    
    @property
    def has_gpu(self) -> bool:
        """Check if GPU is available."""
        return self._has_gpu
    
    def get_optimal_executor_count(self) -> int:
        """
        Calculate optimal number of executors based on current resource usage.
        
        Returns:
            Optimal number of executors
        """
        cpu_count = multiprocessing.cpu_count()
        
        # If system is under heavy load, reduce number of executors
        if self._cpu_usage > 80:
            return max(1, cpu_count // 4)
        elif self._cpu_usage > 60:
            return max(1, cpu_count // 2)
        else:
            return max(1, cpu_count - 1)  # Leave one CPU for system


class LoadBalancer:
    """
    Load balancer that distributes computation tasks across available resources.
    
    This class manages a pool of executors and assigns tasks to them based on
    priority and available resources, ensuring optimal performance.
    """
    
    def __init__(self, thread_pool_size: int = None, process_pool_size: int = None,
                 use_processes: bool = True, monitor_resources: bool = True):
        """
        Initialize the load balancer.
        
        Args:
            thread_pool_size: Size of thread pool (None for auto)
            process_pool_size: Size of process pool (None for auto)
            use_processes: Whether to use processes for CPU-intensive tasks
            monitor_resources: Whether to monitor system resources
        """
        self.use_processes = use_processes
        self.monitor_resources = monitor_resources
        
        # Initialize resource monitor
        self.resource_monitor = ResourceMonitor()
        if monitor_resources:
            self.resource_monitor.start()
        
        # Determine pool sizes
        self.cpu_count = multiprocessing.cpu_count()
        self.thread_pool_size = thread_pool_size or self.cpu_count * 2
        self.process_pool_size = process_pool_size or max(1, self.cpu_count - 1)
        
        # Create thread and process pools
        self.thread_pool = ThreadPoolExecutor(max_workers=self.thread_pool_size)
        
        if use_processes:
            self.process_pool = ProcessPoolExecutor(max_workers=self.process_pool_size)
        else:
            self.process_pool = None
        
        # Task queue for prioritization
        self.task_queue = queue.PriorityQueue()
        
        # Track pending and completed tasks
        self.pending_tasks = {}
        self.completed_tasks = {}
        self.task_results = {}
        
        # Start worker thread
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._process_tasks, daemon=True)
        self._worker_thread.start()
        
        logger.info(f"Load balancer initialized with {self.thread_pool_size} threads and "
                  f"{self.process_pool_size if use_processes else 0} processes")
    
    def submit_task(self, func: Callable, args: tuple = None, kwargs: dict = None,
                   priority: ComputationPriority = ComputationPriority.NORMAL,
                   cpu_intensive: bool = False, task_id: str = None,
                   timeout: float = None) -> str:
        """
        Submit a task for execution.
        
        Args:
            func: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            priority: Task priority
            cpu_intensive: Whether task is CPU intensive (use process pool)
            task_id: Unique task identifier (generated if None)
            timeout: Timeout in seconds
            
        Returns:
            Task ID
        """
        task = ComputationTask(
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            task_id=task_id,
            timeout=timeout
        )
        
        # Store task information
        self.pending_tasks[task.task_id] = {
            'task': task,
            'cpu_intensive': cpu_intensive,
            'submitted_at': time.time(),
            'started_at': None,
            'completed_at': None,
            'status': 'pending'
        }
        
        # Add to priority queue
        self.task_queue.put(task)
        
        logger.debug(f"Task submitted: {task.task_id} (priority: {priority.name})")
        return task.task_id
    
    def _process_tasks(self):
        """Process tasks from the queue and submit to appropriate executor."""
        while not self._stop_event.is_set():
            try:
                # Get next task from queue
                task = self.task_queue.get(timeout=1.0)
                
                # Check if task is still valid
                if task.task_id not in self.pending_tasks:
                    logger.debug(f"Task {task.task_id} no longer valid, skipping")
                    self.task_queue.task_done()
                    continue
                
                # Update task status
                task_info = self.pending_tasks[task.task_id]
                task_info['status'] = 'running'
                task_info['started_at'] = time.time()
                
                # Choose executor based on task characteristics
                cpu_intensive = task_info['cpu_intensive']
                
                if cpu_intensive and self.use_processes and self.process_pool:
                    executor = self.process_pool
                    logger.debug(f"Submitting task {task.task_id} to process pool")
                else:
                    executor = self.thread_pool
                    logger.debug(f"Submitting task {task.task_id} to thread pool")
                
                # Submit to executor
                future = executor.submit(self._execute_task, task)
                future.add_done_callback(lambda f, tid=task.task_id: self._task_completed(tid, f))
                
                # Mark task as done in queue
                self.task_queue.task_done()
                
            except queue.Empty:
                # No tasks in queue, continue waiting
                continue
            except Exception as e:
                logger.error(f"Error processing task: {e}")
                time.sleep(0.1)
    
    def _execute_task(self, task: ComputationTask) -> Tuple[Any, str, Exception]:
        """
        Execute a task with timeout handling.
        
        Args:
            task: Task to execute
            
        Returns:
            Tuple of (result, status, exception)
        """
        result = None
        status = 'completed'
        exception = None
        
        try:
            # Execute task with timeout if specified
            if task.timeout is not None:
                # Simple timeout implementation
                start_time = time.time()
                while time.time() - start_time < task.timeout:
                    result = task.func(*task.args, **task.kwargs)
                    return result, status, None
                
                # If we get here, timeout occurred
                status = 'timeout'
                exception = TimeoutError(f"Task {task.task_id} timed out after {task.timeout}s")
            else:
                # No timeout, just execute
                result = task.func(*task.args, **task.kwargs)
                
        except Exception as e:
            status = 'error'
            exception = e
            
        return result, status, exception
    
    def _task_completed(self, task_id: str, future):
        """Handle task completion."""
        try:
            # Check if task exists
            if task_id not in self.pending_tasks:
                logger.warning(f"Completed task {task_id} not found in pending tasks")
                return
            
            # Get result and status
            result, status, exception = future.result()
            
            # Update task information
            task_info = self.pending_tasks.pop(task_id)
            task_info['status'] = status
            task_info['completed_at'] = time.time()
            task_info['result'] = result
            task_info['exception'] = exception
            
            # Store in completed tasks
            self.completed_tasks[task_id] = task_info
            
            # Store result
            self.task_results[task_id] = {
                'result': result,
                'status': status,
                'exception': exception
            }
            
            log_level = logging.DEBUG
            if status != 'completed':
                log_level = logging.ERROR
            
            duration = task_info['completed_at'] - task_info['started_at']
            logger.log(log_level, f"Task {task_id} {status} in {duration:.3f}s")
            
        except Exception as e:
            logger.error(f"Error handling task completion for {task_id}: {e}")
    
    def get_task_result(self, task_id: str, wait: bool = True, timeout: float = None) -> Dict[str, Any]:
        """
        Get the result of a task.
        
        Args:
            task_id: Task identifier
            wait: Whether to wait for task completion
            timeout: Maximum time to wait in seconds
            
        Returns:
            Dictionary with task result information
        """
        start_time = time.time()
        
        while wait and task_id not in self.task_results:
            if timeout is not None and time.time() - start_time > timeout:
                return {
                    'result': None,
                    'status': 'wait_timeout',
                    'exception': TimeoutError(f"Waiting for task {task_id} timed out")
                }
            time.sleep(0.1)
        
        # Return result if available
        if task_id in self.task_results:
            return self.task_results[task_id]
        
        # Task not found
        return {
            'result': None,
            'status': 'not_found',
            'exception': ValueError(f"Task {task_id} not found")
        }
    
    def wait_for_tasks(self, task_ids: List[str], timeout: float = None) -> Dict[str, Dict[str, Any]]:
        """
        Wait for multiple tasks to complete.
        
        Args:
            task_ids: List of task identifiers
            timeout: Maximum time to wait in seconds
            
        Returns:
            Dictionary of task ID to result information
        """
        results = {}
        remaining_tasks = set(task_ids)
        start_time = time.time()
        
        while remaining_tasks:
            # Check for timeout
            if timeout is not None and time.time() - start_time > timeout:
                # Add timeout status for remaining tasks
                for task_id in remaining_tasks:
                    results[task_id] = {
                        'result': None,
                        'status': 'wait_timeout',
                        'exception': TimeoutError(f"Waiting for task {task_id} timed out")
                    }
                break
            
            # Check for completed tasks
            for task_id in list(remaining_tasks):
                if task_id in self.task_results:
                    results[task_id] = self.task_results[task_id]
                    remaining_tasks.remove(task_id)
            
            # Sleep briefly before checking again
            if remaining_tasks:
                time.sleep(0.1)
        
        # Add any missing results (should not happen, but just in case)
        for task_id in task_ids:
            if task_id not in results:
                results[task_id] = self.get_task_result(task_id, wait=False)
        
        return results
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if task was cancelled, False otherwise
        """
        # Check if task exists and is still pending
        if task_id in self.pending_tasks and self.pending_tasks[task_id]['status'] == 'pending':
            # Mark as cancelled in pending tasks
            self.pending_tasks[task_id]['status'] = 'cancelled'
            
            # Add to completed tasks
            self.completed_tasks[task_id] = self.pending_tasks.pop(task_id)
            
            # Add to results
            self.task_results[task_id] = {
                'result': None,
                'status': 'cancelled',
                'exception': None
            }
            
            logger.info(f"Task {task_id} cancelled")
            return True
        
        logger.debug(f"Cannot cancel task {task_id}: not found or not pending")
        return False
    
    def shutdown(self, wait: bool = True):
        """
        Shut down the load balancer and release resources.
        
        Args:
            wait: Whether to wait for pending tasks to complete
        """
        # Stop worker thread
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        
        # Shut down executors
        self.thread_pool.shutdown(wait=wait)
        if self.process_pool:
            self.process_pool.shutdown(wait=wait)
        
        # Stop resource monitor
        if self.monitor_resources:
            self.resource_monitor.stop()
        
        logger.info("Load balancer shut down")


# Global instance for convenience
load_balancer = None

def initialize_load_balancer(thread_pool_size: int = None, process_pool_size: int = None,
                           use_processes: bool = True, monitor_resources: bool = True) -> LoadBalancer:
    """
    Initialize the global load balancer.
    
    Args:
        thread_pool_size: Size of thread pool (None for auto)
        process_pool_size: Size of process pool (None for auto)
        use_processes: Whether to use processes for CPU-intensive tasks
        monitor_resources: Whether to monitor system resources
        
    Returns:
        Load balancer instance
    """
    global load_balancer
    
    if load_balancer is None:
        load_balancer = LoadBalancer(
            thread_pool_size=thread_pool_size,
            process_pool_size=process_pool_size,
            use_processes=use_processes,
            monitor_resources=monitor_resources
        )
    
    return load_balancer

def get_load_balancer() -> LoadBalancer:
    """
    Get the global load balancer instance.
    
    Returns:
        Load balancer instance
    
    Raises:
        RuntimeError: If load balancer not initialized
    """
    global load_balancer
    
    if load_balancer is None:
        # Auto-initialize with default settings
        load_balancer = initialize_load_balancer()
    
    return load_balancer
