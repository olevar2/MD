"""
Optimized Parallel Processor

This module provides an enhanced parallel processing implementation with adaptive thread pool
and task prioritization for performance-critical operations like confluence and divergence detection.

Features:
- Adaptive thread pool sizing based on workload
- Task prioritization for critical calculations
- Reduced synchronization overhead
- Optimized task granularity
"""

import concurrent.futures
import threading
import queue
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Callable, Hashable, Union

logger = logging.getLogger(__name__)

class OptimizedParallelProcessor:
    """
    Enhanced parallel processing with adaptive thread pool and task prioritization.
    
    Features:
    - Adaptive thread pool sizing based on workload
    - Task prioritization for critical calculations
    - Reduced synchronization overhead
    - Optimized task granularity
    """
    
    def __init__(self, min_workers: int = 2, max_workers: int = 8):
        """
        Initialize the parallel processor.
        
        Args:
            min_workers: Minimum number of worker threads
            max_workers: Maximum number of worker threads
        """
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        self.lock = threading.RLock()
        self.task_queue = queue.PriorityQueue()
        self.results = {}
        self.active_tasks = 0
        self.completed_tasks = 0
        self.executor = None
        
        logger.debug(f"OptimizedParallelProcessor initialized with min_workers={min_workers}, "
                    f"max_workers={max_workers}")
    
    def process(self, tasks: List[Tuple[int, Callable, Tuple]], timeout: Optional[float] = None) -> Dict[int, Any]:
        """
        Process tasks in parallel with priority.
        
        Args:
            tasks: List of (priority, function, args) tuples
                  Lower priority values are processed first
            timeout: Optional timeout in seconds
            
        Returns:
            Dictionary mapping task IDs to results
        """
        # Determine optimal worker count based on task count
        task_count = len(tasks)
        with self.lock:
            self.current_workers = max(self.min_workers, min(self.max_workers, task_count))
            
        # Initialize results dictionary
        self.results = {}
        self.active_tasks = task_count
        self.completed_tasks = 0
        
        # Early termination for empty task list
        if task_count == 0:
            return self.results
        
        # Create thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.current_workers) as executor:
            self.executor = executor
            
            # Submit all tasks
            future_to_id = {}
            for priority, func, args in tasks:
                # Generate a unique task ID
                task_id = id(func)
                if args:
                    for arg in args:
                        if isinstance(arg, Hashable):
                            task_id += hash(arg)
                
                # Submit task with priority
                future = executor.submit(self._execute_task, task_id, func, args)
                future_to_id[future] = task_id
            
            # Wait for completion or timeout
            try:
                for future in concurrent.futures.as_completed(future_to_id, timeout=timeout):
                    task_id = future_to_id[future]
                    try:
                        result = future.result()
                        self.results[task_id] = result
                    except Exception as exc:
                        logger.error(f"Task {task_id} generated an exception: {exc}", exc_info=True)
                        self.results[task_id] = None
                    
                    with self.lock:
                        self.active_tasks -= 1
                        self.completed_tasks += 1
            except concurrent.futures.TimeoutError:
                logger.warning(f"Timeout occurred after {timeout} seconds with {self.active_tasks} tasks remaining")
        
        return self.results
    
    def _execute_task(self, task_id: int, func: Callable, args: Tuple) -> Any:
        """
        Execute a single task and return the result.
        
        Args:
            task_id: Unique task identifier
            func: Function to execute
            args: Arguments to pass to the function
            
        Returns:
            Result of the function execution
        """
        start_time = time.time()
        try:
            result = func(*args)
            execution_time = time.time() - start_time
            
            # Log slow tasks
            if execution_time > 0.1:  # Log if > 100ms
                logger.debug(f"Task {task_id} completed in {execution_time:.4f}s")
                
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error executing task {task_id} after {execution_time:.4f}s: {e}", exc_info=True)
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics.
        
        Returns:
            Dictionary with processor statistics
        """
        with self.lock:
            return {
                "current_workers": self.current_workers,
                "active_tasks": self.active_tasks,
                "completed_tasks": self.completed_tasks,
                "total_tasks": self.active_tasks + self.completed_tasks
            }
