"""
Resource management and optimization service for indicator calculations.

Provides adaptive resource management, load balancing, and caching
optimization for indicator calculations.
"""
from typing import Dict, Any, List, Optional, Union, Callable
import logging
from datetime import datetime, timedelta
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import pandas as pd
from pathlib import Path
import json
import psutil
import gc
logger = logging.getLogger(__name__)


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ResourceMetrics:
    """Tracks and analyzes resource utilization metrics."""

    def __init__(self, sampling_interval: float=1.0):
    """
      init  .
    
    Args:
        sampling_interval: Description of sampling_interval
    
    """

        self.sampling_interval = sampling_interval
        self.cpu_usage: List[float] = []
        self.memory_usage: List[float] = []
        self.thread_count: List[int] = []
        self.process_count: List[int] = []
        self._lock = threading.Lock()

    def record_metrics(self) ->None:
        """Record current resource utilization metrics."""
        with self._lock:
            self.cpu_usage.append(psutil.cpu_percent(interval=None, percpu=
                True))
            memory = psutil.Process().memory_info()
            self.memory_usage.append(memory.rss / 1024 / 1024)
            self.thread_count.append(threading.active_count())
            self.process_count.append(len(psutil.Process().children()))

    def get_summary(self) ->Dict[str, Any]:
        """Get summary statistics of resource utilization."""
        with self._lock:
            return {'cpu': {'current': np.mean(self.cpu_usage[-1]) if self.
                cpu_usage else 0, 'mean': np.mean([np.mean(cpu) for cpu in
                self.cpu_usage]) if self.cpu_usage else 0, 'max': np.max([
                np.max(cpu) for cpu in self.cpu_usage]) if self.cpu_usage else
                0}, 'memory': {'current': self.memory_usage[-1] if self.
                memory_usage else 0, 'mean': np.mean(self.memory_usage) if
                self.memory_usage else 0, 'max': np.max(self.memory_usage) if
                self.memory_usage else 0}, 'threads': {'current': self.
                thread_count[-1] if self.thread_count else 0, 'mean': np.
                mean(self.thread_count) if self.thread_count else 0, 'max':
                np.max(self.thread_count) if self.thread_count else 0},
                'processes': {'current': self.process_count[-1] if self.
                process_count else 0, 'mean': np.mean(self.process_count) if
                self.process_count else 0, 'max': np.max(self.process_count
                ) if self.process_count else 0}}

    def clear(self) ->None:
        """Clear recorded metrics."""
        with self._lock:
            self.cpu_usage.clear()
            self.memory_usage.clear()
            self.thread_count.clear()
            self.process_count.clear()


class LoadBalancer:
    """Manages workload distribution across available resources."""

    def __init__(self, min_threads: int=2, max_threads: int=None,
        min_processes: int=1, max_processes: int=None):
    """
      init  .
    
    Args:
        min_threads: Description of min_threads
        max_threads: Description of max_threads
        min_processes: Description of min_processes
        max_processes: Description of max_processes
    
    """

        self.min_threads = min_threads
        self.max_threads = max_threads or multiprocessing.cpu_count() * 2
        self.min_processes = min_processes
        self.max_processes = max_processes or multiprocessing.cpu_count()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_threads)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_processes)
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get_executor(self, task_type: str) ->Union[ThreadPoolExecutor,
        ProcessPoolExecutor]:
        """
        Get appropriate executor based on task type.
        
        Args:
            task_type: Type of task ('io_bound' or 'cpu_bound')
            
        Returns:
            Appropriate executor for the task type
        """
        if task_type == 'io_bound':
            return self.thread_pool
        elif task_type == 'cpu_bound':
            return self.process_pool
        else:
            raise ValueError(f'Unknown task type: {task_type}')

    def submit_task(self, task_id: str, task_type: str, func: Callable, *
        args, **kwargs) ->Any:
        """
        Submit a task for execution.
        
        Args:
            task_id: Unique identifier for the task
            task_type: Type of task ('io_bound' or 'cpu_bound')
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Future object representing the task
        """
        executor = self.get_executor(task_type)
        with self._lock:
            self.active_tasks[task_id] = {'type': task_type, 'start_time':
                datetime.utcnow(), 'status': 'running'}
        future = executor.submit(func, *args, **kwargs)
        future.add_done_callback(lambda f: self._task_completed(task_id))
        return future

    def _task_completed(self, task_id: str) ->None:
        """Mark task as completed and update statistics."""
        with self._lock:
            if task_id in self.active_tasks:
                self.active_tasks[task_id]['status'] = 'completed'
                self.active_tasks[task_id]['end_time'] = datetime.utcnow()

    def get_active_tasks(self) ->Dict[str, Dict[str, Any]]:
        """Get information about currently active tasks."""
        with self._lock:
            return {task_id: task_info.copy() for task_id, task_info in
                self.active_tasks.items() if task_info['status'] == 'running'}


class CacheManager:
    """Manages caching of indicator calculations."""

    def __init__(self, cache_dir: str='cache', max_memory_size: int=1024,
        max_cache_age: timedelta=timedelta(hours=1)):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_memory_size = max_memory_size
        self.max_cache_age = max_cache_age
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_stats = {'hits': 0, 'misses': 0, 'evictions': 0}
        self._lock = threading.Lock()

    @with_exception_handling
    def get(self, key: str) ->Optional[Any]:
        """
        Retrieve item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached item if found and valid, None otherwise
        """
        with self._lock:
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if datetime.utcnow() - entry['timestamp'
                    ] <= self.max_cache_age:
                    self.cache_stats['hits'] += 1
                    return entry['data']
                else:
                    del self.memory_cache[key]
            cache_file = self.cache_dir / f'{key}.json'
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        entry = json.load(f)
                        if datetime.fromisoformat(entry['timestamp']
                            ) + self.max_cache_age >= datetime.utcnow():
                            self.memory_cache[key] = {'data': entry['data'],
                                'timestamp': datetime.fromisoformat(entry[
                                'timestamp'])}
                            self.cache_stats['hits'] += 1
                            return entry['data']
                        else:
                            cache_file.unlink()
                except Exception as e:
                    logger.error(f'Error reading cache file: {str(e)}')
            self.cache_stats['misses'] += 1
            return None

    @with_exception_handling
    def put(self, key: str, value: Any) ->None:
        """
        Store item in cache.
        
        Args:
            key: Cache key
            value: Item to cache
        """
        with self._lock:
            timestamp = datetime.utcnow()
            self.memory_cache[key] = {'data': value, 'timestamp': timestamp}
            cache_file = self.cache_dir / f'{key}.json'
            try:
                with open(cache_file, 'w') as f:
                    json.dump({'data': value, 'timestamp': timestamp.
                        isoformat()}, f)
            except Exception as e:
                logger.error(f'Error writing cache file: {str(e)}')
            self._check_memory_usage()

    def _check_memory_usage(self) ->None:
        """Check memory usage and evict items if necessary."""
        current_size = psutil.Process().memory_info().rss / 1024 / 1024
        if current_size > self.max_memory_size:
            sorted_items = sorted(self.memory_cache.items(), key=lambda x:
                x[1]['timestamp'])
            items_to_remove = max(len(sorted_items) // 4, 1)
            for i, (key, _) in enumerate(sorted_items):
                if (i >= items_to_remove and current_size <= self.
                    max_memory_size):
                    break
                if key in self.memory_cache:
                    self.memory_cache[key]['data'] = None
                    del self.memory_cache[key]
                    self.cache_stats['evictions'] += 1
                current_size = psutil.Process().memory_info().rss / 1024 / 1024
            gc.collect()

    @with_exception_handling
    def clear(self) ->None:
        """Clear all cached items."""
        with self._lock:
            self.memory_cache.clear()
            for cache_file in self.cache_dir.glob('*.json'):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.error(f'Error deleting cache file: {str(e)}')
            self.cache_stats = {'hits': 0, 'misses': 0, 'evictions': 0}

    def get_stats(self) ->Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.cache_stats['hits'] + self.cache_stats[
                'misses']
            hit_rate = self.cache_stats['hits'
                ] / total_requests if total_requests > 0 else 0
            return {'hit_rate': hit_rate, 'total_items': len(self.
                memory_cache), 'memory_size_mb': psutil.Process().
                memory_info().rss / 1024 / 1024, **self.cache_stats}


class AdaptiveResourceManager:
    """
    Manages and optimizes resource allocation for indicator calculations.
    
    This class coordinates resource metrics, load balancing, and caching
    to optimize performance of indicator calculations.
    """

    def __init__(self, cache_dir: str='cache', max_memory_size: int=1024,
        max_cache_age: timedelta=timedelta(hours=1)):
        self.resource_metrics = ResourceMetrics()
        self.load_balancer = LoadBalancer()
        self.cache_manager = CacheManager(cache_dir=cache_dir,
            max_memory_size=max_memory_size, max_cache_age=max_cache_age)
        self._start_monitoring()

    def _start_monitoring(self) ->None:
        """Start periodic resource monitoring."""

        def monitor():
    """
    Monitor.
    
    """

            while True:
                self.resource_metrics.record_metrics()
                threading.Event().wait(1.0)
        threading.Thread(target=monitor, daemon=True).start()

    @with_exception_handling
    def submit_calculation(self, calc_id: str, calc_func: Callable, *args,
        cache_key: Optional[str]=None, **kwargs) ->Any:
        """
        Submit an indicator calculation with automatic optimization.
        
        Args:
            calc_id: Unique identifier for the calculation
            calc_func: Calculation function to execute
            *args: Positional arguments for the function
            cache_key: Optional cache key for results
            **kwargs: Keyword arguments for the function
            
        Returns:
            Calculation result or Future object
        """
        if cache_key:
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
        task_type = getattr(calc_func, 'task_type', 'cpu_bound')
        future = self.load_balancer.submit_task(calc_id, task_type,
            calc_func, *args, **kwargs)
        if cache_key:

            @with_exception_handling
            def cache_callback(future):
    """
    Cache callback.
    
    Args:
        future: Description of future
    
    """

                try:
                    result = future.result()
                    self.cache_manager.put(cache_key, result)
                except Exception:
                    pass
            future.add_done_callback(cache_callback)
        return future

    def get_system_status(self) ->Dict[str, Any]:
        """Get comprehensive system status."""
        return {'resources': self.resource_metrics.get_summary(),
            'active_tasks': self.load_balancer.get_active_tasks(),
            'cache_stats': self.cache_manager.get_stats()}

    def cleanup(self) ->None:
        """Cleanup resources and caches."""
        self.cache_manager.clear()
        self.load_balancer.thread_pool.shutdown()
        self.load_balancer.process_pool.shutdown()
