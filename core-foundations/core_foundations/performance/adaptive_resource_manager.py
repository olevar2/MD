"""
Adaptive Resource Management System implementation for Phase 1.
Provides real-time resource monitoring, intelligent load distribution,
and automatic resource optimization.
"""

import psutil
import threading
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ResourceMetrics:
    """Tracks and analyzes system resource usage."""
    
    def __init__(self, history_size: int = 3600):  # 1 hour of history at 1 sample/second
        self.history_size = history_size
        self.cpu_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.io_history = deque(maxlen=history_size)
        self._lock = threading.Lock()

    def record_metrics(self) -> Dict[str, float]:
        """Record current system metrics."""
        with self._lock:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_percent = psutil.virtual_memory().percent
            io_counters = psutil.disk_io_counters()
            
            self.cpu_history.append((time.time(), cpu_percent))
            self.memory_history.append((time.time(), memory_percent))
            self.io_history.append((time.time(), io_counters))

            return {
                "cpu": cpu_percent,
                "memory": memory_percent,
                "disk_io": io_counters.read_bytes + io_counters.write_bytes
            }

    def get_average_metrics(self, duration_seconds: int = 300) -> Dict[str, float]:
        """Get average metrics over the specified duration."""
        with self._lock:
            cutoff_time = time.time() - duration_seconds
            
            # Calculate CPU average
            recent_cpu = [(t, v) for t, v in self.cpu_history if t >= cutoff_time]
            cpu_avg = sum(v for _, v in recent_cpu) / len(recent_cpu) if recent_cpu else 0

            # Calculate memory average
            recent_memory = [(t, v) for t, v in self.memory_history if t >= cutoff_time]
            memory_avg = sum(v for _, v in recent_memory) / len(recent_memory) if recent_memory else 0

            return {
                "cpu_avg": cpu_avg,
                "memory_avg": memory_avg
            }

class LoadBalancer:
    """Manages task distribution and resource allocation."""
    
    def __init__(self, max_threads: Optional[int] = None):
        self.max_threads = max_threads or (psutil.cpu_count() * 2)
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.thread_pool = []
        self._lock = threading.Lock()

    def submit_task(self, task_id: str, priority: int = 1, 
                   func: Callable[..., Any] = None, *args, **kwargs) -> Any:
        """Submit a task for execution with specified priority."""
        with self._lock:
            if len(self.thread_pool) >= self.max_threads:
                self._wait_for_available_thread()

            task_info = {
                "id": task_id,
                "priority": priority,
                "start_time": time.time(),
                "status": "running"
            }
            
            self.active_tasks[task_id] = task_info
            
            thread = threading.Thread(
                target=self._execute_task,
                args=(task_id, func, args, kwargs)
            )
            self.thread_pool.append(thread)
            thread.start()

            return task_id

    def _execute_task(self, task_id: str, func: Callable[..., Any], 
                     args: tuple, kwargs: dict) -> None:
        """Execute a task and update its status."""
        try:
            result = func(*args, **kwargs)
            self.active_tasks[task_id]["result"] = result
            self.active_tasks[task_id]["status"] = "completed"
        except Exception as e:
            logger.error(f"Task {task_id} failed: {str(e)}")
            self.active_tasks[task_id]["status"] = "failed"
            self.active_tasks[task_id]["error"] = str(e)
        finally:
            with self._lock:
                self.thread_pool = [t for t in self.thread_pool if t.is_alive()]

    def _wait_for_available_thread(self) -> None:
        """Wait for a thread to become available."""
        while len(self.thread_pool) >= self.max_threads:
            self.thread_pool = [t for t in self.thread_pool if t.is_alive()]
            if len(self.thread_pool) >= self.max_threads:
                time.sleep(0.1)

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific task."""
        return self.active_tasks.get(task_id)

class AdaptiveResourceManager:
    """
    Coordinates resource monitoring, load balancing, and optimization.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.resource_metrics = ResourceMetrics()
        self.load_balancer = LoadBalancer()
        
        # Resource thresholds
        self.cpu_threshold = self.config.get('cpu_threshold', 80)  # 80% CPU threshold
        self.memory_threshold = self.config.get('memory_threshold', 85)  # 85% memory threshold
        
        # Start monitoring
        self._start_monitoring()

    def _start_monitoring(self) -> None:
        """Start the resource monitoring thread."""
        def monitor():
            while True:
                try:
                    metrics = self.resource_metrics.record_metrics()
                    self._check_thresholds(metrics)
                    time.sleep(1)  # 1 second interval
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(5)  # Back off on error

        threading.Thread(target=monitor, daemon=True).start()

    def _check_thresholds(self, metrics: Dict[str, float]) -> None:
        """Check if resource usage exceeds thresholds and take action."""
        if metrics['cpu'] > self.cpu_threshold:
            logger.warning(f"High CPU usage: {metrics['cpu']}%")
            self._optimize_cpu_usage()

        if metrics['memory'] > self.memory_threshold:
            logger.warning(f"High memory usage: {metrics['memory']}%")
            self._optimize_memory_usage()

    def _optimize_cpu_usage(self) -> None:
        """Implement CPU optimization strategies."""
        # Reduce thread pool size temporarily
        self.load_balancer.max_threads = max(2, self.load_balancer.max_threads - 2)
        
        # Schedule thread pool size restoration
        def restore_thread_pool():
            time.sleep(300)  # Wait 5 minutes
            self.load_balancer.max_threads = psutil.cpu_count() * 2
        
        threading.Thread(target=restore_thread_pool, daemon=True).start()

    def _optimize_memory_usage(self) -> None:
        """Implement memory optimization strategies."""
        # Trigger garbage collection
        import gc
        gc.collect()

    def submit_task(self, task_id: str, func: Callable[..., Any], 
                   *args, **kwargs) -> str:
        """Submit a task for execution with automatic resource management."""
        # Check resource availability before submitting
        metrics = self.resource_metrics.get_average_metrics(60)  # Last minute average
        
        # Adjust priority based on resource usage
        priority = 1
        if metrics['cpu_avg'] > self.cpu_threshold:
            priority = 2  # Lower priority when system is under load
        
        return self.load_balancer.submit_task(task_id, priority, func, *args, **kwargs)

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and metrics."""
        metrics = self.resource_metrics.get_average_metrics()
        active_tasks = len([t for t in self.load_balancer.active_tasks.values() 
                          if t['status'] == 'running'])
        
        return {
            "resources": metrics,
            "active_tasks": active_tasks,
            "thread_pool_size": len(self.load_balancer.thread_pool),
            "max_threads": self.load_balancer.max_threads
        }
