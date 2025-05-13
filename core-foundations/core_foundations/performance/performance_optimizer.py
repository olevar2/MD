"""
Performance Optimization System implementation for Phase 1.
Provides automatic performance analysis, monitoring, and optimization capabilities.
"""

import time
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import statistics
from .adaptive_resource_manager import AdaptiveResourceManager
from .multi_level_cache import MultiLevelCache

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """Collects and analyzes performance metrics."""
    
    def __init__(self, history_size: int = 1000):
    """
      init  .
    
    Args:
        history_size: Description of history_size
    
    """

        self.history_size = history_size
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.last_optimization: Dict[str, datetime] = {}
        self._lock = threading.Lock()

    def record_operation_time(self, operation: str, duration: float) -> None:
        """Record the duration of an operation."""
        with self._lock:
            self.operation_times[operation].append(duration)
            if len(self.operation_times[operation]) > self.history_size:
                self.operation_times[operation].pop(0)

    def record_error(self, operation: str) -> None:
        """Record an operation error."""
        with self._lock:
            self.error_counts[operation] += 1

    def get_statistics(self, operation: str) -> Dict[str, float]:
        """Get performance statistics for an operation."""
        with self._lock:
            times = self.operation_times.get(operation, [])
            if not times:
                return {}

            return {
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "p95": statistics.quantiles(times, n=20)[18],  # 95th percentile
                "min": min(times),
                "max": max(times),
                "error_rate": self.error_counts[operation] / len(times)
            }

class PerformanceOptimizer:
    """
    Coordinates performance monitoring, analysis, and automatic optimization.
    """
    
    def __init__(
        self,
        resource_manager: AdaptiveResourceManager,
        cache_manager: MultiLevelCache,
        config: Optional[Dict[str, Any]] = None
    ):
    """
      init  .
    
    Args:
        resource_manager: Description of resource_manager
        cache_manager: Description of cache_manager
        config: Description of config
        Any]]: Description of Any]]
    
    """

        self.resource_manager = resource_manager
        self.cache_manager = cache_manager
        self.config = config or {}
        self.metrics = PerformanceMetrics()
        
        # Performance thresholds
        self.response_time_threshold = self.config.get('response_time_threshold', 1.0)  # seconds
        self.error_rate_threshold = self.config.get('error_rate_threshold', 0.05)  # 5%
        
        # Optimization settings
        self.min_optimization_interval = self.config.get('min_optimization_interval', 300)  # 5 minutes
        self.optimization_strategies: Dict[str, Callable] = {
            'cache_optimization': self._optimize_cache,
            'resource_optimization': self._optimize_resources,
            'query_optimization': self._optimize_queries
        }
        
        # Start monitoring
        self._start_monitoring()

    def _start_monitoring(self) -> None:
        """Start the performance monitoring thread."""
        def monitor():
    """
    Monitor.
    
    """

            while True:
                try:
                    self._analyze_performance()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Performance monitoring error: {e}")
                    time.sleep(5)  # Back off on error

        threading.Thread(target=monitor, daemon=True).start()

    def _analyze_performance(self) -> None:
        """Analyze performance metrics and trigger optimizations if needed."""
        for operation in self.metrics.operation_times.keys():
            stats = self.metrics.get_statistics(operation)
            if not stats:
                continue

            needs_optimization = False
            optimization_reason = []

            # Check response time
            if stats['p95'] > self.response_time_threshold:
                needs_optimization = True
                optimization_reason.append(f"High response time: {stats['p95']:.2f}s")

            # Check error rate
            if stats['error_rate'] > self.error_rate_threshold:
                needs_optimization = True
                optimization_reason.append(f"High error rate: {stats['error_rate']:.2%}")

            if needs_optimization:
                self._trigger_optimization(operation, optimization_reason)

    def _trigger_optimization(self, operation: str, reasons: List[str]) -> None:
        """Trigger optimization strategies for an operation."""
        last_optimization = self.metrics.last_optimization.get(operation, datetime.min)
        if datetime.now() - last_optimization < timedelta(seconds=self.min_optimization_interval):
            return

        logger.info(f"Triggering optimization for {operation}: {', '.join(reasons)}")
        
        # Apply optimization strategies
        for strategy_name, strategy_func in self.optimization_strategies.items():
            try:
                strategy_func(operation)
            except Exception as e:
                logger.error(f"Error in {strategy_name} for {operation}: {e}")

        self.metrics.last_optimization[operation] = datetime.now()

    def _optimize_cache(self, operation: str) -> None:
        """Optimize cache settings for an operation."""
        stats = self.metrics.get_statistics(operation)
        cache_metrics = self.cache_manager.get_metrics()

        # Adjust cache levels based on access patterns
        if stats['mean'] < 0.1:  # Fast operation
            # Prioritize L1 cache
            self.cache_manager.put(f"{operation}_config", {
                "preferred_level": "l1_memory",
                "ttl": 3600  # 1 hour
            })
        elif stats['mean'] < 1.0:  # Medium operation
            # Use L2 cache
            self.cache_manager.put(f"{operation}_config", {
                "preferred_level": "l2_disk",
                "ttl": 7200  # 2 hours
            })
        else:  # Slow operation
            # Use L3 cache
            self.cache_manager.put(f"{operation}_config", {
                "preferred_level": "l3_database",
                "ttl": 86400  # 24 hours
            })

    def _optimize_resources(self, operation: str) -> None:
        """Optimize resource allocation for an operation."""
        system_status = self.resource_manager.get_system_status()
        stats = self.metrics.get_statistics(operation)

        # Adjust thread pool size based on operation characteristics
        if stats['mean'] > self.response_time_threshold:
            current_threads = system_status['thread_pool_size']
            if system_status['resources']['cpu_avg'] < 70:  # CPU has headroom
                new_thread_count = min(current_threads + 2, system_status['max_threads'])
                self.resource_manager.load_balancer.max_threads = new_thread_count
                logger.info(f"Increased thread pool size to {new_thread_count} for {operation}")

    def _optimize_queries(self, operation: str) -> None:
        """Optimize database queries for an operation."""
        # This would integrate with your specific database optimization strategy
        # For now, we'll just log that optimization would occur
        logger.info(f"Would optimize queries for {operation}")

    def wrap_operation(self, operation: str) -> Callable:
        """
        Decorator to automatically track operation performance.
        
        Usage:
            @optimizer.wrap_operation("my_operation")
            def my_function():
                # Function code
        """
        def decorator(func: Callable) -> Callable:
    """
    Decorator.
    
    Args:
        func: Description of func
    
    Returns:
        Callable: Description of return value
    
    """

            def wrapper(*args, **kwargs) -> Any:
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    self.metrics.record_operation_time(operation, duration)
                    return result
                except Exception as e:
                    self.metrics.record_error(operation)
                    raise
            return wrapper
        return decorator

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        report = {
            "operations": {},
            "system_status": self.resource_manager.get_system_status(),
            "cache_metrics": self.cache_manager.get_metrics()
        }

        for operation in self.metrics.operation_times.keys():
            report["operations"][operation] = {
                "statistics": self.metrics.get_statistics(operation),
                "error_count": self.metrics.error_counts[operation],
                "last_optimization": self.metrics.last_optimization.get(operation)
            }

        return report
