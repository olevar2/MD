"""
Performance Monitoring Module

This module provides a standardized performance monitoring system for the forex trading platform.
It includes performance tracking, metrics collection, and performance optimization utilities.

Features:
- Operation performance tracking
- Resource usage monitoring
- Performance optimization
- Metrics collection
- Performance reporting
"""

import os
import time
import logging
import threading
import functools
import traceback
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, TypeVar, cast
from datetime import datetime, timedelta
from contextlib import contextmanager
from enum import Enum
import json
import statistics

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from prometheus_client import Counter, Gauge, Histogram, Summary

from common_lib.errors.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    BaseError,
    ServiceError
)

# Type variable for function
F = TypeVar('F', bound=Callable[..., Any])

# Create logger
logger = logging.getLogger(__name__)


class PerformanceMetricType(Enum):
    """Performance metric types."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    CUSTOM = "custom"


class PerformanceMonitor:
    """
    Performance monitoring for critical operations with detailed metrics tracking.
    
    This class provides a standardized way to monitor performance of critical operations
    across the forex trading platform. It includes performance tracking, metrics collection,
    and performance optimization utilities.
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'PerformanceMonitor':
        """
        Get the singleton instance of the performance monitor.
        
        Returns:
            PerformanceMonitor instance
        """
        if cls._instance is None:
            cls._instance = PerformanceMonitor()
        return cls._instance
    
    def __init__(
        self,
        service_name: Optional[str] = None,
        enable_resource_monitoring: bool = True,
        resource_monitoring_interval: float = 1.0,
        metrics_capacity: int = 1000,
        latency_buckets: Optional[List[float]] = None
    ):
        """
        Initialize the performance monitor.
        
        Args:
            service_name: Name of the service
            enable_resource_monitoring: Whether to enable resource monitoring
            resource_monitoring_interval: Interval for resource monitoring in seconds
            metrics_capacity: Maximum number of metrics to store
            latency_buckets: Buckets for latency histograms
        """
        self.service_name = service_name or os.environ.get("SERVICE_NAME", "unknown")
        self.enable_resource_monitoring = enable_resource_monitoring and PSUTIL_AVAILABLE
        self.resource_monitoring_interval = resource_monitoring_interval
        self.metrics_capacity = metrics_capacity
        self.latency_buckets = latency_buckets or [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
        
        # Initialize metrics storage
        self.operation_metrics: Dict[str, Dict[str, Any]] = {}
        self.resource_metrics: Dict[str, List[float]] = {
            "cpu_usage": [],
            "memory_usage": [],
            "disk_usage": [],
            "network_usage": []
        }
        
        # Initialize Prometheus metrics
        self._init_prometheus_metrics()
        
        # Start resource monitoring if enabled
        self._resource_monitoring_thread = None
        self._stop_resource_monitoring = threading.Event()
        if self.enable_resource_monitoring:
            self._start_resource_monitoring()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        # Operation metrics
        self.operation_latency = Histogram(
            "operation_latency_seconds",
            "Operation latency in seconds",
            ["service", "component", "operation"],
            buckets=self.latency_buckets
        )
        
        self.operation_count = Counter(
            "operation_count_total",
            "Total number of operations",
            ["service", "component", "operation", "status"]
        )
        
        self.operation_error_count = Counter(
            "operation_error_count_total",
            "Total number of operation errors",
            ["service", "component", "operation", "error_type"]
        )
        
        # Resource metrics
        self.cpu_usage = Gauge(
            "cpu_usage_percent",
            "CPU usage in percent",
            ["service"]
        )
        
        self.memory_usage = Gauge(
            "memory_usage_bytes",
            "Memory usage in bytes",
            ["service"]
        )
        
        self.disk_usage = Gauge(
            "disk_usage_bytes",
            "Disk usage in bytes",
            ["service", "path"]
        )
        
        self.network_usage = Gauge(
            "network_usage_bytes",
            "Network usage in bytes",
            ["service", "direction"]
        )
    
    def _start_resource_monitoring(self):
        """Start resource monitoring thread."""
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available, resource monitoring disabled")
            return
        
        self._resource_monitoring_thread = threading.Thread(
            target=self._resource_monitoring_loop,
            daemon=True
        )
        self._resource_monitoring_thread.start()
    
    def _resource_monitoring_loop(self):
        """Resource monitoring loop."""
        if not PSUTIL_AVAILABLE:
            return
        
        process = psutil.Process(os.getpid())
        
        while not self._stop_resource_monitoring.is_set():
            try:
                # Get CPU usage
                cpu_percent = process.cpu_percent()
                self.resource_metrics["cpu_usage"].append(cpu_percent)
                self.cpu_usage.labels(service=self.service_name).set(cpu_percent)
                
                # Get memory usage
                memory_info = process.memory_info()
                memory_usage = memory_info.rss
                self.resource_metrics["memory_usage"].append(memory_usage)
                self.memory_usage.labels(service=self.service_name).set(memory_usage)
                
                # Get disk usage
                disk_usage = 0
                for disk in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(disk.mountpoint)
                        disk_usage += usage.used
                        self.disk_usage.labels(
                            service=self.service_name,
                            path=disk.mountpoint
                        ).set(usage.used)
                    except (PermissionError, FileNotFoundError):
                        pass
                self.resource_metrics["disk_usage"].append(disk_usage)
                
                # Get network usage
                network_usage = 0
                net_io = psutil.net_io_counters()
                network_usage = net_io.bytes_sent + net_io.bytes_recv
                self.resource_metrics["network_usage"].append(network_usage)
                self.network_usage.labels(
                    service=self.service_name,
                    direction="sent"
                ).set(net_io.bytes_sent)
                self.network_usage.labels(
                    service=self.service_name,
                    direction="recv"
                ).set(net_io.bytes_recv)
                
                # Limit metrics storage
                for metric_name, values in self.resource_metrics.items():
                    if len(values) > self.metrics_capacity:
                        self.resource_metrics[metric_name] = values[-self.metrics_capacity:]
                
                # Sleep for the monitoring interval
                time.sleep(self.resource_monitoring_interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {str(e)}")
                time.sleep(self.resource_monitoring_interval * 5)
    
    def stop_resource_monitoring(self):
        """Stop resource monitoring thread."""
        if self._resource_monitoring_thread is not None:
            self._stop_resource_monitoring.set()
            self._resource_monitoring_thread.join(timeout=5)
            self._resource_monitoring_thread = None
    
    @contextmanager
    def track_operation(
        self,
        component: str,
        operation: str
    ):
        """
        Context manager for tracking operation performance.
        
        Args:
            component: Component name
            operation: Operation name
            
        Yields:
            None
        """
        start_time = time.time()
        operation_key = f"{component}.{operation}"
        
        # Initialize operation metrics if not exists
        if operation_key not in self.operation_metrics:
            self.operation_metrics[operation_key] = {
                "latency": [],
                "count": 0,
                "error_count": 0,
                "last_error": None,
                "last_error_time": None
            }
        
        try:
            # Yield control to the context block
            yield
            
            # Record successful operation
            duration = time.time() - start_time
            self._record_operation(component, operation, duration, success=True)
        except Exception as e:
            # Record failed operation
            duration = time.time() - start_time
            self._record_operation(component, operation, duration, success=False, error=e)
            
            # Re-raise the exception
            raise
    
    def _record_operation(
        self,
        component: str,
        operation: str,
        duration: float,
        success: bool,
        error: Optional[Exception] = None
    ):
        """
        Record operation metrics.
        
        Args:
            component: Component name
            operation: Operation name
            duration: Operation duration in seconds
            success: Whether the operation was successful
            error: Exception if the operation failed
        """
        operation_key = f"{component}.{operation}"
        
        # Update operation metrics
        metrics = self.operation_metrics[operation_key]
        metrics["latency"].append(duration)
        metrics["count"] += 1
        
        if not success:
            metrics["error_count"] += 1
            metrics["last_error"] = str(error)
            metrics["last_error_time"] = datetime.utcnow().isoformat()
        
        # Limit metrics storage
        if len(metrics["latency"]) > self.metrics_capacity:
            metrics["latency"] = metrics["latency"][-self.metrics_capacity:]
        
        # Update Prometheus metrics
        self.operation_latency.labels(
            service=self.service_name,
            component=component,
            operation=operation
        ).observe(duration)
        
        self.operation_count.labels(
            service=self.service_name,
            component=component,
            operation=operation,
            status="success" if success else "error"
        ).inc()
        
        if not success:
            self.operation_error_count.labels(
                service=self.service_name,
                component=component,
                operation=operation,
                error_type=error.__class__.__name__ if error else "unknown"
            ).inc()
    
    def track_function(
        self,
        component: str,
        operation: Optional[str] = None
    ) -> Callable[[F], F]:
        """
        Decorator for tracking function performance.
        
        Args:
            component: Component name
            operation: Operation name (defaults to function name)
            
        Returns:
            Decorated function
        """
        def decorator(func: F) -> F:
    """
    Decorator.
    
    Args:
        func: Description of func
    
    Returns:
        F: Description of return value
    
    """

            nonlocal operation
            if operation is None:
                operation = func.__name__
            
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

                with self.track_operation(component, operation):
                    return func(*args, **kwargs)
            
            return cast(F, wrapper)
        
        return decorator
    
    def get_operation_metrics(
        self,
        component: Optional[str] = None,
        operation: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get operation metrics.
        
        Args:
            component: Component name (optional)
            operation: Operation name (optional)
            
        Returns:
            Operation metrics
        """
        if component and operation:
            operation_key = f"{component}.{operation}"
            if operation_key in self.operation_metrics:
                metrics = self.operation_metrics[operation_key]
                return {
                    "component": component,
                    "operation": operation,
                    "count": metrics["count"],
                    "error_count": metrics["error_count"],
                    "error_rate": metrics["error_count"] / metrics["count"] if metrics["count"] > 0 else 0,
                    "last_error": metrics["last_error"],
                    "last_error_time": metrics["last_error_time"],
                    "latency": {
                        "min": min(metrics["latency"]) if metrics["latency"] else 0,
                        "max": max(metrics["latency"]) if metrics["latency"] else 0,
                        "mean": statistics.mean(metrics["latency"]) if metrics["latency"] else 0,
                        "median": statistics.median(metrics["latency"]) if metrics["latency"] else 0,
                        "p95": statistics.quantiles(metrics["latency"], n=20)[18] if len(metrics["latency"]) >= 20 else None,
                        "p99": statistics.quantiles(metrics["latency"], n=100)[98] if len(metrics["latency"]) >= 100 else None
                    }
                }
            return {}
        
        result = {}
        for operation_key, metrics in self.operation_metrics.items():
            comp, op = operation_key.split(".", 1)
            if component and comp != component:
                continue
            
            result[operation_key] = {
                "component": comp,
                "operation": op,
                "count": metrics["count"],
                "error_count": metrics["error_count"],
                "error_rate": metrics["error_count"] / metrics["count"] if metrics["count"] > 0 else 0,
                "last_error": metrics["last_error"],
                "last_error_time": metrics["last_error_time"],
                "latency": {
                    "min": min(metrics["latency"]) if metrics["latency"] else 0,
                    "max": max(metrics["latency"]) if metrics["latency"] else 0,
                    "mean": statistics.mean(metrics["latency"]) if metrics["latency"] else 0,
                    "median": statistics.median(metrics["latency"]) if metrics["latency"] else 0,
                    "p95": statistics.quantiles(metrics["latency"], n=20)[18] if len(metrics["latency"]) >= 20 else None,
                    "p99": statistics.quantiles(metrics["latency"], n=100)[98] if len(metrics["latency"]) >= 100 else None
                }
            }
        
        return result
    
    def get_resource_metrics(self) -> Dict[str, Any]:
        """
        Get resource metrics.
        
        Returns:
            Resource metrics
        """
        result = {}
        for metric_name, values in self.resource_metrics.items():
            if not values:
                result[metric_name] = {
                    "current": 0,
                    "min": 0,
                    "max": 0,
                    "mean": 0
                }
                continue
            
            result[metric_name] = {
                "current": values[-1],
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values)
            }
        
        return result
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get performance report.
        
        Returns:
            Performance report
        """
        return {
            "service": self.service_name,
            "timestamp": datetime.utcnow().isoformat(),
            "operations": self.get_operation_metrics(),
            "resources": self.get_resource_metrics()
        }
    
    def __del__(self):
        """Destructor."""
        self.stop_resource_monitoring()


# Create singleton instance
performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """
    Get the singleton instance of the performance monitor.
    
    Returns:
        PerformanceMonitor instance
    """
    return performance_monitor


def track_operation(
    component: str,
    operation: str
) -> Callable[[F], F]:
    """
    Decorator for tracking operation performance.
    
    Args:
        component: Component name
        operation: Operation name
        
    Returns:
        Decorated function
    """
    return performance_monitor.track_function(component, operation)


@contextmanager
def track_performance(
    component: str,
    operation: str
):
    """
    Context manager for tracking operation performance.
    
    Args:
        component: Component name
        operation: Operation name
        
    Yields:
        None
    """
    with performance_monitor.track_operation(component, operation):
        yield
