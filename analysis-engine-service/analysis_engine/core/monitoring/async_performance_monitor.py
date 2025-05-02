"""
Async Performance Monitor

This module provides tools for monitoring the performance of asynchronous operations.
"""

import time
import asyncio
import logging
import functools
from typing import Dict, Any, Optional, Callable, List, Awaitable
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class AsyncPerformanceMetrics:
    """Metrics for tracking async operation performance."""
    
    def __init__(self, name: str):
        """
        Initialize metrics for an operation.
        
        Args:
            name: Name of the operation
        """
        self.name = name
        self.call_count = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0
        self.error_count = 0
        self.last_call_time = None
        self.last_error_time = None
        self.last_error = None
        
    def record_call(self, duration: float, error: Optional[Exception] = None):
        """
        Record a call to the operation.
        
        Args:
            duration: Duration of the call in seconds
            error: Exception if the call failed, None otherwise
        """
        self.call_count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.last_call_time = datetime.now()
        
        if error:
            self.error_count += 1
            self.last_error_time = datetime.now()
            self.last_error = str(error)
            
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the current metrics.
        
        Returns:
            Dictionary of metrics
        """
        avg_time = self.total_time / self.call_count if self.call_count > 0 else 0
        error_rate = self.error_count / self.call_count if self.call_count > 0 else 0
        
        return {
            "name": self.name,
            "call_count": self.call_count,
            "total_time": self.total_time,
            "avg_time": avg_time,
            "min_time": self.min_time if self.min_time != float('inf') else 0,
            "max_time": self.max_time,
            "error_count": self.error_count,
            "error_rate": error_rate,
            "last_call_time": self.last_call_time.isoformat() if self.last_call_time else None,
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None,
            "last_error": self.last_error
        }

class AsyncPerformanceMonitor:
    """Monitor for tracking async operation performance."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'AsyncPerformanceMonitor':
        """
        Get the singleton instance of the monitor.
        
        Returns:
            AsyncPerformanceMonitor instance
        """
        if cls._instance is None:
            cls._instance = AsyncPerformanceMonitor()
        return cls._instance
    
    def __init__(self):
        """Initialize the monitor."""
        self.metrics: Dict[str, AsyncPerformanceMetrics] = {}
        self.enabled = True
        self.report_interval = 3600  # 1 hour in seconds
        self._reporting_task = None
        
    def get_metrics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metrics for an operation or all operations.
        
        Args:
            operation_name: Name of the operation, or None for all operations
            
        Returns:
            Dictionary of metrics
        """
        if operation_name:
            if operation_name in self.metrics:
                return self.metrics[operation_name].get_metrics()
            return {}
        
        return {name: metrics.get_metrics() for name, metrics in self.metrics.items()}
    
    def get_operation_metrics(self, operation_name: str) -> AsyncPerformanceMetrics:
        """
        Get or create metrics for an operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            AsyncPerformanceMetrics instance
        """
        if operation_name not in self.metrics:
            self.metrics[operation_name] = AsyncPerformanceMetrics(operation_name)
        return self.metrics[operation_name]
    
    @asynccontextmanager
    async def track_operation(self, operation_name: str):
        """
        Context manager for tracking an async operation.
        
        Args:
            operation_name: Name of the operation
            
        Yields:
            None
        """
        if not self.enabled:
            yield
            return
            
        metrics = self.get_operation_metrics(operation_name)
        start_time = time.perf_counter()
        error = None
        
        try:
            yield
        except Exception as e:
            error = e
            raise
        finally:
            duration = time.perf_counter() - start_time
            metrics.record_call(duration, error)
    
    def track_async_function(self, func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        """
        Decorator for tracking an async function.
        
        Args:
            func: Async function to track
            
        Returns:
            Decorated function
        """
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            operation_name = f"{func.__module__}.{func.__qualname__}"
            
            async with self.track_operation(operation_name):
                return await func(*args, **kwargs)
                
        return wrapper
    
    async def start_reporting(self, interval: Optional[int] = None):
        """
        Start periodic reporting of metrics.
        
        Args:
            interval: Reporting interval in seconds, or None to use default
        """
        if interval is not None:
            self.report_interval = interval
            
        if self._reporting_task is not None:
            self._reporting_task.cancel()
            
        self._reporting_task = asyncio.create_task(self._report_metrics_periodically())
        logger.info(f"Started async performance reporting with interval {self.report_interval}s")
    
    async def stop_reporting(self):
        """Stop periodic reporting of metrics."""
        if self._reporting_task is not None:
            self._reporting_task.cancel()
            try:
                await self._reporting_task
            except asyncio.CancelledError:
                pass
            self._reporting_task = None
            logger.info("Stopped async performance reporting")
    
    async def _report_metrics_periodically(self):
        """Report metrics periodically."""
        try:
            while True:
                await asyncio.sleep(self.report_interval)
                self._log_metrics_report()
        except asyncio.CancelledError:
            logger.debug("Metrics reporting task cancelled")
            raise
    
    def _log_metrics_report(self):
        """Log a report of all metrics."""
        if not self.metrics:
            logger.info("No async operations have been tracked yet")
            return
            
        logger.info(f"Async Performance Report - {len(self.metrics)} operations tracked")
        
        # Sort operations by total time (descending)
        sorted_metrics = sorted(
            [m.get_metrics() for m in self.metrics.values()],
            key=lambda m: m["total_time"],
            reverse=True
        )
        
        for metrics in sorted_metrics[:10]:  # Top 10 by total time
            logger.info(
                f"{metrics['name']}: "
                f"calls={metrics['call_count']}, "
                f"avg={metrics['avg_time']:.6f}s, "
                f"min={metrics['min_time']:.6f}s, "
                f"max={metrics['max_time']:.6f}s, "
                f"errors={metrics['error_count']} ({metrics['error_rate']:.2%})"
            )
            
        if len(sorted_metrics) > 10:
            logger.info(f"... and {len(sorted_metrics) - 10} more operations")

# Convenience functions
def get_async_monitor() -> AsyncPerformanceMonitor:
    """
    Get the singleton instance of the async performance monitor.
    
    Returns:
        AsyncPerformanceMonitor instance
    """
    return AsyncPerformanceMonitor.get_instance()

async def track_async_operation(operation_name: str):
    """
    Context manager for tracking an async operation.
    
    Args:
        operation_name: Name of the operation
        
    Returns:
        Async context manager
    """
    return get_async_monitor().track_operation(operation_name)

def track_async_function(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """
    Decorator for tracking an async function.
    
    Args:
        func: Async function to track
        
    Returns:
        Decorated function
    """
    return get_async_monitor().track_async_function(func)
