"""
Metrics Collection Module for Forex Trading Platform.

This module provides metrics collection capabilities using Prometheus,
allowing for monitoring of service performance and behavior.
"""

import os
import time
import logging
import functools
import threading
from typing import Dict, Any, Optional, Callable, List, Union, TypeVar, cast
import inspect
from contextlib import contextmanager
import psutil
import gc

# Prometheus imports
from prometheus_client import Counter, Gauge, Histogram, Summary
from prometheus_client import start_http_server, REGISTRY, multiprocess, CollectorRegistry

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for function decorators
F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Any])

# Default configuration
DEFAULT_METRICS_PORT = 9090
DEFAULT_SERVICE_NAME = "forex-trading-service"

# Global registry for metrics
_registry = REGISTRY
_metrics = {}
_metrics_lock = threading.RLock()

def setup_metrics(
    service_name: str = None,
    metrics_port: int = None,
    enable_multiprocess: bool = False
) -> None:
    """
    Set up Prometheus metrics collection.
    
    Args:
        service_name: Name of the service
        metrics_port: Port for the metrics server
        enable_multiprocess: Enable multiprocess mode for metrics
    """
    global _registry
    
    # Get configuration from parameters or environment variables
    service_name = service_name or os.environ.get("SERVICE_NAME", DEFAULT_SERVICE_NAME)
    metrics_port = metrics_port or int(os.environ.get("METRICS_PORT", DEFAULT_METRICS_PORT))
    
    # Set up multiprocess mode if enabled
    if enable_multiprocess:
        multiprocess_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
        if multiprocess_dir:
            logger.info(f"Setting up multiprocess metrics collection in {multiprocess_dir}")
            _registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(_registry)
    
    # Start the HTTP server for metrics
    start_http_server(metrics_port, registry=_registry)
    
    logger.info(f"Prometheus metrics server started on port {metrics_port} for {service_name}")
    
    # Register default metrics
    _register_default_metrics(service_name)

def _register_default_metrics(service_name: str) -> None:
    """
    Register default metrics for the service.
    
    Args:
        service_name: Name of the service
    """
    # Process metrics
    get_gauge(
        name="process_cpu_usage_percent",
        description="CPU usage percentage for the process",
        labels=["service"]
    ).labels(service=service_name).set_function(
        lambda: psutil.Process().cpu_percent(interval=0.1)
    )
    
    get_gauge(
        name="process_memory_usage_bytes",
        description="Memory usage in bytes for the process",
        labels=["service", "type"]
    ).labels(service=service_name, type="rss").set_function(
        lambda: psutil.Process().memory_info().rss
    )
    
    get_gauge(
        name="process_open_files",
        description="Number of open files by the process",
        labels=["service"]
    ).labels(service=service_name).set_function(
        lambda: len(psutil.Process().open_files())
    )
    
    # GC metrics
    get_gauge(
        name="gc_objects",
        description="Number of objects tracked by the garbage collector",
        labels=["service", "generation"]
    ).labels(service=service_name, generation="0").set_function(
        lambda: len(gc.get_objects())
    )

def get_counter(
    name: str,
    description: str,
    labels: List[str] = None
) -> Counter:
    """
    Get or create a Prometheus counter.
    
    Args:
        name: Name of the counter
        description: Description of the counter
        labels: Labels for the counter
        
    Returns:
        Prometheus counter
    """
    return _get_metric(Counter, name, description, labels)

def get_gauge(
    name: str,
    description: str,
    labels: List[str] = None
) -> Gauge:
    """
    Get or create a Prometheus gauge.
    
    Args:
        name: Name of the gauge
        description: Description of the gauge
        labels: Labels for the gauge
        
    Returns:
        Prometheus gauge
    """
    return _get_metric(Gauge, name, description, labels)

def get_histogram(
    name: str,
    description: str,
    labels: List[str] = None,
    buckets: List[float] = None
) -> Histogram:
    """
    Get or create a Prometheus histogram.
    
    Args:
        name: Name of the histogram
        description: Description of the histogram
        labels: Labels for the histogram
        buckets: Buckets for the histogram
        
    Returns:
        Prometheus histogram
    """
    kwargs = {}
    if buckets:
        kwargs["buckets"] = buckets
    return _get_metric(Histogram, name, description, labels, **kwargs)

def get_summary(
    name: str,
    description: str,
    labels: List[str] = None
) -> Summary:
    """
    Get or create a Prometheus summary.
    
    Args:
        name: Name of the summary
        description: Description of the summary
        labels: Labels for the summary
        
    Returns:
        Prometheus summary
    """
    return _get_metric(Summary, name, description, labels)

def _get_metric(
    metric_class,
    name: str,
    description: str,
    labels: List[str] = None,
    **kwargs
):
    """
    Get or create a Prometheus metric.
    
    Args:
        metric_class: Prometheus metric class
        name: Name of the metric
        description: Description of the metric
        labels: Labels for the metric
        **kwargs: Additional arguments for the metric
        
    Returns:
        Prometheus metric
    """
    global _metrics, _metrics_lock
    
    # Normalize labels
    labels = labels or []
    
    # Create a unique key for the metric
    key = f"{metric_class.__name__}:{name}:{','.join(sorted(labels))}"
    
    with _metrics_lock:
        if key not in _metrics:
            _metrics[key] = metric_class(
                name=name,
                documentation=description,
                labelnames=labels,
                registry=_registry,
                **kwargs
            )
        
        return _metrics[key]

def track_execution_time(name: str, labels: Dict[str, str] = None) -> Callable[[F], F]:
    """
    Decorator for tracking execution time of a function.
    
    Args:
        name: Name of the metric
        labels: Labels for the metric
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the histogram
            histogram = get_histogram(
                name=f"{name}_seconds",
                description=f"Execution time of {func.__name__}",
                labels=list(labels.keys()) if labels else []
            )
            
            # Start timing
            start_time = time.time()
            
            try:
                # Call the function
                result = func(*args, **kwargs)
                return result
            finally:
                # Record execution time
                execution_time = time.time() - start_time
                
                # Add labels if provided
                if labels:
                    histogram.labels(**labels).observe(execution_time)
                else:
                    histogram.observe(execution_time)
        
        return cast(F, wrapper)
    
    return decorator

def track_memory_usage(name: str, labels: Dict[str, str] = None) -> Callable[[F], F]:
    """
    Decorator for tracking memory usage of a function.
    
    Args:
        name: Name of the metric
        labels: Labels for the metric
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the gauge
            gauge = get_gauge(
                name=f"{name}_bytes",
                description=f"Memory usage of {func.__name__}",
                labels=list(labels.keys()) if labels else []
            )
            
            # Get initial memory usage
            gc.collect()
            process = psutil.Process()
            start_memory = process.memory_info().rss
            
            try:
                # Call the function
                result = func(*args, **kwargs)
                return result
            finally:
                # Force garbage collection
                gc.collect()
                
                # Get final memory usage
                end_memory = process.memory_info().rss
                memory_used = end_memory - start_memory
                
                # Add labels if provided
                if labels:
                    gauge.labels(**labels).set(memory_used)
                else:
                    gauge.set(memory_used)
        
        return cast(F, wrapper)
    
    return decorator
