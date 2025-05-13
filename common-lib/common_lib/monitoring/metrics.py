"""
Metrics Module

This module provides metrics collection functionality for the platform.
"""

import time
import logging
from typing import Dict, Any, Optional, List, Callable, ClassVar
from enum import Enum
from functools import wraps

from prometheus_client import Counter, Gauge, Histogram, Summary, REGISTRY, CollectorRegistry, push_to_gateway


class MetricType(Enum):
    """
    Metric types.
    """
    
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class MetricsManager:
    """
    Metrics manager for the platform.
    
    This class provides a singleton manager for metrics.
    """
    
    _instance: ClassVar[Optional["MetricsManager"]] = None
    
    def __new__(cls, *args, **kwargs):
        """
        Create a new instance of the metrics manager.
        
        Returns:
            Singleton instance of the metrics manager
        """
        if cls._instance is None:
            cls._instance = super(MetricsManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        service_name: str,
        push_gateway_url: Optional[str] = None,
        push_interval: float = 15.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the metrics manager.
        
        Args:
            service_name: Name of the service
            push_gateway_url: URL of the Prometheus push gateway
            push_interval: Interval in seconds for pushing metrics to the gateway
            logger: Logger to use (if None, creates a new logger)
        """
        # Skip initialization if already initialized
        if getattr(self, "_initialized", False):
            return
        
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.service_name = service_name
        self.push_gateway_url = push_gateway_url
        self.push_interval = push_interval
        
        # Create registry
        self.registry = CollectorRegistry()
        
        # Create metrics
        self.metrics = {}
        
        # Create default metrics
        self._create_default_metrics()
        
        self._initialized = True
    
    def _create_default_metrics(self):
        """
        Create default metrics.
        """
        # Create request metrics
        self.create_counter(
            name="requests_total",
            description="Total number of requests",
            labels=["service", "endpoint", "method", "status"]
        )
        
        self.create_histogram(
            name="request_duration_seconds",
            description="Request duration in seconds",
            labels=["service", "endpoint", "method"],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
        )
        
        # Create database metrics
        self.create_counter(
            name="database_queries_total",
            description="Total number of database queries",
            labels=["service", "operation", "status"]
        )
        
        self.create_histogram(
            name="database_query_duration_seconds",
            description="Database query duration in seconds",
            labels=["service", "operation"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5]
        )
        
        # Create service metrics
        self.create_counter(
            name="service_calls_total",
            description="Total number of service calls",
            labels=["service", "target_service", "operation", "status"]
        )
        
        self.create_histogram(
            name="service_call_duration_seconds",
            description="Service call duration in seconds",
            labels=["service", "target_service", "operation"],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
        )
        
        # Create error metrics
        self.create_counter(
            name="errors_total",
            description="Total number of errors",
            labels=["service", "error_type", "error_code"]
        )
        
        # Create resource metrics
        self.create_gauge(
            name="memory_usage_bytes",
            description="Memory usage in bytes",
            labels=["service"]
        )
        
        self.create_gauge(
            name="cpu_usage_percent",
            description="CPU usage in percent",
            labels=["service"]
        )
    
    def create_counter(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        namespace: str = "forex"
    ) -> Counter:
        """
        Create a counter metric.
        
        Args:
            name: Name of the metric
            description: Description of the metric
            labels: Labels for the metric
            namespace: Namespace for the metric
            
        Returns:
            Counter metric
        """
        labels = labels or []
        metric_name = f"{namespace}_{name}"
        
        if metric_name in self.metrics:
            return self.metrics[metric_name]
        
        counter = Counter(
            name,
            description,
            labels,
            registry=self.registry,
            namespace=namespace
        )
        
        self.metrics[metric_name] = counter
        
        return counter
    
    def create_gauge(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        namespace: str = "forex"
    ) -> Gauge:
        """
        Create a gauge metric.
        
        Args:
            name: Name of the metric
            description: Description of the metric
            labels: Labels for the metric
            namespace: Namespace for the metric
            
        Returns:
            Gauge metric
        """
        labels = labels or []
        metric_name = f"{namespace}_{name}"
        
        if metric_name in self.metrics:
            return self.metrics[metric_name]
        
        gauge = Gauge(
            name,
            description,
            labels,
            registry=self.registry,
            namespace=namespace
        )
        
        self.metrics[metric_name] = gauge
        
        return gauge
    
    def create_histogram(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None,
        namespace: str = "forex"
    ) -> Histogram:
        """
        Create a histogram metric.
        
        Args:
            name: Name of the metric
            description: Description of the metric
            labels: Labels for the metric
            buckets: Buckets for the histogram
            namespace: Namespace for the metric
            
        Returns:
            Histogram metric
        """
        labels = labels or []
        buckets = buckets or [0.01, 0.1, 1, 10, 100]
        metric_name = f"{namespace}_{name}"
        
        if metric_name in self.metrics:
            return self.metrics[metric_name]
        
        histogram = Histogram(
            name,
            description,
            labels,
            buckets=buckets,
            registry=self.registry,
            namespace=namespace
        )
        
        self.metrics[metric_name] = histogram
        
        return histogram
    
    def create_summary(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        namespace: str = "forex"
    ) -> Summary:
        """
        Create a summary metric.
        
        Args:
            name: Name of the metric
            description: Description of the metric
            labels: Labels for the metric
            namespace: Namespace for the metric
            
        Returns:
            Summary metric
        """
        labels = labels or []
        metric_name = f"{namespace}_{name}"
        
        if metric_name in self.metrics:
            return self.metrics[metric_name]
        
        summary = Summary(
            name,
            description,
            labels,
            registry=self.registry,
            namespace=namespace
        )
        
        self.metrics[metric_name] = summary
        
        return summary
    
    def get_metric(self, name: str, namespace: str = "forex") -> Any:
        """
        Get a metric by name.
        
        Args:
            name: Name of the metric
            namespace: Namespace for the metric
            
        Returns:
            Metric
            
        Raises:
            KeyError: If the metric is not found
        """
        metric_name = f"{namespace}_{name}"
        
        if metric_name not in self.metrics:
            raise KeyError(f"Metric not found: {metric_name}")
        
        return self.metrics[metric_name]
    
    def push_metrics(self):
        """
        Push metrics to the Prometheus push gateway.
        """
        if not self.push_gateway_url:
            return
        
        try:
            push_to_gateway(
                self.push_gateway_url,
                job=self.service_name,
                registry=self.registry
            )
        except Exception as e:
            self.logger.error(f"Error pushing metrics to gateway: {str(e)}")
    
    def track_request(
        self,
        endpoint: str,
        method: str,
        status: str,
        duration: float
    ):
        """
        Track a request.
        
        Args:
            endpoint: Endpoint of the request
            method: HTTP method of the request
            status: Status of the request
            duration: Duration of the request in seconds
        """
        # Increment request counter
        self.get_metric("requests_total").labels(
            service=self.service_name,
            endpoint=endpoint,
            method=method,
            status=status
        ).inc()
        
        # Record request duration
        self.get_metric("request_duration_seconds").labels(
            service=self.service_name,
            endpoint=endpoint,
            method=method
        ).observe(duration)
    
    def track_database_query(
        self,
        operation: str,
        status: str,
        duration: float
    ):
        """
        Track a database query.
        
        Args:
            operation: Operation of the query
            status: Status of the query
            duration: Duration of the query in seconds
        """
        # Increment query counter
        self.get_metric("database_queries_total").labels(
            service=self.service_name,
            operation=operation,
            status=status
        ).inc()
        
        # Record query duration
        self.get_metric("database_query_duration_seconds").labels(
            service=self.service_name,
            operation=operation
        ).observe(duration)
    
    def track_service_call(
        self,
        target_service: str,
        operation: str,
        status: str,
        duration: float
    ):
        """
        Track a service call.
        
        Args:
            target_service: Target service of the call
            operation: Operation of the call
            status: Status of the call
            duration: Duration of the call in seconds
        """
        # Increment call counter
        self.get_metric("service_calls_total").labels(
            service=self.service_name,
            target_service=target_service,
            operation=operation,
            status=status
        ).inc()
        
        # Record call duration
        self.get_metric("service_call_duration_seconds").labels(
            service=self.service_name,
            target_service=target_service,
            operation=operation
        ).observe(duration)
    
    def track_error(
        self,
        error_type: str,
        error_code: str
    ):
        """
        Track an error.
        
        Args:
            error_type: Type of the error
            error_code: Code of the error
        """
        # Increment error counter
        self.get_metric("errors_total").labels(
            service=self.service_name,
            error_type=error_type,
            error_code=error_code
        ).inc()
    
    def set_memory_usage(self, memory_usage: float):
        """
        Set memory usage.
        
        Args:
            memory_usage: Memory usage in bytes
        """
        # Set memory usage gauge
        self.get_metric("memory_usage_bytes").labels(
            service=self.service_name
        ).set(memory_usage)
    
    def set_cpu_usage(self, cpu_usage: float):
        """
        Set CPU usage.
        
        Args:
            cpu_usage: CPU usage in percent
        """
        # Set CPU usage gauge
        self.get_metric("cpu_usage_percent").labels(
            service=self.service_name
        ).set(cpu_usage)


def track_time(
    metric_type: MetricType,
    metric_name: str,
    description: str,
    labels: Optional[Dict[str, str]] = None,
    buckets: Optional[List[float]] = None
):
    """
    Decorator for tracking execution time of a function.
    
    Args:
        metric_type: Type of the metric
        metric_name: Name of the metric
        description: Description of the metric
        labels: Labels for the metric
        buckets: Buckets for the histogram
        
    Returns:
        Decorated function
    """
    def decorator(func):
    """
    Decorator.
    
    Args:
        func: Description of func
    
    """

        @wraps(func)
        def wrapper(*args, **kwargs):
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

            # Get metrics manager
            metrics_manager = MetricsManager()
            
            # Create metric if it doesn't exist
            if metric_type == MetricType.COUNTER:
                metric = metrics_manager.create_counter(
                    metric_name,
                    description,
                    list(labels.keys()) if labels else None
                )
            elif metric_type == MetricType.GAUGE:
                metric = metrics_manager.create_gauge(
                    metric_name,
                    description,
                    list(labels.keys()) if labels else None
                )
            elif metric_type == MetricType.HISTOGRAM:
                metric = metrics_manager.create_histogram(
                    metric_name,
                    description,
                    list(labels.keys()) if labels else None,
                    buckets
                )
            elif metric_type == MetricType.SUMMARY:
                metric = metrics_manager.create_summary(
                    metric_name,
                    description,
                    list(labels.keys()) if labels else None
                )
            else:
                raise ValueError(f"Invalid metric type: {metric_type}")
            
            # Start timer
            start_time = time.time()
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Record execution time
                if metric_type == MetricType.COUNTER:
                    metric.labels(**labels).inc()
                elif metric_type == MetricType.GAUGE:
                    metric.labels(**labels).set(time.time() - start_time)
                elif metric_type == MetricType.HISTOGRAM:
                    metric.labels(**labels).observe(time.time() - start_time)
                elif metric_type == MetricType.SUMMARY:
                    metric.labels(**labels).observe(time.time() - start_time)
                
                return result
            except Exception as e:
                # Record execution time
                if metric_type == MetricType.COUNTER:
                    metric.labels(**labels).inc()
                elif metric_type == MetricType.GAUGE:
                    metric.labels(**labels).set(time.time() - start_time)
                elif metric_type == MetricType.HISTOGRAM:
                    metric.labels(**labels).observe(time.time() - start_time)
                elif metric_type == MetricType.SUMMARY:
                    metric.labels(**labels).observe(time.time() - start_time)
                
                # Re-raise exception
                raise
        
        return wrapper
    
    return decorator