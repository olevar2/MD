"""
Standardized Monitoring Module

This module provides standardized monitoring and observability for the service.
It includes health checks, metrics collection, distributed tracing, and logging.
"""

import os
import time
import uuid
import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable, TypeVar, Union

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import generate_latest, REGISTRY

from common_lib.monitoring import (
    MetricsManager,
    MetricType,
    track_time,
    TracingManager,
    trace_function,
    HealthManager,
    HealthCheck,
    HealthStatus,
    configure_logging,
    get_logger,
    log_with_context
)

# Import service-specific settings
from .config.settings import get_service_settings, get_monitoring_settings

# Get service settings
settings = get_service_settings()
monitoring_settings = get_monitoring_settings()

# Configure logging
configure_logging(
    service_name=settings.SERVICE_NAME,
    log_level=settings.LOG_LEVEL,
    json_format=True,
    console_output=True
)

# Create logger
logger = get_logger(settings.SERVICE_NAME)

# Create metrics manager
metrics_manager = MetricsManager(
    service_name=settings.SERVICE_NAME,
    push_gateway_url=monitoring_settings.get("push_gateway_url")
)

# Create tracing manager
tracing_manager = TracingManager(
    service_name=settings.SERVICE_NAME,
    exporter_type=monitoring_settings.get("tracing_exporter_type", "otlp"),
    exporter_endpoint=monitoring_settings.get("tracing_exporter_endpoint", "localhost:4317")
)

# Create health manager
health_manager = HealthManager(
    service_name=settings.SERVICE_NAME,
    version=settings.VERSION
)

# Create standard metrics
http_requests_total = metrics_manager.create_counter(
    name="http_requests_total",
    description="Total number of HTTP requests",
    labels=["method", "endpoint", "status"]
)

http_request_duration_seconds = metrics_manager.create_histogram(
    name="http_request_duration_seconds",
    description="HTTP request duration in seconds",
    labels=["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 10]
)

http_request_size_bytes = metrics_manager.create_histogram(
    name="http_request_size_bytes",
    description="HTTP request size in bytes",
    labels=["method", "endpoint"],
    buckets=[10, 100, 1000, 10000, 100000, 1000000]
)

http_response_size_bytes = metrics_manager.create_histogram(
    name="http_response_size_bytes",
    description="HTTP response size in bytes",
    labels=["method", "endpoint"],
    buckets=[10, 100, 1000, 10000, 100000, 1000000]
)

active_requests = metrics_manager.create_gauge(
    name="active_requests",
    description="Number of active requests",
    labels=["method", "endpoint"]
)

service_calls_total = metrics_manager.create_counter(
    name="service_calls_total",
    description="Total number of service calls",
    labels=["target_service", "operation", "status"]
)

service_call_duration_seconds = metrics_manager.create_histogram(
    name="service_call_duration_seconds",
    description="Service call duration in seconds",
    labels=["target_service", "operation"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 10]
)

database_queries_total = metrics_manager.create_counter(
    name="database_queries_total",
    description="Total number of database queries",
    labels=["operation", "status"]
)

database_query_duration_seconds = metrics_manager.create_histogram(
    name="database_query_duration_seconds",
    description="Database query duration in seconds",
    labels=["operation"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
)

cache_operations_total = metrics_manager.create_counter(
    name="cache_operations_total",
    description="Total number of cache operations",
    labels=["operation", "status"]
)

cache_operation_duration_seconds = metrics_manager.create_histogram(
    name="cache_operation_duration_seconds",
    description="Cache operation duration in seconds",
    labels=["operation"],
    buckets=[0.0001, 0.001, 0.01, 0.1, 1]
)

errors_total = metrics_manager.create_counter(
    name="errors_total",
    description="Total number of errors",
    labels=["error_type", "error_code"]
)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware for monitoring HTTP requests.
    """
    
    async def dispatch(self, request: Request, call_next):
        """
        Process request and collect metrics.
        
        Args:
            request: HTTP request
            call_next: Next middleware or route handler
            
        Returns:
            HTTP response
        """
        # Skip monitoring for monitoring endpoints
        path = request.url.path
        if path in ["/health", "/ready", "/metrics"]:
            return await call_next(request)
        
        # Generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        
        # Add correlation ID to request state
        request.state.correlation_id = correlation_id
        
        # Start timer
        start_time = time.time()
        
        # Extract trace context
        carrier = {}
        for key, value in request.headers.items():
            carrier[key] = value
        
        # Extract context
        context = tracing_manager.extract_context(carrier)
        
        # Start span
        with tracing_manager.start_span(
            name=f"{request.method} {path}",
            context=context,
            attributes={
                "http.method": request.method,
                "http.url": str(request.url),
                "http.host": request.headers.get("host", ""),
                "http.user_agent": request.headers.get("user-agent", ""),
                "http.correlation_id": correlation_id
            }
        ) as span:
            try:
                # Increment active requests
                method = request.method
                active_requests.labels(method=method, endpoint=path).inc()
                
                # Track request size
                content_length = request.headers.get("content-length")
                if content_length:
                    http_request_size_bytes.labels(method=method, endpoint=path).observe(int(content_length))
                
                # Process request
                response = await call_next(request)
                
                # Add correlation ID to response headers
                response.headers["X-Correlation-ID"] = correlation_id
                
                # Record request duration
                duration = time.time() - start_time
                
                # Track response size
                response_length = response.headers.get("content-length")
                if response_length:
                    http_response_size_bytes.labels(method=method, endpoint=path).observe(int(response_length))
                
                # Track request
                status = str(response.status_code)
                http_requests_total.labels(method=method, endpoint=path, status=status).inc()
                http_request_duration_seconds.labels(method=method, endpoint=path).observe(duration)
                
                # Add response attributes to span
                span.set_attribute("http.status_code", response.status_code)
                
                return response
            except Exception as e:
                # Record error
                error_type = e.__class__.__name__
                errors_total.labels(error_type=error_type, error_code="500").inc()
                
                # Add error attributes to span
                span.set_attribute("error", True)
                span.set_attribute("error.type", error_type)
                span.set_attribute("error.message", str(e))
                
                # Log error
                logger.exception(f"Error processing request: {str(e)}", extra={"correlation_id": correlation_id})
                
                # Re-raise exception
                raise
            finally:
                # Decrement active requests
                active_requests.labels(method=method, endpoint=path).dec()


def register_health_check(
    name: str,
    check_func: Callable[[], Union[bool, Dict[str, Any], Awaitable[Union[bool, Dict[str, Any]]]]],
    description: Optional[str] = None,
    timeout: float = 5.0
) -> None:
    """
    Register a health check.
    
    Args:
        name: Name of the health check
        check_func: Function to check health
        description: Description of the health check
        timeout: Timeout for the health check in seconds
    """
    health_manager.register_health_check(
        name=name,
        check_func=check_func,
        description=description,
        timeout=timeout
    )


def setup_monitoring(app: FastAPI) -> None:
    """
    Set up monitoring for a FastAPI application.
    
    Args:
        app: FastAPI application
    """
    # Add monitoring middleware
    app.add_middleware(MonitoringMiddleware)
    
    # Add health check endpoints
    @app.get("/health")
    async def health():
        """
        Health check endpoint.
        
        Returns:
            Health check result
        """
        return await health_manager.check_health()
    
    @app.get("/ready")
    async def ready():
        """
        Readiness check endpoint.
        
        Returns:
            Readiness check result
        """
        return await health_manager.check_health()
    
    @app.get("/metrics")
    async def metrics():
        """
        Metrics endpoint.
        
        Returns:
            Prometheus metrics
        """
        return Response(
            content=generate_latest(REGISTRY),
            media_type="text/plain"
        )
    
    # Register default health check
    register_health_check(
        name="default",
        check_func=lambda: True,
        description="Default health check"
    )
    
    # Log setup
    logger.info(f"Monitoring set up for {settings.SERVICE_NAME}")


def track_service_call(
    target_service: str,
    operation: str,
    status: str,
    duration: float
) -> None:
    """
    Track a service call.
    
    Args:
        target_service: Target service of the call
        operation: Operation of the call
        status: Status of the call
        duration: Duration of the call in seconds
    """
    service_calls_total.labels(
        target_service=target_service,
        operation=operation,
        status=status
    ).inc()
    
    service_call_duration_seconds.labels(
        target_service=target_service,
        operation=operation
    ).observe(duration)


def track_database_query(
    operation: str,
    status: str,
    duration: float
) -> None:
    """
    Track a database query.
    
    Args:
        operation: Operation of the query
        status: Status of the query
        duration: Duration of the query in seconds
    """
    database_queries_total.labels(
        operation=operation,
        status=status
    ).inc()
    
    database_query_duration_seconds.labels(
        operation=operation
    ).observe(duration)


def track_cache_operation(
    operation: str,
    status: str,
    duration: float
) -> None:
    """
    Track a cache operation.
    
    Args:
        operation: Operation of the cache
        status: Status of the cache operation
        duration: Duration of the cache operation in seconds
    """
    cache_operations_total.labels(
        operation=operation,
        status=status
    ).inc()
    
    cache_operation_duration_seconds.labels(
        operation=operation
    ).observe(duration)


def track_error(
    error_type: str,
    error_code: str
) -> None:
    """
    Track an error.
    
    Args:
        error_type: Type of the error
        error_code: Error code
    """
    errors_total.labels(
        error_type=error_type,
        error_code=error_code
    ).inc()
