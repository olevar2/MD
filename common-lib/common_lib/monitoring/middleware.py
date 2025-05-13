"""
Standardized Metrics Middleware for FastAPI Applications.

This module provides a standardized middleware for collecting metrics in FastAPI applications
using the metrics_standards module.
"""

import time
import logging
from typing import Callable, Dict, Any, Optional, List
from contextlib import contextmanager
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from common_lib.monitoring.metrics_standards import StandardMetrics

# Configure logging
logger = logging.getLogger(__name__)

class StandardMetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting standardized metrics in FastAPI applications.

    This middleware:
    1. Measures request duration
    2. Records request count
    3. Records request size
    4. Records response size
    5. Records error count

    All metrics follow the standardized naming and labeling conventions defined in
    the metrics_standards module.
    """

    def __init__(
        self,
        app: ASGIApp,
        service_name: str,
        exclude_paths: List[str] = None,
        metrics_instance: Optional[StandardMetrics] = None
    ):
        """
        Initialize the middleware.

        Args:
            app: The ASGI application
            service_name: Name of the service
            exclude_paths: List of paths to exclude from metrics collection
            metrics_instance: Instance of StandardMetrics to use (if None, a new instance will be created)
        """
        super().__init__(app)
        self.service_name = service_name
        self.exclude_paths = exclude_paths or ["/metrics", "/health", "/docs", "/redoc", "/openapi.json"]
        self.metrics = metrics_instance or StandardMetrics(service_name)

        logger.info(f"Initialized StandardMetricsMiddleware for {service_name}")

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Process the request and collect metrics.

        Args:
            request: FastAPI request
            call_next: Function to call the next middleware

        Returns:
            FastAPI response
        """
        # Skip metrics collection for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        # Get request path and method
        path = request.url.path
        method = request.method

        # Start timer
        start_time = time.time()

        # Get request size
        request_size = int(request.headers.get("content-length", 0))

        # Record request size
        self.metrics.api_request_size_bytes.labels(
            service=self.service_name,
            endpoint=path,
            method=method
        ).observe(request_size)

        # Process the request
        try:
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Get response size
            response_size = int(response.headers.get("content-length", 0))

            # Record metrics
            self.metrics.api_requests_total.labels(
                service=self.service_name,
                endpoint=path,
                method=method,
                status_code=str(response.status_code)
            ).inc()

            # Record duration
            self.metrics.api_request_duration_seconds.labels(
                service=self.service_name,
                endpoint=path,
                method=method
            ).observe(duration)

            # Record response size
            self.metrics.api_response_size_bytes.labels(
                service=self.service_name,
                endpoint=path,
                method=method
            ).observe(response_size)

            # Record error if status code is 4xx or 5xx
            if response.status_code >= 400:
                error_type = "client_error" if response.status_code < 500 else "server_error"
                self.metrics.api_errors_total.labels(
                    service=self.service_name,
                    endpoint=path,
                    method=method,
                    error_type=error_type
                ).inc()

            return response
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time

            # Record metrics for exception
            self.metrics.api_requests_total.labels(
                service=self.service_name,
                endpoint=path,
                method=method,
                status_code="500"
            ).inc()

            # Record duration
            self.metrics.api_request_duration_seconds.labels(
                service=self.service_name,
                endpoint=path,
                method=method
            ).observe(duration)

            # Record error
            self.metrics.api_errors_total.labels(
                service=self.service_name,
                endpoint=path,
                method=method,
                error_type="exception"
            ).inc()

            # Re-raise the exception
            raise

class DatabaseMetricsMiddleware:
    """
    Middleware for collecting database metrics.

    This class is not a middleware in the traditional sense, but rather a utility
    for collecting database metrics. It can be used to wrap database operations
    to collect metrics.
    """

    def __init__(
        self,
        service_name: str,
        database_name: str,
        metrics_instance: Optional[StandardMetrics] = None
    ):
        """
        Initialize the middleware.

        Args:
            service_name: Name of the service
            database_name: Name of the database
            metrics_instance: Instance of StandardMetrics to use (if None, a new instance will be created)
        """
        self.service_name = service_name
        self.database_name = database_name
        self.metrics = metrics_instance or StandardMetrics(service_name)

        logger.info(f"Initialized DatabaseMetricsMiddleware for {service_name} - {database_name}")

    @contextmanager
    def track_operation(
        self,
        operation: str,
        table: str
    ):
        """
        Track a database operation.

        Args:
            operation: Name of the operation (e.g., "select", "insert", "update", "delete")
            table: Name of the table

        Yields:
            None
        """
        # Start timer
        start_time = time.time()

        # Increment operation counter
        self.metrics.db_operations_total.labels(
            service=self.service_name,
            database=self.database_name,
            operation=operation,
            table=table
        ).inc()

        try:
            # Yield control back to the caller
            yield

            # Calculate duration
            duration = time.time() - start_time

            # Record duration
            self.metrics.db_operation_duration_seconds.labels(
                service=self.service_name,
                database=self.database_name,
                operation=operation,
                table=table
            ).observe(duration)
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time

            # Record duration
            self.metrics.db_operation_duration_seconds.labels(
                service=self.service_name,
                database=self.database_name,
                operation=operation,
                table=table
            ).observe(duration)

            # Record error
            self.metrics.db_errors_total.labels(
                service=self.service_name,
                database=self.database_name,
                operation=operation,
                error_type=type(e).__name__
            ).inc()

            # Re-raise the exception
            raise

class CacheMetricsMiddleware:
    """
    Middleware for collecting cache metrics.

    This class is not a middleware in the traditional sense, but rather a utility
    for collecting cache metrics. It can be used to wrap cache operations
    to collect metrics.
    """

    def __init__(
        self,
        service_name: str,
        cache_type: str,
        metrics_instance: Optional[StandardMetrics] = None
    ):
        """
        Initialize the middleware.

        Args:
            service_name: Name of the service
            cache_type: Type of cache (e.g., "redis", "local", "distributed")
            metrics_instance: Instance of StandardMetrics to use (if None, a new instance will be created)
        """
        self.service_name = service_name
        self.cache_type = cache_type
        self.metrics = metrics_instance or StandardMetrics(service_name)

        logger.info(f"Initialized CacheMetricsMiddleware for {service_name} - {cache_type}")

    @contextmanager
    def track_operation(
        self,
        operation: str
    ):
        """
        Track a cache operation.

        Args:
            operation: Name of the operation (e.g., "get", "set", "delete")

        Yields:
            None
        """
        # Start timer
        start_time = time.time()

        # Increment operation counter
        self.metrics.cache_operations_total.labels(
            service=self.service_name,
            cache_type=self.cache_type,
            operation=operation
        ).inc()

        try:
            # Yield control back to the caller
            yield

            # Calculate duration
            duration = time.time() - start_time

            # Record duration
            self.metrics.cache_operation_duration_seconds.labels(
                service=self.service_name,
                cache_type=self.cache_type,
                operation=operation
            ).observe(duration)
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time

            # Record duration
            self.metrics.cache_operation_duration_seconds.labels(
                service=self.service_name,
                cache_type=self.cache_type,
                operation=operation
            ).observe(duration)

            # Re-raise the exception
            raise

    def record_hit(self):
        """Record a cache hit."""
        self.metrics.cache_hits_total.labels(
            service=self.service_name,
            cache_type=self.cache_type
        ).inc()

    def record_miss(self):
        """Record a cache miss."""
        self.metrics.cache_misses_total.labels(
            service=self.service_name,
            cache_type=self.cache_type
        ).inc()

    def update_hit_ratio(self, ratio: float):
        """
        Update the cache hit ratio.

        Args:
            ratio: Hit ratio (0-1)
        """
        self.metrics.cache_hit_ratio.labels(
            service=self.service_name,
            cache_type=self.cache_type
        ).set(ratio)

    def update_size(self, size: int):
        """
        Update the cache size.

        Args:
            size: Number of items in the cache
        """
        self.metrics.cache_size.labels(
            service=self.service_name,
            cache_type=self.cache_type
        ).set(size)

    def update_memory_usage(self, memory_usage: int):
        """
        Update the cache memory usage.

        Args:
            memory_usage: Memory usage in bytes
        """
        self.metrics.cache_memory_usage_bytes.labels(
            service=self.service_name,
            cache_type=self.cache_type
        ).set(memory_usage)
