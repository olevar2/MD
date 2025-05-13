#!/usr/bin/env python3
"""
Script to implement observability enhancements across services.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

def find_service_directories(root_dir: str) -> List[str]:
    """Find all service directories in the given directory."""
    service_dirs = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path) and (item.endswith('-service') or item.endswith('_service')):
            service_dirs.append(item_path)
    
    return service_dirs

def create_health_check_template() -> str:
    """Create a template for health check endpoints."""
    template = """#!/usr/bin/env python3
\"\"\"
Health check endpoints for the service.
\"\"\"

import logging
import time
import os
import socket
import psutil
import platform
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

# Import database and service clients
# from .database import get_db_connection
# from .service_clients import get_service_client

logger = logging.getLogger(__name__)

# Health check router
health_router = APIRouter(tags=["Health"])

class HealthStatus(BaseModel):
    """
    HealthStatus class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    \"\"\"Health status response model.\"\"\"
    status: str
    version: str
    uptime: float
    timestamp: float
    hostname: str
    details: Dict[str, Any]

class DependencyStatus(BaseModel):
    """
    DependencyStatus class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    \"\"\"Dependency status model.\"\"\"
    name: str
    status: str
    latency: float
    details: Optional[Dict[str, Any]] = None

# Service start time
START_TIME = time.time()

# Service version
VERSION = os.environ.get("SERVICE_VERSION", "0.1.0")

@health_router.get("/health", response_model=HealthStatus)
async def health_check(request: Request) -> HealthStatus:
    """
    Health check.
    
    Args:
        request: Description of request
    
    Returns:
        HealthStatus: Description of return value
    
    """

    \"\"\"
    Basic health check endpoint.
    
    Returns:
        Health status of the service
    \"\"\"
    current_time = time.time()
    uptime = current_time - START_TIME
    
    # Basic system info
    hostname = socket.gethostname()
    
    # Collect health details
    details = {
        "system": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent
        }
    }
    
    return HealthStatus(
        status="healthy",
        version=VERSION,
        uptime=uptime,
        timestamp=current_time,
        hostname=hostname,
        details=details
    )

@health_router.get("/health/liveness")
async def liveness_check() -> Dict[str, str]:
    """
    Liveness check.
    
    Returns:
        Dict[str, str]: Description of return value
    
    """

    \"\"\"
    Liveness probe for Kubernetes.
    
    Returns:
        Simple status response
    \"\"\"
    return {"status": "alive"}

@health_router.get("/health/readiness")
async def readiness_check() -> Dict[str, str]:
    """
    Readiness check.
    
    Returns:
        Dict[str, str]: Description of return value
    
    """

    \"\"\"
    Readiness probe for Kubernetes.
    
    Returns:
        Simple status response
    \"\"\"
    # TODO: Add checks for database and other dependencies
    return {"status": "ready"}

@health_router.get("/health/dependencies")
async def dependency_check() -> Dict[str, List[DependencyStatus]]:
    """
    Dependency check.
    
    Returns:
        Dict[str, List[DependencyStatus]]: Description of return value
    
    """

    \"\"\"
    Check status of all dependencies.
    
    Returns:
        Status of all dependencies
    \"\"\"
    dependencies = []
    
    # Check database
    # try:
    #     db = get_db_connection()
    #     start_time = time.time()
    #     db.execute("SELECT 1")
    #     latency = time.time() - start_time
    #     dependencies.append(
    #         DependencyStatus(
    #             name="database",
    #             status="healthy",
    #             latency=latency
    #         )
    #     )
    # except Exception as e:
    #     logger.error(f"Database health check failed: {str(e)}")
    #     dependencies.append(
    #         DependencyStatus(
    #             name="database",
    #             status="unhealthy",
    #             latency=0.0,
    #             details={"error": str(e)}
    #         )
    #     )
    
    # Check other services
    # try:
    #     client = get_service_client()
    #     start_time = time.time()
    #     response = await client.health_check()
    #     latency = time.time() - start_time
    #     dependencies.append(
    #         DependencyStatus(
    #             name="other-service",
    #             status="healthy" if response.get("status") == "healthy" else "unhealthy",
    #             latency=latency,
    #             details=response
    #         )
    #     )
    # except Exception as e:
    #     logger.error(f"Service health check failed: {str(e)}")
    #     dependencies.append(
    #         DependencyStatus(
    #             name="other-service",
    #             status="unhealthy",
    #             latency=0.0,
    #             details={"error": str(e)}
    #         )
    #     )
    
    return {"dependencies": dependencies}
"""
    
    return template

def create_metrics_template() -> str:
    """Create a template for metrics endpoints."""
    template = """#!/usr/bin/env python3
\"\"\"
Metrics collection and export for the service.
\"\"\"

import logging
import time
import os
from typing import Dict, Any, List, Optional, Callable
from functools import wraps

from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from fastapi import APIRouter, Response, Request

logger = logging.getLogger(__name__)

# Metrics router
metrics_router = APIRouter(tags=["Metrics"])

# Create a registry
registry = CollectorRegistry()

# Define metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"],
    registry=registry
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    registry=registry
)

http_requests_in_progress = Gauge(
    "http_requests_in_progress",
    "Number of HTTP requests in progress",
    ["method", "endpoint"],
    registry=registry
)

# Business metrics
business_operation_total = Counter(
    "business_operation_total",
    "Total number of business operations",
    ["operation", "status"],
    registry=registry
)

business_operation_duration_seconds = Histogram(
    "business_operation_duration_seconds",
    "Business operation duration in seconds",
    ["operation"],
    registry=registry
)

# System metrics
system_memory_usage = Gauge(
    "system_memory_usage",
    "System memory usage in bytes",
    registry=registry
)

system_cpu_usage = Gauge(
    "system_cpu_usage",
    "System CPU usage percentage",
    registry=registry
)

# Middleware for HTTP metrics
async def metrics_middleware(request: Request, call_next):
    """
    Metrics middleware.
    
    Args:
        request: Description of request
        call_next: Description of call_next
    
    """

    \"\"\"
    Middleware to collect HTTP metrics.
    
    Args:
        request: FastAPI request
        call_next: Next middleware or endpoint
        
    Returns:
        Response from next middleware or endpoint
    \"\"\"
    method = request.method
    endpoint = request.url.path
    
    # Exclude metrics endpoint from metrics
    if endpoint == "/metrics":
        return await call_next(request)
    
    # Track in-progress requests
    http_requests_in_progress.labels(method=method, endpoint=endpoint).inc()
    
    # Track request duration
    start_time = time.time()
    
    try:
        response = await call_next(request)
        status = response.status_code
        
        # Record metrics
        http_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
        http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(
            time.time() - start_time
        )
        
        return response
    except Exception as e:
        # Record metrics for exceptions
        http_requests_total.labels(method=method, endpoint=endpoint, status=500).inc()
        http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(
            time.time() - start_time
        )
        
        raise
    finally:
        # Track in-progress requests
        http_requests_in_progress.labels(method=method, endpoint=endpoint).dec()

# Decorator for business metrics
def track_business_operation(operation: str) -> Callable:
    """
    Track business operation.
    
    Args:
        operation: Description of operation
    
    Returns:
        Callable: Description of return value
    
    """

    \"\"\"
    Decorator to track business operations.
    
    Args:
        operation: Name of the operation
        
    Returns:
        Decorated function
    \"\"\"
    def decorator(func: Callable) -> Callable:
    """
    Decorator.
    
    Args:
        func: Description of func
    
    Returns:
        Callable: Description of return value
    
    """

        @wraps(func)
        async def wrapper(*args, **kwargs):
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

            # Track operation duration
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Record metrics
                business_operation_total.labels(operation=operation, status="success").inc()
                business_operation_duration_seconds.labels(operation=operation).observe(
                    time.time() - start_time
                )
                
                return result
            except Exception as e:
                # Record metrics for exceptions
                business_operation_total.labels(operation=operation, status="error").inc()
                business_operation_duration_seconds.labels(operation=operation).observe(
                    time.time() - start_time
                )
                
                raise
        
        return wrapper
    
    return decorator

@metrics_router.get("/metrics")
async def get_metrics() -> Response:
    """
    Get metrics.
    
    Returns:
        Response: Description of return value
    
    """

    \"\"\"
    Endpoint to expose Prometheus metrics.
    
    Returns:
        Prometheus metrics in text format
    \"\"\"
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST
    )
"""
    
    return template

def create_logging_template() -> str:
    """Create a template for structured logging."""
    template = """#!/usr/bin/env python3
\"\"\"
Structured logging setup for the service.
\"\"\"

import logging
import json
import sys
import os
import socket
import traceback
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Union

# Constants
SERVICE_NAME = os.environ.get("SERVICE_NAME", "unknown-service")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

class StructuredLogFormatter(logging.Formatter):
    """
    StructuredLogFormatter class that inherits from logging.Formatter.
    
    Attributes:
        Add attributes here
    """

    \"\"\"
    Formatter for structured JSON logs.
    \"\"\"
    
    def __init__(self, service_name: str, environment: str):
    """
      init  .
    
    Args:
        service_name: Description of service_name
        environment: Description of environment
    
    """

        \"\"\"
        Initialize the formatter.
        
        Args:
            service_name: Name of the service
            environment: Deployment environment
        \"\"\"
        super().__init__()
        self.service_name = service_name
        self.environment = environment
        self.hostname = socket.gethostname()
    
    def format(self, record: logging.LogRecord) -> str:
    """
    Format.
    
    Args:
        record: Description of record
    
    Returns:
        str: Description of return value
    
    """

        \"\"\"
        Format a log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON string
        \"\"\"
        # Basic log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": self.service_name,
            "environment": self.environment,
            "hostname": self.hostname,
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "path": record.pathname,
            "line": record.lineno,
            "function": record.funcName
        }
        
        # Add correlation ID if available
        if hasattr(record, "correlation_id"):
            log_data["correlation_id"] = record.correlation_id
        
        # Add request ID if available
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        
        # Add user ID if available
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        
        # Add extra fields
        if hasattr(record, "data") and record.data:
            log_data["data"] = record.data
        
        # Add exception info if available
        if record.exc_info:
            exception_type, exception_value, exception_traceback = record.exc_info
            log_data["exception"] = {
                "type": exception_type.__name__,
                "message": str(exception_value),
                "traceback": traceback.format_exception(
                    exception_type, exception_value, exception_traceback
                )
            }
        
        return json.dumps(log_data)

def setup_logging(
    service_name: str = SERVICE_NAME,
    environment: str = ENVIRONMENT,
    log_level: str = LOG_LEVEL
) -> None:
    """
    Setup logging.
    
    Args:
        service_name: Description of service_name
        environment: Description of environment
        log_level: Description of log_level
    
    """

    \"\"\"
    Set up structured logging.
    
    Args:
        service_name: Name of the service
        environment: Deployment environment
        log_level: Log level
    \"\"\"
    # Create formatter
    formatter = StructuredLogFormatter(service_name, environment)
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Disable propagation for some loggers
    for logger_name in ["uvicorn", "uvicorn.access"]:
        logger = logging.getLogger(logger_name)
        logger.propagate = False
        logger.handlers = [handler]
        logger.setLevel(getattr(logging, log_level.upper()))

class StructuredLogger:
    """
    StructuredLogger class.
    
    Attributes:
        Add attributes here
    """

    \"\"\"
    Logger that adds structured context to log messages.
    \"\"\"
    
    def __init__(self, name: str):
    """
      init  .
    
    Args:
        name: Description of name
    
    """

        \"\"\"
        Initialize the logger.
        
        Args:
            name: Logger name
        \"\"\"
        self.logger = logging.getLogger(name)
        self.context = {}
    
    def with_context(self, **kwargs) -> "StructuredLogger":
    """
    With context.
    
    Args:
        kwargs: Description of kwargs
    
    Returns:
        "StructuredLogger": Description of return value
    
    """

        \"\"\"
        Add context to the logger.
        
        Args:
            **kwargs: Context key-value pairs
            
        Returns:
            Logger with context
        \"\"\"
        logger = StructuredLogger(self.logger.name)
        logger.context = {**self.context, **kwargs}
        return logger
    
    def _log(self, level: int, msg: str, *args, **kwargs) -> None:
    """
     log.
    
    Args:
        level: Description of level
        msg: Description of msg
        args: Description of args
        kwargs: Description of kwargs
    
    """

        \"\"\"
        Log a message with context.
        
        Args:
            level: Log level
            msg: Log message
            *args: Message format args
            **kwargs: Additional log data
        \"\"\"
        # Extract data from kwargs
        data = kwargs.pop("data", {})
        
        # Add context to data
        if self.context:
            data = {**self.context, **data}
        
        # Create extra dict
        extra = kwargs.pop("extra", {})
        extra["data"] = data
        
        # Log message
        self.logger.log(level, msg, *args, **{**kwargs, "extra": extra})
    
    def debug(self, msg: str, *args, **kwargs) -> None:
    """
    Debug.
    
    Args:
        msg: Description of msg
        args: Description of args
        kwargs: Description of kwargs
    
    """

        \"\"\"Log a debug message.\"\"\"
        self._log(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs) -> None:
    """
    Info.
    
    Args:
        msg: Description of msg
        args: Description of args
        kwargs: Description of kwargs
    
    """

        \"\"\"Log an info message.\"\"\"
        self._log(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs) -> None:
    """
    Warning.
    
    Args:
        msg: Description of msg
        args: Description of args
        kwargs: Description of kwargs
    
    """

        \"\"\"Log a warning message.\"\"\"
        self._log(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs) -> None:
    """
    Error.
    
    Args:
        msg: Description of msg
        args: Description of args
        kwargs: Description of kwargs
    
    """

        \"\"\"Log an error message.\"\"\"
        self._log(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs) -> None:
    """
    Critical.
    
    Args:
        msg: Description of msg
        args: Description of args
        kwargs: Description of kwargs
    
    """

        \"\"\"Log a critical message.\"\"\"
        self._log(logging.CRITICAL, msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs) -> None:
    """
    Exception.
    
    Args:
        msg: Description of msg
        args: Description of args
        kwargs: Description of kwargs
    
    """

        \"\"\"Log an exception message.\"\"\"
        kwargs["exc_info"] = kwargs.get("exc_info", True)
        self._log(logging.ERROR, msg, *args, **kwargs)

def get_logger(name: str) -> StructuredLogger:
    \"\"\"
    Get a structured logger.
    
    Args:
        name: Logger name
        
    Returns:
        Structured logger
    \"\"\"
    return StructuredLogger(name)
"""
    
    return template

def create_tracing_template() -> str:
    """Create a template for distributed tracing."""
    template = """#!/usr/bin/env python3
\"\"\"
Distributed tracing setup for the service.
\"\"\"

import logging
import os
import time
from typing import Dict, Any, Optional, Callable, List
from functools import wraps

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
from fastapi import Request, Response

logger = logging.getLogger(__name__)

# Constants
SERVICE_NAME = os.environ.get("SERVICE_NAME", "unknown-service")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")
OTLP_ENDPOINT = os.environ.get("OTLP_ENDPOINT", "localhost:4317")

def setup_tracing(
    service_name: str = SERVICE_NAME,
    environment: str = ENVIRONMENT,
    otlp_endpoint: str = OTLP_ENDPOINT
) -> None:
    """
    Setup tracing.
    
    Args:
        service_name: Description of service_name
        environment: Description of environment
        otlp_endpoint: Description of otlp_endpoint
    
    """

    \"\"\"
    Set up distributed tracing.
    
    Args:
        service_name: Name of the service
        environment: Deployment environment
        otlp_endpoint: OpenTelemetry collector endpoint
    \"\"\"
    # Create resource
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: service_name,
        ResourceAttributes.DEPLOYMENT_ENVIRONMENT: environment
    })
    
    # Create tracer provider
    tracer_provider = TracerProvider(resource=resource)
    
    # Create exporter
    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
    
    # Create processor
    span_processor = BatchSpanProcessor(otlp_exporter)
    
    # Add processor to provider
    tracer_provider.add_span_processor(span_processor)
    
    # Set global tracer provider
    trace.set_tracer_provider(tracer_provider)

def get_tracer(name: str) -> trace.Tracer:
    """
    Get tracer.
    
    Args:
        name: Description of name
    
    Returns:
        trace.Tracer: Description of return value
    
    """

    \"\"\"
    Get a tracer.
    
    Args:
        name: Tracer name
        
    Returns:
        Tracer
    \"\"\"
    return trace.get_tracer(name)

# Middleware for tracing
async def tracing_middleware(request: Request, call_next):
    """
    Tracing middleware.
    
    Args:
        request: Description of request
        call_next: Description of call_next
    
    """

    \"\"\"
    Middleware to add tracing to requests.
    
    Args:
        request: FastAPI request
        call_next: Next middleware or endpoint
        
    Returns:
        Response from next middleware or endpoint
    \"\"\"
    tracer = get_tracer("fastapi")
    
    # Extract trace context from headers
    # TODO: Implement trace context extraction
    
    # Create span
    with tracer.start_as_current_span(
        f"{request.method} {request.url.path}",
        kind=trace.SpanKind.SERVER
    ) as span:
        # Add request attributes
        span.set_attribute("http.method", request.method)
        span.set_attribute("http.url", str(request.url))
        span.set_attribute("http.host", request.headers.get("host", ""))
        span.set_attribute("http.user_agent", request.headers.get("user-agent", ""))
        
        # Call next middleware or endpoint
        try:
            response = await call_next(request)
            
            # Add response attributes
            span.set_attribute("http.status_code", response.status_code)
            
            return response
        except Exception as e:
            # Record exception
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            
            raise

# Decorator for tracing functions
def trace_function(name: Optional[str] = None) -> Callable:
    """
    Trace function.
    
    Args:
        name: Description of name
    
    Returns:
        Callable: Description of return value
    
    """

    \"\"\"
    Decorator to trace a function.
    
    Args:
        name: Name of the span
        
    Returns:
        Decorated function
    \"\"\"
    def decorator(func: Callable) -> Callable:
    """
    Decorator.
    
    Args:
        func: Description of func
    
    Returns:
        Callable: Description of return value
    
    """

        @wraps(func)
        async def wrapper(*args, **kwargs):
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

            tracer = get_tracer(func.__module__)
            
            # Create span
            with tracer.start_as_current_span(
                name or func.__name__,
                kind=trace.SpanKind.INTERNAL
            ) as span:
                # Add function attributes
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                
                # Call function
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    # Record exception
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    
                    raise
        
        return wrapper
    
    return decorator
"""
    
    return template

def create_observability_files(service_dir: str) -> None:
    """Create observability files for a service."""
    # Create health_check.py
    health_check_file = os.path.join(service_dir, 'health_check.py')
    if not os.path.exists(health_check_file):
        with open(health_check_file, 'w') as f:
            f.write(create_health_check_template())
        print(f"Created health check file: {health_check_file}")
    
    # Create metrics.py
    metrics_file = os.path.join(service_dir, 'metrics.py')
    if not os.path.exists(metrics_file):
        with open(metrics_file, 'w') as f:
            f.write(create_metrics_template())
        print(f"Created metrics file: {metrics_file}")
    
    # Create logging_setup.py
    logging_file = os.path.join(service_dir, 'logging_setup.py')
    if not os.path.exists(logging_file):
        with open(logging_file, 'w') as f:
            f.write(create_logging_template())
        print(f"Created logging file: {logging_file}")
    
    # Create tracing.py
    tracing_file = os.path.join(service_dir, 'tracing.py')
    if not os.path.exists(tracing_file):
        with open(tracing_file, 'w') as f:
            f.write(create_tracing_template())
        print(f"Created tracing file: {tracing_file}")

def main():
    """Main function to implement observability enhancements."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Find service directories
    service_dirs = find_service_directories(root_dir)
    
    # Create observability files for each service
    for service_dir in service_dirs:
        create_observability_files(service_dir)
    
    print("Observability enhancements completed.")

if __name__ == '__main__':
    main()
