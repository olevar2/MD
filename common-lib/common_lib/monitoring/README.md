# Monitoring and Observability

This module provides standardized monitoring and observability for the Forex Trading Platform. It includes structured logging, metrics collection, distributed tracing, and health checks.

## Key Components

1. **Structured Logging**: Standardized logging configuration with JSON format, correlation IDs, and integration with distributed tracing
2. **Metrics Collection**: Prometheus metrics for tracking all key operations, performance indicators, and resource usage
3. **Distributed Tracing**: OpenTelemetry tracing for tracking requests across services
4. **Health Checks**: Health checks for services, databases, and other dependencies

## Structured Logging

The structured logging module provides a standardized logging configuration with JSON format, correlation IDs, and integration with distributed tracing.

```python
from common_lib.monitoring import configure_logging, get_logger, log_with_context

# Configure logging
configure_logging(
    service_name="my-service",
    log_level="INFO",
    json_format=True,
    correlation_id="my-correlation-id",
    log_file="/var/log/my-service.log",
    console_output=True
)

# Get logger
logger = get_logger("my-service", correlation_id="my-correlation-id")

# Log message
logger.info("Hello, world!")

# Log message with context
log_with_context(
    logger=logger,
    level="INFO",
    message="Hello, world!",
    context={"key": "value"},
    correlation_id="my-correlation-id"
)
```

## Metrics Collection

The metrics collection module provides Prometheus metrics for tracking all key operations, performance indicators, and resource usage.

```python
from common_lib.monitoring.metrics import MetricsManager, MetricType, track_time

# Create metrics manager
metrics_manager = MetricsManager(service_name="my-service")

# Create counter
counter = metrics_manager.create_counter(
    name="requests_total",
    description="Total number of requests",
    labels=["endpoint", "method", "status"]
)

# Increment counter
counter.labels(endpoint="/api/v1/users", method="GET", status="200").inc()

# Create histogram
histogram = metrics_manager.create_histogram(
    name="request_duration_seconds",
    description="Request duration in seconds",
    labels=["endpoint", "method"],
    buckets=[0.01, 0.1, 1, 10]
)

# Record duration
histogram.labels(endpoint="/api/v1/users", method="GET").observe(0.5)

# Track execution time
@track_time(
    metric_type=MetricType.HISTOGRAM,
    metric_name="function_duration_seconds",
    description="Function execution time in seconds",
    labels={"function": "my_function"}
)
def my_function():
    # Function implementation
    pass
```

## Distributed Tracing

The distributed tracing module provides OpenTelemetry tracing for tracking requests across services.

```python
from common_lib.monitoring.tracing import TracingManager, trace_function

# Initialize tracing
tracing_manager = TracingManager(
    service_name="my-service",
    exporter_type="jaeger",
    exporter_endpoint="localhost:6831"
)

# Trace function
@trace_function(
    name="my_function",
    attributes={"key": "value"}
)
def my_function():
    # Function implementation
    pass

# Start span
with tracing_manager.start_span(
    name="my_span",
    attributes={"key": "value"}
) as span:
    # Span implementation
    span.set_attribute("key", "value")
```

## Health Checks

The health checks module provides health checks for services, databases, and other dependencies.

```python
from common_lib.monitoring.health import HealthManager, HealthStatus

# Create health manager
health_manager = HealthManager(service_name="my-service")

# Register health check
async def check_database():
    # Check database connection
    return HealthStatus.UP

health_manager.register_health_check(
    name="database",
    check_func=check_database,
    description="Check database connection",
    timeout=5.0
)

# Check health
health = await health_manager.check_health()
print(health)

# Check if service is healthy
is_healthy = await health_manager.is_healthy()
print(is_healthy)

# Check if service is ready
is_ready = await health_manager.is_ready()
print(is_ready)
```

## Integration with FastAPI

The monitoring and observability components can be integrated with FastAPI applications.

```python
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from common_lib.monitoring import (
    configure_logging,
    get_logger,
    log_with_context
)
from common_lib.monitoring.metrics import MetricsManager
from common_lib.monitoring.tracing import TracingManager
from common_lib.monitoring.health import HealthManager

# Create FastAPI application
app = FastAPI()

# Configure logging
configure_logging(
    service_name="my-service",
    log_level="INFO",
    json_format=True,
    console_output=True
)

# Create logger
logger = get_logger("my-service")

# Create metrics manager
metrics_manager = MetricsManager(service_name="my-service")

# Create tracing manager
tracing_manager = TracingManager(
    service_name="my-service",
    exporter_type="jaeger",
    exporter_endpoint="localhost:6831"
)

# Create health manager
health_manager = HealthManager(service_name="my-service")

# Add middleware
@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
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
        name=f"{request.method} {request.url.path}",
        context=context,
        kind=trace.SpanKind.SERVER,
        attributes={
            "http.method": request.method,
            "http.url": str(request.url),
            "http.host": request.headers.get("host", ""),
            "http.user_agent": request.headers.get("user-agent", "")
        }
    ) as span:
        try:
            # Process request
            response = await call_next(request)
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            
            # Record request duration
            duration = time.time() - start_time
            
            # Track request
            metrics_manager.track_request(
                endpoint=request.url.path,
                method=request.method,
                status=str(response.status_code),
                duration=duration
            )
            
            # Add response attributes to span
            span.set_attribute("http.status_code", response.status_code)
            
            return response
        except Exception as e:
            # Record error
            metrics_manager.track_error(
                error_type=e.__class__.__name__,
                error_code="500"
            )
            
            # Log error
            log_with_context(
                logger=logger,
                level="ERROR",
                message=f"Error processing request: {str(e)}",
                context={
                    "method": request.method,
                    "url": str(request.url),
                    "error": str(e)
                },
                correlation_id=correlation_id,
                exc_info=e
            )
            
            # Add error to span
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            
            # Return error response
            return JSONResponse(
                status_code=500,
                content={"error": str(e)},
                headers={"X-Correlation-ID": correlation_id}
            )

# Add health check endpoints
@app.get("/health")
async def health():
    return await health_manager.check_health()

@app.get("/health/ready")
async def ready():
    is_ready = await health_manager.is_ready()
    return {"ready": is_ready}

@app.get("/health/live")
async def live():
    return {"alive": True}

# Add metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(
        content=prometheus_client.generate_latest(),
        media_type="text/plain"
    )
```
