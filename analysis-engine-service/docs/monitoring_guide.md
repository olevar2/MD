# Analysis Engine Service Monitoring Guide

## Overview

This guide describes the monitoring capabilities of the Analysis Engine Service, including metrics collection, structured logging, health checks, and dashboards.

## Metrics Collection

The Analysis Engine Service collects comprehensive metrics for all key operations using Prometheus. These metrics are exposed at the `/metrics` endpoint and can be scraped by Prometheus.

### Key Metrics

#### Analysis Operation Metrics

- `analysis_engine_requests_total`: Total number of analysis requests
  - Labels: `operation`, `symbol`, `timeframe`
- `analysis_engine_errors_total`: Total number of analysis errors
  - Labels: `operation`, `symbol`, `timeframe`, `error_type`
- `analysis_engine_duration_seconds`: Duration of analysis operations in seconds
  - Labels: `operation`, `symbol`, `timeframe`

#### Resource Usage Metrics

- `analysis_engine_resource_usage`: Resource usage percentage
  - Labels: `resource_type` (cpu, memory, disk)

#### Cache Metrics

- `analysis_engine_cache_operations_total`: Total number of cache operations
  - Labels: `operation`, `cache_type`
- `analysis_engine_cache_hits_total`: Total number of cache hits
  - Labels: `cache_type`
- `analysis_engine_cache_misses_total`: Total number of cache misses
  - Labels: `cache_type`

#### Dependency Health Metrics

- `analysis_engine_dependency_health`: Health status of dependencies (1=healthy, 0=unhealthy)
  - Labels: `dependency_name`
- `analysis_engine_dependency_latency_seconds`: Latency of dependency operations in seconds
  - Labels: `dependency_name`, `operation`

#### API Metrics

- `analysis_engine_api_duration_seconds`: Duration of API requests in seconds
  - Labels: `endpoint`, `method`, `status_code`
- `analysis_engine_api_requests_total`: Total number of API requests
  - Labels: `endpoint`, `method`, `status_code`

### Using Metrics in Code

The `MetricsRecorder` class provides utility methods for recording metrics:

```python
from analysis_engine.monitoring.metrics import MetricsRecorder

# Record an analysis request
MetricsRecorder.record_analysis_request(
    operation="analyze_market",
    symbol="EURUSD",
    timeframe="1h"
)

# Time an analysis operation
with MetricsRecorder.time_analysis_operation(
    operation="analyze_market",
    symbol="EURUSD",
    timeframe="1h"
):
    # Perform analysis
    result = analyze_market(symbol="EURUSD", timeframe="1h")

# Record resource usage
MetricsRecorder.record_resource_usage(
    resource_type="cpu",
    usage=cpu_percent
)
```

## Structured Logging

The Analysis Engine Service uses structured logging with correlation IDs and context data. This makes it easier to trace requests across services and debug issues.

### Key Features

- **Correlation IDs**: Each request has a unique correlation ID that is propagated across services
- **Request IDs**: Each request has a unique request ID
- **Context Data**: Logs include context data like function name, line number, and thread ID
- **JSON Format**: Logs are formatted as JSON for easy parsing

### Using Structured Logging in Code

```python
from analysis_engine.monitoring.structured_logging import (
    get_structured_logger,
    set_correlation_id,
    get_correlation_id,
    log_execution_time
)

# Get a structured logger
logger = get_structured_logger(__name__)

# Set a correlation ID
correlation_id = set_correlation_id()

# Log with context data
logger.info(
    "Processing analysis request",
    {
        "symbol": "EURUSD",
        "timeframe": "1h",
        "correlation_id": correlation_id
    }
)

# Log execution time
@log_execution_time(logger)
def analyze_market(symbol, timeframe):
    # Function implementation
    pass
```

## Health Checks

The Analysis Engine Service provides comprehensive health checks for the service and its dependencies.

### Health Check Endpoints

- `/api/health/live`: Liveness probe (simple check that service is running)
- `/api/health/ready`: Readiness probe (check if service is ready to receive traffic)
- `/api/health`: Detailed health check with component and dependency status

### Health Check Response

The detailed health check endpoint returns a JSON response with the following structure:

```json
{
  "status": "healthy",
  "components": [
    {
      "name": "market_regime_analyzer",
      "status": "healthy",
      "message": "Market regime analyzer is healthy",
      "details": {
        "latency_ms": 5.2
      }
    }
  ],
  "dependencies": [
    {
      "name": "feature-store-service",
      "status": "healthy",
      "latency_ms": 12.5,
      "message": "Feature store service connection is healthy",
      "details": {
        "latency_ms": 12.5,
        "status_code": 200
      }
    }
  ],
  "uptime_seconds": 3600,
  "version": "1.0.0",
  "start_time": "2023-06-01T12:00:00Z",
  "current_time": "2023-06-01T13:00:00Z",
  "resource_usage": {
    "cpu_percent": 15.2,
    "memory_percent": 45.6,
    "memory_rss_mb": 256.3,
    "memory_vms_mb": 512.7,
    "disk_percent": 32.1
  }
}
```

### Adding Health Checks

Health checks are automatically set up for all services and analyzers registered with the `ServiceContainer`. You can also add custom health checks:

```python
from analysis_engine.monitoring.health_checks import (
    check_database_connection,
    check_redis_connection,
    check_service_connection
)

# Get the health check instance
health_check = service_container.health_check

# Add a database health check
health_check.add_dependency_check(
    "database",
    lambda: check_database_connection(db_client, "database")
)

# Add a Redis health check
health_check.add_dependency_check(
    "redis",
    lambda: check_redis_connection(redis_client, "redis")
)

# Add a service health check
health_check.add_dependency_check(
    "feature-store-service",
    lambda: check_service_connection(
        "http://feature-store-service:8000",
        "feature-store-service"
    )
)
```

## Dashboards

The Analysis Engine Service includes a Grafana dashboard for visualizing service metrics. The dashboard is defined in the `monitoring/grafana-dashboard.json` file.

### Dashboard Panels

The dashboard includes the following panels:

- **Service Overview**:
  - Dependency Health
  - Resource Usage
  - API Request Rate

- **Analysis Performance**:
  - Analysis Duration (p95)
  - Analysis Request Rate
  - Analysis Error Rate
  - Dependency Latency (p95)

- **API Performance**:
  - API Request Duration (p95)
  - API Status Codes

- **Cache Performance**:
  - Cache Operations
  - Cache Hit Rate

### Importing the Dashboard

To import the dashboard into Grafana:

1. Go to Dashboards > Import
2. Upload the `grafana-dashboard.json` file or paste its contents
3. Configure the Prometheus data source

## Alerts

The Analysis Engine Service includes Prometheus alerting rules for monitoring service health. The rules are defined in the `monitoring/alerts.yml` file.

### Key Alerts

- **HighErrorRate**: Alert when the error rate for an operation exceeds 5% for 5 minutes
- **HighLatency**: Alert when the p95 latency for an operation exceeds 2 seconds for 5 minutes
- **DependencyUnhealthy**: Alert when a dependency is unhealthy for 5 minutes
- **HighResourceUsage**: Alert when memory usage exceeds 85% for 10 minutes
- **HighCPUUsage**: Alert when CPU usage exceeds 90% for 10 minutes
- **LowCacheHitRate**: Alert when the cache hit rate falls below 50% for 15 minutes
- **APIErrorRate**: Alert when the API error rate exceeds 1% for 5 minutes

### Configuring Alerts

To configure alerts in Prometheus:

1. Add the `alerts.yml` file to your Prometheus configuration
2. Update the `prometheus.yml` file to include the alerts:

```yaml
rule_files:
  - "alerts.yml"
```

## Conclusion

The Analysis Engine Service provides comprehensive monitoring capabilities that help you track service performance, detect issues, and debug problems. By using these capabilities, you can ensure that the service is running smoothly and quickly identify and resolve any issues that arise.
