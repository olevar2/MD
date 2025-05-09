# Metrics Naming Convention

This document defines the standard naming patterns, label sets, and metric types for all services in the forex trading platform.

## Metric Naming

All metrics should follow the pattern:

```
[domain]_[entity]_[action]_[unit]
```

Where:
- `domain`: The domain area (e.g., `api`, `db`, `cache`, `broker`, `model`, `strategy`, `feature`, `market`, `system`)
- `entity`: The entity being measured (e.g., `request`, `operation`, `connection`, `prediction`)
- `action`: The action being measured (e.g., `duration`, `count`, `size`, `rate`)
- `unit`: The unit of measurement (e.g., `seconds`, `bytes`, `total`, `ratio`)

Examples:
- `api_request_duration_seconds`: Duration of API requests in seconds
- `db_operation_count_total`: Total number of database operations
- `cache_hit_ratio`: Ratio of cache hits to total cache operations
- `model_prediction_latency_seconds`: Latency of model predictions in seconds

## Label Naming

Labels should be consistent across all metrics. Use the following standard labels:

- `service`: The name of the service (e.g., `analysis-engine-service`, `trading-gateway-service`)
- `endpoint`: The API endpoint (e.g., `/api/v1/analysis`, `/api/v1/orders`)
- `method`: The HTTP method (e.g., `GET`, `POST`, `PUT`, `DELETE`)
- `status_code`: The HTTP status code (e.g., `200`, `400`, `500`)
- `error_type`: The type of error (e.g., `validation_error`, `database_error`, `timeout`)
- `operation`: The operation being performed (e.g., `read`, `write`, `update`, `delete`)
- `database`: The database being accessed (e.g., `postgres`, `timescale`, `redis`)
- `table`: The database table being accessed (e.g., `orders`, `trades`, `users`)
- `cache_type`: The type of cache (e.g., `redis`, `local`, `distributed`)
- `broker`: The broker being accessed (e.g., `interactive_brokers`, `oanda`, `fxcm`)
- `instrument`: The financial instrument (e.g., `EUR_USD`, `USD_JPY`, `GBP_USD`)
- `timeframe`: The timeframe (e.g., `1m`, `5m`, `15m`, `1h`, `4h`, `1d`)
- `strategy`: The trading strategy (e.g., `trend_following`, `mean_reversion`, `breakout`)
- `model`: The ML model (e.g., `price_prediction`, `regime_detection`, `pattern_recognition`)
- `feature`: The feature (e.g., `price`, `volume`, `volatility`, `momentum`)
- `component`: The component (e.g., `api`, `database`, `cache`, `broker`)
- `instance`: The instance (e.g., `instance-1`, `instance-2`, `instance-3`)

## Metric Types

### Counters

Counters should be used for metrics that only increase over time, such as the number of requests, errors, or operations.

Naming pattern: `[domain]_[entity]_[action]_total`

Examples:
- `api_requests_total`: Total number of API requests
- `api_errors_total`: Total number of API errors
- `db_operations_total`: Total number of database operations

### Gauges

Gauges should be used for metrics that can go up and down, such as current resource usage, connection counts, or queue sizes.

Naming pattern: `[domain]_[entity]_[unit]`

Examples:
- `system_cpu_usage_percent`: CPU usage percentage
- `system_memory_usage_bytes`: Memory usage in bytes
- `db_connections`: Number of database connections
- `cache_size`: Number of items in the cache

### Histograms

Histograms should be used for metrics that measure distributions, such as request durations, response sizes, or batch sizes.

Naming pattern: `[domain]_[entity]_[action]_[unit]`

Examples:
- `api_request_duration_seconds`: API request duration in seconds
- `api_response_size_bytes`: API response size in bytes
- `db_operation_duration_seconds`: Database operation duration in seconds

### Summaries

Summaries should be used for metrics that calculate quantiles, such as request durations or operation latencies.

Naming pattern: `[domain]_[entity]_[action]_[unit]_summary`

Examples:
- `api_request_duration_seconds_summary`: API request duration in seconds (summary)
- `db_operation_duration_seconds_summary`: Database operation duration in seconds (summary)

## Standard Buckets

### Latency Buckets

For latency histograms, use the following standard buckets:

- Fast operations (API requests, cache operations):
  ```
  [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
  ```

- Medium operations (database operations, broker operations):
  ```
  [0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
  ```

- Slow operations (batch processing, model training):
  ```
  [0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
  ```

### Size Buckets

For size histograms, use the following standard buckets:

```
[1024, 10*1024, 100*1024, 1024*1024, 10*1024*1024, 100*1024*1024, 1024*1024*1024]
```

### Count Buckets

For count histograms, use the following standard buckets:

```
[1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
```

## Examples

### API Metrics

```python
# API request counter
api_requests_total = get_counter(
    name="api_requests_total",
    description="Total number of API requests",
    labels=["service", "endpoint", "method", "status_code"]
)

# API request duration
api_request_duration_seconds = get_histogram(
    name="api_request_duration_seconds",
    description="API request duration in seconds",
    labels=["service", "endpoint", "method"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# API errors
api_errors_total = get_counter(
    name="api_errors_total",
    description="Total number of API errors",
    labels=["service", "endpoint", "method", "error_type"]
)
```

### Database Metrics

```python
# Database operations counter
db_operations_total = get_counter(
    name="db_operations_total",
    description="Total number of database operations",
    labels=["service", "database", "operation", "table"]
)

# Database operation duration
db_operation_duration_seconds = get_histogram(
    name="db_operation_duration_seconds",
    description="Database operation duration in seconds",
    labels=["service", "database", "operation", "table"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
)

# Database connections
db_connections = get_gauge(
    name="db_connections",
    description="Number of database connections",
    labels=["service", "database"]
)
```

### System Metrics

```python
# System health
system_health = get_gauge(
    name="system_health",
    description="System health status (1 = healthy, 0 = unhealthy)",
    labels=["service", "component"]
)

# System resource usage
system_cpu_usage_percent = get_gauge(
    name="system_cpu_usage_percent",
    description="CPU usage percentage",
    labels=["service", "instance"]
)

system_memory_usage_bytes = get_gauge(
    name="system_memory_usage_bytes",
    description="Memory usage in bytes",
    labels=["service", "instance"]
)
```
