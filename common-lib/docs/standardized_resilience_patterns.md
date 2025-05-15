# Standardized Resilience Patterns

This document provides a comprehensive guide to the standardized resilience patterns implemented in the common-lib package for the Forex Trading Platform.

## Overview

Modern distributed systems face numerous challenges such as network failures, latency issues, and service outages. The standardized resilience patterns in this package help services handle these challenges gracefully and consistently across the platform.

## Key Patterns

### Circuit Breaker

The Circuit Breaker pattern prevents a cascade of failures when a downstream service is failing. It automatically detects failures and stops operation calls.

**Key Features:**
- Failure threshold configuration
- Automatic recovery with half-open state
- Metrics collection
- Integration with monitoring

```python
from common_lib.resilience import with_standard_circuit_breaker

@with_standard_circuit_breaker(
    service_name="my-service",
    resource_name="external-api",
    service_type="external-api"
)
async def call_external_api():
    # This function will be protected by a circuit breaker
    # After failures exceed the threshold, the circuit will open
    ...
```

### Retry Policy

The Retry Policy pattern automatically retries temporary failures using configurable strategies.

**Key Features:**
- Exponential backoff with jitter
- Configurable retry attempts
- Specific exception filtering
- Database-specific retry support

```python
from common_lib.resilience import with_standard_retry

@with_standard_retry(
    service_name="my-service",
    operation_name="fetch-data",
    service_type="external-api",
    exceptions=[ConnectionError, TimeoutError]
)
async def fetch_data():
    # This function will retry with exponential backoff
    # The delay between retries will increase with each attempt
    ...
```

### Bulkhead Pattern

The Bulkhead pattern isolates failures by partitioning service resources, preventing one failing component from taking down the entire system.

**Key Features:**
- Configurable concurrent execution limits
- Wait queuing with limits
- Timeout configuration for waiting
- Metrics for monitoring

```python
from common_lib.resilience import with_standard_bulkhead

@with_standard_bulkhead(
    service_name="my-service",
    operation_name="database-query",
    service_type="database"
)
async def query_database():
    # This function will be limited to a configurable number of concurrent executions
    # with a maximum number waiting in the queue
    ...
```

### Timeout Pattern

The Timeout pattern ensures operations complete within specific time constraints, preventing hanging operations from affecting the service.

**Key Features:**
- Configurable timeout duration
- Automatic cancellation of long-running operations
- Integration with monitoring
- Customizable timeout handling

```python
from common_lib.resilience import with_standard_timeout

@with_standard_timeout(
    service_name="my-service",
    operation_name="slow-operation",
    service_type="external-api"
)
async def slow_operation():
    # This operation will be terminated if it takes too long
    ...
```

### Combined Resilience

The Combined Resilience pattern applies multiple resilience patterns together, providing comprehensive protection for service calls.

**Key Features:**
- Configurable combination of circuit breaker, retry, bulkhead, and timeout
- Standardized configuration for different service types
- Simplified application through decorators
- Consistent behavior across services

```python
from common_lib.resilience import with_standard_resilience

@with_standard_resilience(
    service_name="my-service",
    operation_name="critical-operation",
    service_type="critical"
)
async def critical_operation():
    # This function is protected by all resilience patterns
    # with configurations appropriate for critical operations
    ...
```

## Specialized Resilience Patterns

The standardized resilience patterns include specialized configurations for different types of operations:

### Database Resilience

```python
from common_lib.resilience import with_database_resilience

@with_database_resilience(
    service_name="my-service",
    operation_name="database-operation"
)
async def database_operation():
    # This function is protected by resilience patterns
    # with configurations appropriate for database operations
    ...
```

### Broker API Resilience

```python
from common_lib.resilience import with_broker_api_resilience

@with_broker_api_resilience(
    service_name="my-service",
    operation_name="broker-operation"
)
async def broker_operation():
    # This function is protected by resilience patterns
    # with configurations appropriate for broker API operations
    ...
```

### Market Data Resilience

```python
from common_lib.resilience import with_market_data_resilience

@with_market_data_resilience(
    service_name="my-service",
    operation_name="market-data-operation"
)
async def market_data_operation():
    # This function is protected by resilience patterns
    # with configurations appropriate for market data operations
    ...
```

### External API Resilience

```python
from common_lib.resilience import with_external_api_resilience

@with_external_api_resilience(
    service_name="my-service",
    operation_name="external-api-operation"
)
async def external_api_operation():
    # This function is protected by resilience patterns
    # with configurations appropriate for external API operations
    ...
```

### Critical Service Resilience

```python
from common_lib.resilience import with_critical_resilience

@with_critical_resilience(
    service_name="my-service",
    operation_name="critical-operation"
)
async def critical_operation():
    # This function is protected by resilience patterns
    # with configurations appropriate for critical operations
    ...
```

### High-Throughput Service Resilience

```python
from common_lib.resilience import with_high_throughput_resilience

@with_high_throughput_resilience(
    service_name="my-service",
    operation_name="high-throughput-operation"
)
async def high_throughput_operation():
    # This function is protected by resilience patterns
    # with configurations appropriate for high-throughput operations
    ...
```

## Standardized Configuration

The standardized resilience patterns use predefined configurations for different service types:

### Critical Service Configuration

```python
CRITICAL_SERVICE = ResilienceConfig(
    service_name="critical-service",
    operation_name="critical-operation",
    circuit_breaker=CircuitBreakerConfig(
        failure_threshold=3,
        reset_timeout_seconds=60.0,
        half_open_max_calls=1
    ),
    retry=RetryConfig(
        max_attempts=5,
        base_delay=0.5,
        max_delay=30.0,
        backoff_factor=2.0,
        jitter=True
    ),
    bulkhead=BulkheadConfig(
        max_concurrent=5,
        max_queue=10
    ),
    timeout=TimeoutConfig(
        timeout_seconds=10.0
    )
)
```

### Standard Service Configuration

```python
STANDARD_SERVICE = ResilienceConfig(
    service_name="standard-service",
    operation_name="standard-operation",
    circuit_breaker=CircuitBreakerConfig(
        failure_threshold=5,
        reset_timeout_seconds=30.0,
        half_open_max_calls=2
    ),
    retry=RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        max_delay=60.0,
        backoff_factor=2.0,
        jitter=True
    ),
    bulkhead=BulkheadConfig(
        max_concurrent=10,
        max_queue=20
    ),
    timeout=TimeoutConfig(
        timeout_seconds=30.0
    )
)
```

### High-Throughput Service Configuration

```python
HIGH_THROUGHPUT_SERVICE = ResilienceConfig(
    service_name="high-throughput-service",
    operation_name="high-throughput-operation",
    circuit_breaker=CircuitBreakerConfig(
        failure_threshold=10,
        reset_timeout_seconds=15.0,
        half_open_max_calls=5
    ),
    retry=RetryConfig(
        max_attempts=2,
        base_delay=0.1,
        max_delay=5.0,
        backoff_factor=2.0,
        jitter=True
    ),
    bulkhead=BulkheadConfig(
        max_concurrent=50,
        max_queue=100
    ),
    timeout=TimeoutConfig(
        timeout_seconds=5.0
    )
)
```

### Database Operation Configuration

```python
DATABASE_OPERATION = ResilienceConfig(
    service_name="database-service",
    operation_name="database-operation",
    circuit_breaker=CircuitBreakerConfig(
        failure_threshold=3,
        reset_timeout_seconds=60.0,
        half_open_max_calls=1
    ),
    retry=RetryConfig(
        max_attempts=3,
        base_delay=0.5,
        max_delay=10.0,
        backoff_factor=2.0,
        jitter=True
    ),
    bulkhead=BulkheadConfig(
        max_concurrent=20,
        max_queue=30
    ),
    timeout=TimeoutConfig(
        timeout_seconds=15.0
    )
)
```

### External API Configuration

```python
EXTERNAL_API = ResilienceConfig(
    service_name="external-api",
    operation_name="api-call",
    circuit_breaker=CircuitBreakerConfig(
        failure_threshold=3,
        reset_timeout_seconds=60.0,
        half_open_max_calls=1
    ),
    retry=RetryConfig(
        max_attempts=5,
        base_delay=1.0,
        max_delay=30.0,
        backoff_factor=2.0,
        jitter=True
    ),
    bulkhead=BulkheadConfig(
        max_concurrent=10,
        max_queue=20
    ),
    timeout=TimeoutConfig(
        timeout_seconds=10.0
    )
)
```

### Broker API Configuration

```python
BROKER_API = ResilienceConfig(
    service_name="broker-api",
    operation_name="broker-call",
    circuit_breaker=CircuitBreakerConfig(
        failure_threshold=3,
        reset_timeout_seconds=120.0,
        half_open_max_calls=1
    ),
    retry=RetryConfig(
        max_attempts=3,
        base_delay=2.0,
        max_delay=60.0,
        backoff_factor=2.0,
        jitter=True
    ),
    bulkhead=BulkheadConfig(
        max_concurrent=5,
        max_queue=10
    ),
    timeout=TimeoutConfig(
        timeout_seconds=30.0
    )
)
```

### Market Data Configuration

```python
MARKET_DATA = ResilienceConfig(
    service_name="market-data",
    operation_name="data-fetch",
    circuit_breaker=CircuitBreakerConfig(
        failure_threshold=5,
        reset_timeout_seconds=30.0,
        half_open_max_calls=2
    ),
    retry=RetryConfig(
        max_attempts=3,
        base_delay=0.5,
        max_delay=5.0,
        backoff_factor=2.0,
        jitter=True
    ),
    bulkhead=BulkheadConfig(
        max_concurrent=20,
        max_queue=50
    ),
    timeout=TimeoutConfig(
        timeout_seconds=5.0
    )
)
```

## Custom Configuration

You can customize the resilience configuration for specific operations:

```python
from common_lib.resilience import with_standard_resilience

@with_standard_resilience(
    service_name="my-service",
    operation_name="custom-operation",
    service_type="critical",
    config={
        "circuit_breaker": {
            "failure_threshold": 10,
            "reset_timeout_seconds": 120.0
        },
        "retry": {
            "max_attempts": 10,
            "base_delay": 0.1
        }
    }
)
async def custom_operation():
    # This function is protected by resilience patterns
    # with custom configurations
    ...
```

## Factory Functions

The standardized resilience patterns include factory functions for creating resilience components:

```python
from common_lib.resilience import (
    get_circuit_breaker,
    get_retry_policy,
    get_bulkhead,
    get_timeout,
    get_resilience
)

# Get a circuit breaker
circuit_breaker = get_circuit_breaker(
    service_name="my-service",
    resource_name="external-api",
    service_type="external-api"
)

# Get a retry policy
retry_policy = get_retry_policy(
    service_name="my-service",
    operation_name="fetch-data",
    service_type="external-api",
    exceptions=[ConnectionError, TimeoutError]
)

# Get a bulkhead
bulkhead = get_bulkhead(
    service_name="my-service",
    operation_name="database-query",
    service_type="database"
)

# Get a timeout
timeout = get_timeout(
    service_name="my-service",
    operation_name="slow-operation",
    service_type="external-api"
)

# Get a combined resilience
resilience = get_resilience(
    service_name="my-service",
    operation_name="critical-operation",
    service_type="critical"
)
```

## Applying Resilience Patterns

You can apply resilience patterns to existing code using the `apply_standardized_resilience.py` script:

```bash
python tools/script/fix\ platform\ scripts/apply_standardized_resilience.py --directory path/to/service --output resilience_report.json --apply
```

This script will:
1. Analyze the codebase for functions that need resilience patterns
2. Generate a report of functions needing resilience patterns
3. Apply resilience patterns to functions that need them (if `--apply` is specified)

## Best Practices

1. **Use Specialized Decorators**: Use the specialized decorators (`with_database_resilience`, `with_broker_api_resilience`, etc.) for specific types of operations.

2. **Be Consistent**: Use the same resilience patterns for similar operations across services.

3. **Monitor Resilience Metrics**: Monitor the metrics generated by the resilience patterns to identify issues and optimize configurations.

4. **Test Resilience**: Test the resilience patterns with chaos testing to ensure they work as expected.

5. **Document Resilience**: Document the resilience patterns used in each service to help developers understand the system's behavior.

6. **Review Configurations**: Regularly review and update the resilience configurations based on production experience.

7. **Use Factory Functions**: Use the factory functions to create resilience components with standardized configurations.

8. **Apply Resilience Patterns Automatically**: Use the `apply_standardized_resilience.py` script to automatically apply resilience patterns to existing code.

## Conclusion

The standardized resilience patterns provide a consistent and comprehensive approach to handling failures in the Forex Trading Platform. By using these patterns, you can ensure that your services are robust and resilient to failures.