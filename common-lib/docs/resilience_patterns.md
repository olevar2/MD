# Resilience Patterns for Forex Trading Platform

This document provides an overview of the resilience patterns implemented in the common-lib package for the Forex Trading Platform.

## Overview

Modern distributed systems face numerous challenges such as network failures, latency issues, and service outages. The resilience patterns in this package help services handle these challenges gracefully.

## Key Patterns

### Circuit Breaker

The Circuit Breaker pattern prevents a cascade of failures when a downstream service is failing. It automatically detects failures and stops operation calls.

**Key Features:**
- Failure threshold configuration
- Automatic recovery with half-open state
- Metrics collection
- Integration with monitoring

```python
from common_lib.resilience import create_circuit_breaker, CircuitBreakerConfig

# Configure the circuit breaker
config = CircuitBreakerConfig(
    failure_threshold=3,
    reset_timeout_seconds=60
)

cb = create_circuit_breaker("my-service", "external-api", config)

# Use the circuit breaker
result = await cb.execute(my_api_call_function)
```

### Retry Policy

The Retry Policy pattern automatically retries temporary failures using configurable strategies.

**Key Features:**
- Exponential backoff with jitter
- Configurable retry attempts
- Specific exception filtering
- Database-specific retry support

```python
from common_lib.resilience import retry_with_policy

@retry_with_policy(
    max_attempts=3,
    base_delay=1.0,
    jitter=True
)
async def fetch_data():
    # This function will be retried up to 3 times if it raises exceptions
    ...
```

### Timeout Handler

The Timeout Handler prevents operations from blocking indefinitely by enforcing configurable timeouts.

**Key Features:**
- Support for both sync and async functions
- Operation-specific timeout configuration
- Custom timeout error messages

```python
from common_lib.resilience import timeout_handler

@timeout_handler(timeout_seconds=5.0)
async def potentially_slow_operation():
    # This function will fail if it takes more than 5 seconds
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
from common_lib.resilience import bulkhead

@bulkhead(name="database-queries", max_concurrent=10)
async def query_database(query):
    # Only 10 concurrent calls to this function will be allowed
    ...
```

## Combining Patterns

These patterns can be combined to provide comprehensive protection:

```python
@retry_with_policy(max_attempts=3)
@timeout_handler(timeout_seconds=2.0)
@bulkhead(name="external-api", max_concurrent=5)
async def call_external_api(endpoint):
    # This function is protected by multiple resilience patterns
    ...
```

## Testing Resilience Patterns

The test suite demonstrates how to test resilience patterns to ensure they behave as expected. Key tests include:

1. **Circuit Breaker Tests**: Verify that the circuit opens after failures and resets after timeout
2. **Retry Policy Tests**: Ensure that transient failures are retried correctly
3. **Timeout Tests**: Confirm that long-running operations are cancelled appropriately
4. **Bulkhead Tests**: Validate that concurrency limits are enforced correctly

Run the tests using:

```
python run_tests.py
```

## Usage Examples

For complete working examples, see the `usage_demos/resilience_examples.py` file.
