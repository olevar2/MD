# Resilience Module

This module provides resilience patterns for robust service communication in the Forex trading platform.

## Key Components

1. **Circuit Breaker** - Prevents cascading failures by stopping calls to failing services
2. **Retry Policy** - Automatically retries temporary failures with exponential backoff
3. **Timeout Handler** - Ensures operations complete within specific time constraints
4. **Bulkhead Pattern** - Isolates failures by partitioning resources
5. **Combined Resilience** - Applies multiple resilience patterns together

## Usage

### Traditional API

```python
# Circuit Breaker example
from common_lib.resilience import CircuitBreaker, CircuitBreakerConfig, create_circuit_breaker

# Create a circuit breaker
cb = create_circuit_breaker(
    service_name="my-service",
    resource_name="api-endpoint",
    config=CircuitBreakerConfig(
        failure_threshold=5,
        reset_timeout_seconds=30
    )
)

# Use the circuit breaker
async def call_api():
    return await cb.execute(api_client.make_request)


# Retry Policy example
from common_lib.resilience import retry_with_policy, register_common_retryable_exceptions

# Basic usage
@retry_with_policy(
    max_attempts=3,
    base_delay=0.5,
    exceptions=[ConnectionError, TimeoutError]
)
async def fetch_data():
    # This function will retry up to 3 times on ConnectionError or TimeoutError
    ...

# Using pre-registered exceptions
@retry_with_policy(
    max_attempts=3,
    base_delay=0.5,
    exceptions=register_common_retryable_exceptions(),
    service_name="my-service",
    operation_name="fetch_data"
)
def fetch_with_monitoring():
    # This function will retry common network-related exceptions
    # and report metrics for monitoring
    ...


# Timeout Handler example
from common_lib.resilience import timeout_handler

@timeout_handler(timeout_seconds=1.5)
async def slow_operation():
    # This operation will be terminated if it takes more than 1.5 seconds
    ...


# Bulkhead example
from common_lib.resilience import bulkhead

@bulkhead(
    name="database-operations",
    max_concurrent=10,
    max_waiting=5
)
async def query_database():
    # This function will be limited to 10 concurrent executions
    # with a maximum of 5 waiting in the queue
    ...
```

### Standardized Decorators (Recommended)

```python
# Circuit Breaker example
from common_lib.resilience import circuit_breaker

@circuit_breaker(
    service_name="my-service",
    resource_name="external-api",
    failure_threshold=5,
    recovery_timeout=30.0
)
async def call_external_api():
    # This function will be protected by a circuit breaker
    # After 5 failures, the circuit will open for 30 seconds
    ...


# Retry with Backoff example
from common_lib.resilience import retry_with_backoff

@retry_with_backoff(
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0,
    backoff_factor=2.0,
    jitter=True
)
async def fetch_data():
    # This function will retry up to 3 times with exponential backoff
    # The delay between retries will increase with each attempt
    ...


# Timeout example
from common_lib.resilience import timeout

@timeout(timeout_seconds=5.0)
async def slow_operation():
    # This operation will be terminated if it takes more than 5 seconds
    ...


# Bulkhead example
from common_lib.resilience import bulkhead

@bulkhead(
    max_concurrent=10,
    max_queue=20,
    name="database-operations"
)
async def query_database():
    # This function will be limited to 10 concurrent executions
    # with a maximum of 20 waiting in the queue
    ...


# Combined Resilience example
from common_lib.resilience import with_resilience

@with_resilience(
    # Circuit breaker config
    enable_circuit_breaker=True,
    failure_threshold=5,
    recovery_timeout=30.0,
    # Retry config
    enable_retry=True,
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0,
    backoff_factor=2.0,
    jitter=True,
    # Bulkhead config
    enable_bulkhead=True,
    max_concurrent=10,
    max_queue=20,
    # Timeout config
    enable_timeout=True,
    timeout_seconds=5.0
)
async def resilient_operation():
    # This function is protected by all resilience patterns
    ...
```

For more examples, see `usage_demos/resilience_examples.py`.

## Documentation

For detailed documentation, see `docs/resilience_patterns.md`.

## Testing

Run the tests with:

```
python -m unittest tests/test_resilience.py
```

or use the provided scripts:

```
./test_resilience.bat  # Windows batch file
./test_resilience.ps1  # PowerShell script
```
