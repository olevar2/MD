# Resilience Module

This module provides resilience patterns for robust service communication in the Forex trading platform.

## Key Components

1. **Circuit Breaker** - Prevents cascading failures by stopping calls to failing services
2. **Retry Policy** - Automatically retries temporary failures with exponential backoff
3. **Timeout Handler** - Ensures operations complete within specific time constraints
4. **Bulkhead Pattern** - Isolates failures by partitioning resources

## Usage

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

# Class-based configuration
class ApiClient:
    def __init__(self):
        self.max_attempts = 3
        self.base_delay = 1.0
        
    @retry_with_policy(
        max_attempts=lambda self: self.max_attempts,
        base_delay=lambda self: self.base_delay,
        exceptions=[ConnectionError]
    )
    def call_api(self):
        # Uses class attributes for configuration
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
