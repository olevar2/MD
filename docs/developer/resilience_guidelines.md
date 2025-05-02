# Retry Logic Guidelines

## Overview

This guide explains how to properly implement retry logic in the Forex Trading Platform using the centralized resilience patterns from `common_lib.resilience`.

## Basic Usage

### Importing the Retry Decorator

Always import retry functionality from `common_lib.resilience` (not from core-foundations):

```python
from common_lib.resilience import retry_with_policy, RetryExhaustedException
```

### Simple Retry with Default Settings

For basic retry functionality:

```python
from common_lib.resilience import retry_with_policy

@retry_with_policy(
    max_attempts=3,
    base_delay=1.0,
    exceptions=[ConnectionError, TimeoutError]
)
def fetch_data():
    # This function will retry up to 3 times with exponential backoff
    # starting at 1.0 seconds if ConnectionError or TimeoutError occurs
    ...
```

### Retry with Service Identification

For better monitoring and observability:

```python
@retry_with_policy(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    backoff_factor=2.0,
    exceptions=[ConnectionError, TimeoutError],
    service_name="feature-store",
    operation_name="fetch_historical_data"
)
def fetch_historical_data(symbol, timeframe):
    ...
```

## Advanced Usage

### Class-based Configuration

For class methods where configuration is stored in class attributes:

```python
class ApiClient:
    def __init__(self):
        self.max_attempts = 3
        self.base_delay = 1.0
        self.max_delay = 30.0
        self.backoff_factor = 2.0
        self.retryable_exceptions = [ConnectionError, TimeoutError]
        
    @retry_with_policy(
        exceptions=lambda self: self.retryable_exceptions,
        max_attempts=lambda self: self.max_attempts,
        base_delay=lambda self: self.base_delay,
        max_delay=lambda self: self.max_delay,
        backoff_factor=lambda self: self.backoff_factor,
        service_name="my-service",
        operation_name="api_call"
    )
    def api_call(self):
        ...
```

### Adding Jitter

To prevent thundering herd problems:

```python
@retry_with_policy(
    max_attempts=3,
    base_delay=1.0,
    jitter=True,  # Adds randomization to delay times
    exceptions=[ConnectionError]
)
def call_remote_service():
    ...
```

### Using Pre-registered Exceptions

For common scenarios:

```python
from common_lib.resilience import retry_with_policy, register_common_retryable_exceptions

@retry_with_policy(
    max_attempts=3,
    base_delay=1.0,
    exceptions=register_common_retryable_exceptions()
)
def network_operation():
    ...
```

For database operations:

```python
from common_lib.resilience import retry_with_policy, register_database_retryable_exceptions

@retry_with_policy(
    max_attempts=5,
    base_delay=0.5,
    exceptions=register_database_retryable_exceptions()
)
def database_operation():
    ...
```

### Async Functions

The decorator automatically detects and properly handles async functions:

```python
@retry_with_policy(
    max_attempts=3,
    base_delay=1.0,
    exceptions=[ConnectionError]
)
async def fetch_data_async():
    ...
```

### Handling Retry Exhaustion

```python
from common_lib.resilience import retry_with_policy, RetryExhaustedException

try:
    result = fetch_data()
except RetryExhaustedException as e:
    # Handle the case where all retries failed
    logger.error(f"All retries exhausted: {e}")
    # Fall back to default behavior or raise custom exception
```

## Metrics and Monitoring

The retry policy automatically reports metrics if a metric_handler is provided:

```python
from common_lib.monitoring import metrics

@retry_with_policy(
    max_attempts=3,
    exceptions=[ConnectionError],
    metric_handler=metrics.record_metric
)
def monitored_operation():
    ...
```

## Best Practices

1. **Be Specific with Exceptions**: Only retry for exceptions that are likely to be transient.
2. **Set Reasonable Limits**: Don't retry too many times or with delays that are too long.
3. **Use Jitter**: For services with high concurrency, always use jitter to prevent synchronized retry storms.
4. **Include Service/Operation Names**: This helps with monitoring and debugging.
5. **Handle Non-Retryable Cases**: Some errors shouldn't be retried (e.g., authentication errors).
6. **Use Pre-Registered Exceptions**: Use `register_common_retryable_exceptions()` and `register_database_retryable_exceptions()` for common scenarios.

## Migration from Old Retry Implementations

If you're migrating from the older core_foundations.resilience.retry:

1. Change import from:
   ```python
   from core_foundations.resilience.retry import async_retry, RetryConfig
   ```
   to:
   ```python
   from common_lib.resilience import retry_with_policy
   ```

2. Replace RetryConfig class instantiation with decorator parameters:

   Old:
   ```python
   self.retry_config = RetryConfig(
       max_retries=3,
       backoff_factor=1.5,
       max_backoff=30
   )
   
   @async_retry(retry_config_attr="retry_config")
   async def my_method(self):
       ...
   ```

   New:
   ```python
   self.max_attempts = 3
   self.backoff_factor = 1.5
   self.max_delay = 30
   
   @retry_with_policy(
       max_attempts=lambda self: self.max_attempts,
       backoff_factor=lambda self: self.backoff_factor,
       max_delay=lambda self: self.max_delay
   )
   async def my_method(self):
       ...
   ```
