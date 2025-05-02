# Resilience Module for Analysis Engine Service

This module provides resilience patterns for the Analysis Engine Service, building on the common-lib resilience module.

## Key Components

1. **Circuit Breakers** - Prevent cascading failures by stopping calls to failing services
2. **Retry Mechanisms** - Automatically retry temporary failures with exponential backoff
3. **Bulkheads** - Isolate critical operations to prevent resource contention
4. **Timeout Handling** - Prevent hanging operations from affecting the service

## Usage

### Circuit Breakers

```python
from analysis_engine.resilience import create_circuit_breaker

# Create a circuit breaker
cb = create_circuit_breaker(
    service_name="analysis_engine",
    resource_name="feature_store",
    config=get_circuit_breaker_config("feature_store")
)

# Use the circuit breaker
result = cb.execute(lambda: api_client.make_request())
```

### Retry Mechanisms

```python
from analysis_engine.resilience import retry_with_policy

# Apply retry policy to a function
@retry_with_policy(
    max_attempts=3,
    base_delay=1.0,
    jitter=True,
    exceptions=[ConnectionError, TimeoutError]
)
def fetch_data():
    # This function will retry up to 3 times on ConnectionError or TimeoutError
    ...
```

### Bulkheads

```python
from analysis_engine.resilience import bulkhead

# Apply bulkhead to a function
@bulkhead(
    name="database-operations",
    max_concurrent=10,
    max_waiting=5
)
def query_database():
    # This function will be limited to 10 concurrent executions
    # with a maximum of 5 waiting in the queue
    ...
```

### Timeout Handling

```python
from analysis_engine.resilience import timeout_handler

# Apply timeout to a function
@timeout_handler(timeout_seconds=5.0)
def slow_operation():
    # This operation will be terminated if it takes more than 5 seconds
    ...
```

### Combined Resilience Patterns

```python
from analysis_engine.resilience.utils import with_resilience

# Apply all resilience patterns to a function
@with_resilience(
    service_name="analysis_engine",
    operation_name="get_market_data",
    service_type="feature_store",
    exceptions=[requests.RequestException, requests.Timeout]
)
def get_market_data(symbol, timeframe):
    # This function will have circuit breaker, retry, bulkhead, and timeout
    ...
```

## Resilient Clients

### HTTP Client

```python
from analysis_engine.resilience.http_client import get_http_client

# Get a resilient HTTP client
http_client = get_http_client(service_type="feature_store")

# Make a request with resilience patterns
response = http_client.get("https://api.example.com/data")
```

### Database Client

```python
from analysis_engine.resilience.database import get_db_manager, execute_query

# Execute a database query with resilience patterns
result = execute_query(lambda session: session.query(Model).filter_by(id=1).first())

# Or use the context manager
with get_db_manager().get_db_session() as session:
    result = session.query(Model).filter_by(id=1).first()
```

### Redis Client

```python
from analysis_engine.resilience.redis_client import get_redis_client, execute_redis_operation

# Execute a Redis operation with resilience patterns
result = execute_redis_operation(lambda redis: redis.get("key"))
```
