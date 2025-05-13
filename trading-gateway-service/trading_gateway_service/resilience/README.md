# Resilience Module for Trading Gateway Service

This module provides resilience patterns for the Trading Gateway Service, building on the common-lib resilience module.

## Key Components

1. **Circuit Breakers** - Prevent cascading failures by stopping calls to failing services
2. **Retry Mechanisms** - Automatically retry temporary failures with exponential backoff
3. **Bulkheads** - Isolate critical operations to prevent resource contention
4. **Timeout Handling** - Prevent hanging operations from affecting the service

## Usage

### Circuit Breakers

```python
from trading_gateway_service.resilience import create_circuit_breaker

# Create a circuit breaker
cb = create_circuit_breaker(
    service_name="trading_gateway",
    resource_name="broker_api",
    config=get_circuit_breaker_config("broker_api")
)

# Use the circuit breaker
result = await cb.execute(lambda: api_client.make_request())
```

### Retry Mechanisms

```python
from trading_gateway_service.resilience import retry_with_policy

# Apply retry policy to a function
@retry_with_policy(
    max_attempts=3,
    base_delay=1.0,
    jitter=True,
    exceptions=[ConnectionError, TimeoutError]
)
async def fetch_market_data():
    # This function will retry up to 3 times on ConnectionError or TimeoutError
    ...
```

### Bulkheads

```python
from trading_gateway_service.resilience import bulkhead

# Apply bulkhead to a function
@bulkhead(
    name="order-execution",
    max_concurrent=10,
    max_waiting=5
)
async def execute_order():
    # This function will be limited to 10 concurrent executions
    # with a maximum of 5 waiting in the queue
    ...
```

### Timeout Handling

```python
from trading_gateway_service.resilience import timeout_handler

# Apply timeout to a function
@timeout_handler(timeout_seconds=5.0)
async def place_order():
    # This operation will be terminated if it takes more than 5 seconds
    ...
```

### Combined Resilience Patterns

```python
from trading_gateway_service.resilience.utils import with_resilience

# Apply all resilience patterns to a function
@with_resilience(
    service_name="trading_gateway",
    operation_name="place_order",
    service_type="broker_api",
    exceptions=[requests.RequestException, requests.Timeout]
)
async def place_order(order_data):
    # This function will have circuit breaker, retry, bulkhead, and timeout
    ...
```

## Resilient Clients

### Broker API Client

```python
from trading_gateway_service.resilience.broker_client import get_broker_client

# Get a resilient broker client
broker_client = get_broker_client(broker_name="interactive_brokers")

# Make a request with resilience patterns
response = await broker_client.place_order(order_data)
```

### Market Data Client

```python
from trading_gateway_service.resilience.market_data_client import get_market_data_client

# Get a resilient market data client
market_data_client = get_market_data_client(provider="alpha_vantage")

# Make a request with resilience patterns
data = await market_data_client.get_historical_data(symbol="EURUSD", timeframe="1h")
```

## Configuration

The resilience patterns are configured in `config.py`. You can customize the configuration for different service types:

```python
# Circuit breaker configurations
CIRCUIT_BREAKER_CONFIGS = {
    # Broker API
    "broker_api": CircuitBreakerConfig(
        failure_threshold=5,
        reset_timeout_seconds=60,
    ),
    
    # Market data provider
    "market_data": CircuitBreakerConfig(
        failure_threshold=3,
        reset_timeout_seconds=30,
    ),
    
    # Order execution
    "order_execution": CircuitBreakerConfig(
        failure_threshold=3,
        reset_timeout_seconds=120,
    ),
}

# Retry configurations
RETRY_CONFIGS = {
    # Broker API
    "broker_api": {
        "max_attempts": 3,
        "base_delay": 1.0,
        "max_delay": 30.0,
        "backoff_factor": 2.0,
        "jitter": True,
    },
    
    # Market data provider
    "market_data": {
        "max_attempts": 5,
        "base_delay": 0.5,
        "max_delay": 10.0,
        "backoff_factor": 2.0,
        "jitter": True,
    },
}

# Bulkhead configurations
BULKHEAD_CONFIGS = {
    # Broker API
    "broker_api": {
        "max_concurrent": 10,
        "max_waiting": 20,
    },
    
    # Market data provider
    "market_data": {
        "max_concurrent": 20,
        "max_waiting": 50,
    },
    
    # Order execution
    "order_execution": {
        "max_concurrent": 5,
        "max_waiting": 10,
    },
}

# Timeout configurations
TIMEOUT_CONFIGS = {
    # Broker API
    "broker_api": 10.0,  # 10 seconds
    
    # Market data provider
    "market_data": 5.0,  # 5 seconds
    
    # Order execution
    "order_execution": 30.0,  # 30 seconds
}
```
