# Circuit Breaker Pattern

## Overview

The Circuit Breaker pattern prevents cascading failures by stopping calls to failing services. It works like an electrical circuit breaker, "tripping" when a service is failing and preventing further calls until the service recovers.

## When to Use

Use the Circuit Breaker pattern when:

- Calling external services that might fail
- Calling internal services that are critical but might be unstable
- You want to fail fast rather than wait for timeouts
- You need to protect your system from cascading failures

## Implementation in Forex Trading Platform

The Forex Trading Platform provides a `CircuitBreaker` class in the `common_lib.resilience` module:

```python
from common_lib.resilience import CircuitBreaker, CircuitBreakerConfig
from common_lib.exceptions import CircuitBreakerOpenError, ServiceUnavailableError

# Create a circuit breaker
cb = CircuitBreaker(
    service_name="portfolio-service",
    resource_name="trading-gateway",
    config=CircuitBreakerConfig(
        failure_threshold=5,     # Number of failures before opening
        reset_timeout_seconds=30, # Time before trying again
        half_open_max_calls=2     # Max calls in half-open state
    )
)

# Use the circuit breaker
async def execute_trade(trade_request):
    try:
        return await cb.execute(lambda: trading_gateway_client.execute_trade(trade_request))
    except CircuitBreakerOpenError:
        # Handle circuit breaker open
        logger.warning("Circuit breaker open for trading-gateway")
        raise ServiceUnavailableError(
            message="Trading service is currently unavailable",
            details={"retry_after_seconds": 30}
        )
```

## Circuit Breaker States

A circuit breaker has three states:

1. **Closed**: Normal operation, calls pass through to the service
2. **Open**: Service is considered unavailable, calls fail fast without reaching the service
3. **Half-Open**: After the reset timeout, allows a limited number of test calls to check if the service has recovered

## Configuration Parameters

| Parameter | Description | Recommended Value |
|-----------|-------------|------------------|
| `failure_threshold` | Number of failures before opening the circuit | 3-5 for critical services |
| `reset_timeout_seconds` | Time to wait before trying again | 30-60 seconds |
| `half_open_max_calls` | Maximum calls allowed in half-open state | 1-3 calls |
| `failure_window_seconds` | Time window for counting failures | 60 seconds |

## Best Practices

1. **Monitor Circuit Breaker State**: Log and alert when circuit breakers open
2. **Provide Fallbacks**: Implement fallback functionality when circuit breakers are open
3. **Set Appropriate Thresholds**: Configure thresholds based on service criticality
4. **Use with Retry**: Combine with retry for transient failures
5. **Separate Circuit Breakers**: Use different circuit breakers for different services

## Example: Market Data Service

```python
# Create circuit breakers for different external APIs
broker_api_cb = CircuitBreaker(
    service_name="market-data-service",
    resource_name="broker-api",
    config=CircuitBreakerConfig(
        failure_threshold=3,
        reset_timeout_seconds=60
    )
)

news_api_cb = CircuitBreaker(
    service_name="market-data-service",
    resource_name="news-api",
    config=CircuitBreakerConfig(
        failure_threshold=5,
        reset_timeout_seconds=30
    )
)

# Use circuit breakers with fallbacks
async def get_market_data(symbol):
    try:
        # Try to get real-time data from broker API
        return await broker_api_cb.execute(
            lambda: broker_api_client.get_market_data(symbol)
        )
    except CircuitBreakerOpenError:
        logger.warning("Broker API circuit open, using alternative source")
        
        try:
            # Try alternative data source
            return await news_api_cb.execute(
                lambda: news_api_client.get_market_indicators(symbol)
            )
        except CircuitBreakerOpenError:
            logger.warning("All data sources unavailable, using cached data")
            
            # Fall back to cached data
            cached_data = await cache.get(f"market_data:{symbol}")
            if cached_data:
                return {**cached_data, "source": "cache"}
            else:
                raise ServiceUnavailableError(
                    message="Market data is currently unavailable",
                    details={"symbol": symbol}
                )
```

## Related Patterns

- [Retry Pattern](retry.md): Often used with Circuit Breaker for transient failures
- [Fallback Pattern](fallback.md): Provides alternative functionality when circuit is open
- [Bulkhead Pattern](bulkhead.md): Isolates failures to prevent system-wide impact
