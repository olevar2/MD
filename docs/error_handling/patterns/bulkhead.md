# Bulkhead Pattern

## Overview

The Bulkhead pattern isolates failures by partitioning resources. Named after the compartmentalized sections of a ship's hull, this pattern ensures that failures in one part of the system don't cascade to others by limiting resource usage.

## When to Use

Use the Bulkhead pattern when:

- You need to isolate critical operations from non-critical ones
- You want to prevent one operation from consuming all resources
- You need to maintain system stability under heavy load
- Different operations have different resource requirements or priorities

## Implementation in Forex Trading Platform

The Forex Trading Platform provides a `bulkhead` decorator in the `common_lib.resilience` module:

```python
from common_lib.resilience import bulkhead
from common_lib.exceptions import BulkheadFullError, ServiceUnavailableError

# Create bulkheads for different operation types
critical_operations = bulkhead(
    name="critical-operations",
    max_concurrent=20,     # Maximum concurrent executions
    max_queue_size=10      # Maximum queue size for waiting operations
)

non_critical_operations = bulkhead(
    name="non-critical-operations",
    max_concurrent=10,
    max_queue_size=20
)

@critical_operations
async def execute_trade(trade_request):
    """
    Execute a trade with resource protection via bulkhead.
    
    Args:
        trade_request: Validated trade request
        
    Returns:
        Trade execution result
        
    Raises:
        ServiceUnavailableError: If system is under heavy load
    """
    try:
        # This operation is protected by the critical_operations bulkhead
        return await trading_gateway.execute_trade(trade_request)
    except BulkheadFullError:
        # Bulkhead is full, system is under heavy load
        logger.warning(
            "Critical operations bulkhead full, rejecting trade request",
            extra={"bulkhead": "critical-operations"}
        )
        raise ServiceUnavailableError(
            message="System is currently under heavy load, please try again later",
            details={"retry_after_seconds": 30}
        )
```

## Bulkhead Types

The platform supports two types of bulkheads:

1. **Thread Pool Bulkhead**: Limits the number of concurrent executions
2. **Semaphore Bulkhead**: Limits the number of concurrent executions without a queue

## Configuration Parameters

| Parameter | Description | Recommended Value |
|-----------|-------------|------------------|
| `name` | Name of the bulkhead for monitoring | Descriptive name of operation type |
| `max_concurrent` | Maximum concurrent executions | Based on resource capacity |
| `max_queue_size` | Maximum queue size for waiting operations | Based on acceptable wait time |
| `queue_timeout_seconds` | Maximum time to wait in queue | 1-5 seconds for user-facing operations |

## Best Practices

1. **Separate Critical Operations**: Use different bulkheads for operations with different priorities
2. **Size Appropriately**: Configure bulkhead sizes based on available resources
3. **Monitor Rejection Rate**: Track how often operations are rejected by bulkheads
4. **Provide Fallbacks**: Implement fallback functionality when bulkheads reject operations
5. **Combine with Circuit Breaker**: Use circuit breaker to prevent calling failing services
6. **Set Appropriate Timeouts**: Configure queue timeouts based on operation criticality

## Example: Resource Isolation

```python
# Create bulkheads for different resource types
database_operations = bulkhead(
    name="database-operations",
    max_concurrent=30,
    max_queue_size=20
)

external_api_operations = bulkhead(
    name="external-api-operations",
    max_concurrent=10,
    max_queue_size=5
)

analytics_operations = bulkhead(
    name="analytics-operations",
    max_concurrent=5,
    max_queue_size=10
)

# Use bulkheads to isolate different operation types
@database_operations
async def save_portfolio(portfolio):
    return await repository.save(portfolio)

@external_api_operations
async def fetch_market_data(symbol):
    return await market_data_client.get_price(symbol)

@analytics_operations
async def analyze_portfolio(portfolio_id):
    return await analytics_service.analyze_portfolio(portfolio_id)
```

## Monitoring Bulkheads

Monitor these metrics for each bulkhead:

- **Concurrent Executions**: Current number of concurrent executions
- **Queue Size**: Current number of operations waiting in queue
- **Rejection Rate**: Rate at which operations are rejected
- **Execution Time**: Time taken to execute operations
- **Queue Wait Time**: Time operations spend waiting in queue

## Related Patterns

- [Circuit Breaker Pattern](circuit_breaker.md): Prevents calling failing services
- [Timeout Pattern](timeout.md): Ensures operations complete within a time limit
- [Fallback Pattern](fallback.md): Provides alternative functionality when bulkheads reject operations
