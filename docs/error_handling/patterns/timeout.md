# Timeout Pattern

## Overview

The Timeout pattern ensures operations complete within specific time constraints. It prevents operations from hanging indefinitely, which could block resources and degrade system performance.

## When to Use

Use the Timeout pattern when:

- Operations might hang indefinitely
- You need to maintain responsive user interfaces
- You have SLAs for operation completion time
- You want to fail fast rather than wait for slow operations
- You need to prevent resource exhaustion from long-running operations

## Implementation in Forex Trading Platform

The Forex Trading Platform provides a `timeout_handler` decorator in the `common_lib.resilience` module:

```python
from common_lib.resilience import timeout_handler
from common_lib.exceptions import TimeoutError

@timeout_handler(timeout_seconds=5.0)
async def get_real_time_price(symbol):
    """
    Get real-time price with timeout protection.
    
    Args:
        symbol: Trading symbol to get price for
        
    Returns:
        Real-time price data
        
    Raises:
        TimeoutError: If operation exceeds timeout
    """
    try:
        return await market_data_client.get_real_time_price(symbol)
    except TimeoutError:
        logger.warning(
            f"Timeout getting real-time price for {symbol}",
            extra={"symbol": symbol, "timeout_seconds": 5.0}
        )
        raise TimeoutError(
            message=f"Operation timed out: get_real_time_price({symbol})",
            details={"symbol": symbol, "timeout_seconds": 5.0}
        )
```

## Timeout Types

The platform supports several types of timeouts:

1. **Operation Timeout**: Limits the time for a single operation
2. **Request Timeout**: Limits the time for an HTTP request
3. **Connection Timeout**: Limits the time to establish a connection
4. **Read Timeout**: Limits the time to read data from a connection

## Configuration Parameters

| Parameter | Description | Recommended Value |
|-----------|-------------|------------------|
| `timeout_seconds` | Maximum time for operation completion | Based on operation type |
| `cancel_on_timeout` | Whether to cancel the operation on timeout | `True` for most cases |
| `propagate` | Whether to propagate the timeout exception | `True` for most cases |

## Recommended Timeout Values

| Operation Type | Recommended Timeout | Rationale |
|----------------|---------------------|-----------|
| User-facing API | 1-3 seconds | Maintain responsive UI |
| Internal API | 5-10 seconds | Allow for processing time |
| Database operations | 5-15 seconds | Allow for complex queries |
| External API calls | 10-30 seconds | Account for network latency |
| Background jobs | 60+ seconds | Allow for long-running tasks |

## Best Practices

1. **Set Appropriate Timeouts**: Configure timeouts based on operation type and criticality
2. **Provide Fallbacks**: Implement fallback functionality when operations timeout
3. **Log Timeouts**: Log all timeouts for monitoring and debugging
4. **Monitor Timeout Frequency**: Track how often operations timeout
5. **Combine with Circuit Breaker**: Use circuit breaker to prevent calling services with frequent timeouts
6. **Consider Resource Cleanup**: Ensure resources are properly released when operations timeout

## Example: Tiered Timeouts

```python
from common_lib.resilience import timeout_handler
from common_lib.exceptions import TimeoutError

# Different timeouts for different operation types
@timeout_handler(timeout_seconds=2.0)
async def get_cached_price(symbol):
    """Get price from cache (fast operation)."""
    return await cache.get(f"price:{symbol}")

@timeout_handler(timeout_seconds=5.0)
async def get_database_price(symbol):
    """Get price from database (medium operation)."""
    return await db.query(f"SELECT price FROM prices WHERE symbol = '{symbol}'")

@timeout_handler(timeout_seconds=10.0)
async def get_external_price(symbol):
    """Get price from external API (slow operation)."""
    return await external_api.get_price(symbol)

async def get_price_with_fallbacks(symbol):
    """
    Get price with fallbacks if operations timeout.
    
    Args:
        symbol: Trading symbol to get price for
        
    Returns:
        Price data from fastest available source
        
    Raises:
        ServiceUnavailableError: If all sources timeout
    """
    try:
        # Try cache first (fastest)
        return await get_cached_price(symbol)
    except TimeoutError:
        logger.warning(f"Cache timeout for {symbol}, trying database")
        
        try:
            # Try database next
            price = await get_database_price(symbol)
            
            # Update cache in background
            asyncio.create_task(cache.set(f"price:{symbol}", price))
            
            return price
        except TimeoutError:
            logger.warning(f"Database timeout for {symbol}, trying external API")
            
            try:
                # Try external API as last resort
                price = await get_external_price(symbol)
                
                # Update cache and database in background
                asyncio.create_task(cache.set(f"price:{symbol}", price))
                asyncio.create_task(db.execute(
                    f"UPDATE prices SET price = {price} WHERE symbol = '{symbol}'"
                ))
                
                return price
            except TimeoutError:
                logger.error(f"All price sources timed out for {symbol}")
                raise ServiceUnavailableError(
                    message=f"Unable to retrieve price for {symbol}",
                    details={"symbol": symbol}
                )
```

## Related Patterns

- [Circuit Breaker Pattern](circuit_breaker.md): Prevents calling services with frequent timeouts
- [Retry Pattern](retry.md): Retries operations that timeout
- [Fallback Pattern](fallback.md): Provides alternative functionality when operations timeout
- [Bulkhead Pattern](bulkhead.md): Isolates slow operations to prevent system-wide impact
