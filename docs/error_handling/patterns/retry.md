# Retry Pattern

## Overview

The Retry pattern enables an application to handle transient failures by automatically retrying failed operations. This pattern is particularly useful for operations that might fail due to temporary conditions such as network connectivity issues, timeouts, or service unavailability.

## When to Use

Use the Retry pattern when:

- Dealing with transient failures that might resolve on retry
- Working with network operations that might experience temporary issues
- Calling external services that might be temporarily unavailable
- Handling database operations that might experience deadlocks or connection issues

**Important**: Only use retry for operations that are idempotent (can be safely repeated without side effects).

## Implementation in Forex Trading Platform

The Forex Trading Platform provides retry functionality through the `retry_with_policy` decorator in the `common_lib.resilience` module:

```python
from common_lib.resilience import retry_with_policy
from common_lib.exceptions import NetworkError, RetryExhaustedError

@retry_with_policy(
    max_attempts=3,           # Maximum number of attempts
    base_delay=1.0,           # Initial delay between retries (seconds)
    max_delay=10.0,           # Maximum delay between retries (seconds)
    jitter=True,              # Add randomness to delay to prevent thundering herd
    exceptions=[ConnectionError, TimeoutError]  # Exceptions to retry on
)
async def fetch_market_data(symbol: str) -> Dict:
    """
    Fetch market data with retry logic for network errors.
    
    Args:
        symbol: The trading symbol to fetch data for
        
    Returns:
        Dictionary with market data
        
    Raises:
        NetworkError: If network issues persist after retries
        RetryExhaustedError: If all retry attempts fail
    """
    try:
        return await market_data_client.get_price(symbol)
    except (ConnectionError, TimeoutError) as e:
        # These will be caught by the retry decorator and retried
        logger.warning(
            f"Network error fetching market data for {symbol}, retrying...",
            extra={"symbol": symbol, "error": str(e)}
        )
        raise
    except Exception as e:
        # Other exceptions won't be retried
        logger.error(
            f"Error fetching market data for {symbol}",
            extra={"symbol": symbol, "error": str(e)}
        )
        raise NetworkError(
            message=f"Failed to fetch market data for {symbol}",
            details={"symbol": symbol, "original_error": str(e)}
        )
```

## Retry Strategies

The platform supports several retry strategies:

1. **Fixed Delay**: Retry with a constant delay between attempts
2. **Exponential Backoff**: Increase the delay exponentially between attempts
3. **Exponential Backoff with Jitter**: Add randomness to the delay to prevent thundering herd

## Configuration Parameters

| Parameter | Description | Recommended Value |
|-----------|-------------|------------------|
| `max_attempts` | Maximum number of retry attempts | 3-5 for most operations |
| `base_delay` | Initial delay between retries (seconds) | 0.5-2.0 seconds |
| `max_delay` | Maximum delay between retries (seconds) | 10-30 seconds |
| `jitter` | Whether to add randomness to delay | `True` for most cases |
| `exceptions` | List of exceptions to retry on | Only transient exceptions |
| `backoff_factor` | Multiplier for exponential backoff | 2.0 (doubles each retry) |

## Best Practices

1. **Retry Only Idempotent Operations**: Ensure operations can be safely repeated
2. **Limit Retry Attempts**: Don't retry indefinitely, set a reasonable maximum
3. **Use Exponential Backoff**: Increase delay between retries to avoid overwhelming services
4. **Add Jitter**: Randomize delay to prevent synchronized retries from multiple clients
5. **Log Retry Attempts**: Log each retry for monitoring and debugging
6. **Specify Exceptions**: Only retry on specific exceptions that represent transient failures
7. **Combine with Circuit Breaker**: Use circuit breaker to prevent retrying when a service is down

## Example: Database Operations

```python
from sqlalchemy.exc import OperationalError, IntegrityError
from common_lib.resilience import retry_with_policy
from common_lib.exceptions import DataStorageError

@retry_with_policy(
    max_attempts=3,
    base_delay=0.5,
    max_delay=5.0,
    jitter=True,
    exceptions=[OperationalError]  # Only retry operational errors
)
async def save_portfolio(portfolio):
    """
    Save a portfolio to the database with retry for transient errors.
    
    Args:
        portfolio: Portfolio object to save
        
    Returns:
        Saved portfolio with ID
        
    Raises:
        DataValidationError: If portfolio violates constraints
        DataStorageError: If database operation fails
    """
    try:
        return await repository.save(portfolio)
    except IntegrityError as e:
        # Don't retry constraint violations
        logger.error(
            f"Database constraint violation: {str(e)}",
            extra={"portfolio_id": portfolio.id}
        )
        raise DataValidationError(
            message="Portfolio data violates database constraints",
            details={"error": str(e)}
        )
    except OperationalError as e:
        # These will be retried
        if "deadlock" in str(e).lower():
            logger.warning(
                "Deadlock detected, retrying transaction",
                extra={"portfolio_id": portfolio.id}
            )
        elif "connection" in str(e).lower():
            logger.warning(
                "Database connection error, retrying",
                extra={"portfolio_id": portfolio.id}
            )
        raise
    except Exception as e:
        # Other exceptions won't be retried
        logger.error(
            f"Database error: {str(e)}",
            extra={"portfolio_id": portfolio.id}
        )
        raise DataStorageError(
            message="Failed to save portfolio",
            details={"portfolio_id": portfolio.id, "error": str(e)}
        )
```

## Related Patterns

- [Circuit Breaker Pattern](circuit_breaker.md): Prevents retrying when a service is down
- [Timeout Pattern](timeout.md): Ensures operations complete within a time limit
- [Fallback Pattern](fallback.md): Provides alternative functionality when retries fail
