# Trading Gateway Client Migration Guide

This guide provides instructions for migrating from the legacy `TradingGatewayClient` to the new standardized `StandardizedTradingGatewayClient`.

## Overview

The new standardized client offers several advantages:

1. **Built-in Resilience Patterns**: Circuit breaker, retry, timeout, and bulkhead patterns are built into the client.
2. **Standardized Error Handling**: Consistent error handling across all services.
3. **Metrics Collection**: Automatic collection of metrics for monitoring.
4. **Structured Logging**: Detailed logging with correlation IDs for tracing.
5. **Configurable Behavior**: Easily configurable client behavior.

## Migration Steps

### Step 1: Update Imports

Replace imports of the legacy client with imports of the new client factory:

```python
# Old import
from strategy_execution_engine.clients.trading_gateway_client import TradingGatewayClient

# New import
from strategy_execution_engine.clients.client_factory import get_trading_gateway_client
```

### Step 2: Replace Client Instantiation

Replace direct instantiation of the legacy client with the factory function:

```python
# Old instantiation
client = TradingGatewayClient(base_url="http://trading-gateway-service:8000", api_key="my-api-key")

# New instantiation
client = get_trading_gateway_client()
```

If you need custom configuration:

```python
from strategy_execution_engine.clients.client_factory import get_trading_gateway_client_with_config

client = get_trading_gateway_client_with_config({
    "base_url": "http://trading-gateway-service:8000",
    "api_key": "my-api-key",
    "timeout_seconds": 10.0
})
```

### Step 3: Update Method Calls

Update method calls to match the new client's methods:

| Legacy Method | Standardized Method |
|---------------|---------------------|
| `execute_order` | `execute_order` |
| `get_order_status` | `get_order_status` |
| `get_account_info` | `get_account_info` |
| `get_positions` | `get_positions` |
| `get_market_data` | `get_market_data` |
| `check_health` | `check_health` |

Most method signatures remain the same, but there are some differences:

- `get_market_data` now accepts `timeframe` and `count` parameters.

### Step 4: Update Error Handling

The new client uses standardized exceptions from `common_lib.clients.exceptions`:

```python
from common_lib.clients.exceptions import (
    ClientError,
    ClientConnectionError,
    ClientTimeoutError,
    ClientValidationError,
    ClientAuthenticationError
)

try:
    result = await client.execute_order(order)
except ClientTimeoutError as e:
    # Handle timeout specifically
    logger.error(f"Request timed out: {str(e)}")
    # Implement fallback or retry logic
except ClientConnectionError as e:
    # Handle connection issues
    logger.error(f"Connection error: {str(e)}")
    # Implement fallback or retry logic
except ClientError as e:
    # Handle other client errors
    logger.error(f"Client error: {str(e)}")
    # Implement general error handling
```

The client also maps certain errors to domain-specific exceptions:

```python
from common_lib.error import DataFetchError

try:
    result = await client.get_order_status("non-existent-order-id")
except DataFetchError as e:
    # Handle data fetch errors
    logger.error(f"Data fetch error: {str(e)}")
    # Implement appropriate handling
```

### Step 5: Update Configuration

The new client uses `ClientConfig` for configuration. Update your configuration settings in `settings.py`:

```python
# Add these settings to your configuration
client_timeout_seconds: float = 5.0
client_max_retries: int = 3
client_retry_base_delay: float = 0.1
client_retry_backoff_factor: float = 2.0
client_circuit_breaker_threshold: int = 5
client_circuit_breaker_reset_timeout: int = 30
```

## Example Migration

### Before

```python
from strategy_execution_engine.clients.trading_gateway_client import TradingGatewayClient

async def execute_trade(symbol, side, quantity):
    client = TradingGatewayClient()
    
    try:
        order = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "order_type": "market"
        }
        result = await client.execute_order(order)
        return result
    except Exception as e:
        logger.error(f"Error executing trade: {str(e)}")
        raise
    finally:
        await client.close()
```

### After

```python
from strategy_execution_engine.clients.client_factory import get_trading_gateway_client
from common_lib.clients.exceptions import ClientError, ClientTimeoutError

async def execute_trade(symbol, side, quantity):
    client = get_trading_gateway_client()
    
    try:
        order = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "order_type": "market"
        }
        result = await client.execute_order(order)
        return result
    except ClientTimeoutError as e:
        logger.error(f"Trade execution timed out: {str(e)}")
        # Implement fallback or retry logic
        raise
    except ClientError as e:
        logger.error(f"Error executing trade: {str(e)}")
        raise
```

## Backward Compatibility

For backward compatibility, the legacy `TradingGatewayClient` will continue to work, but it is deprecated and will be removed in a future release. We recommend migrating to the new standardized client as soon as possible.

## Testing

The new client includes comprehensive unit tests. You can run them with:

```bash
pytest strategy-execution-engine/tests/unit/clients/test_standardized_trading_gateway_client.py
```
