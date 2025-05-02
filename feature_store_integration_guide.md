# Feature Store Integration Guide

## Overview

This guide provides comprehensive information about the feature store integration in the forex trading platform. The feature store centralizes the calculation and storage of technical indicators and other features used by various services in the platform.

## Architecture

The feature store integration consists of the following components:

1. **Feature Store Service**: Central service for calculating and storing technical indicators and features
2. **Feature Store Client**: Client library for accessing the feature store from other services
3. **Feature Cache**: Caching layer to reduce load on the feature store and improve performance
4. **Indicator Registry**: Registry of available indicators in the feature store

## Feature Store Client

The feature store client provides a unified interface for accessing the feature store from other services. It includes:

- Methods for retrieving OHLCV data
- Methods for retrieving technical indicators
- Methods for computing custom features
- Caching to reduce load on the feature store
- Resilience features (circuit breaker, retry with backoff)
- Fallback mechanisms for when the feature store is unavailable

### Example Usage

```python
from strategy_execution_engine.clients.feature_store_client import FeatureStoreClient

# Initialize client
client = FeatureStoreClient(use_cache=True)

# Get indicators
indicators = await client.get_indicators(
    symbol="EUR/USD",
    start_date="2023-01-01",
    end_date="2023-01-31",
    timeframe="1h",
    indicators=["sma_50", "sma_200", "rsi_14"]
)

# Compute a custom feature
volatility = await client.compute_feature(
    feature_name="volatility",
    symbol="EUR/USD",
    start_date="2023-01-01",
    end_date="2023-01-31",
    timeframe="1h",
    parameters={"window": 14}
)

# Clean up
await client.close()
```

## Feature Cache

The feature cache provides caching for feature store results to reduce load on the feature store and improve performance. It includes:

- In-memory caching with TTL (time-to-live)
- Optional Redis-based distributed caching
- Cache invalidation mechanisms
- Cache statistics

### Example Usage

```python
from strategy_execution_engine.caching.feature_cache import FeatureCache

# Initialize cache
cache = FeatureCache(max_size=1000, default_ttl=300)

# Get from cache
cached_data = cache.get("my_key")

# Set in cache
cache.set("my_key", data, ttl=600)

# Invalidate cache entry
cache.invalidate("my_key")

# Get cache statistics
stats = cache.get_stats()
```

## Migrating from Direct Calculation to Feature Store

When migrating from direct calculation to using the feature store, follow these steps:

1. Identify indicators that are calculated directly
2. Replace direct calculations with feature store client calls
3. Add fallback mechanisms for when the feature store is unavailable
4. Update method signatures to be async if necessary
5. Add cleanup code to close the feature store client

### Example: Before Migration

```python
def calculate_trend(self, price_data):
    # Direct calculation
    ma_50 = self.technical_indicators.calculate_ma(price_data, period=50).iloc[-1]
    ma_200 = self.technical_indicators.calculate_ma(price_data, period=200).iloc[-1]
    
    if ma_50 > ma_200:
        return "bullish"
    else:
        return "bearish"
```

### Example: After Migration

```python
async def calculate_trend(self, price_data):
    # Get symbol and timeframe
    symbol = price_data.get('symbol', 'unknown')
    
    # Get indicators from feature store
    indicators = await self.feature_store_client.get_indicators(
        symbol=symbol,
        start_date=price_data.index[0],
        end_date=price_data.index[-1],
        indicators=["sma_50", "sma_200"]
    )
    
    # Use indicators from feature store
    if not indicators.empty:
        ma_50 = indicators["sma_50"].iloc[-1]
        ma_200 = indicators["sma_200"].iloc[-1]
    else:
        # Fallback to direct calculation
        ma_50 = self.technical_indicators.calculate_ma(price_data, period=50).iloc[-1]
        ma_200 = self.technical_indicators.calculate_ma(price_data, period=200).iloc[-1]
    
    if ma_50 > ma_200:
        return "bullish"
    else:
        return "bearish"
```

## Best Practices

1. **Use Caching**: Always enable caching to reduce load on the feature store
2. **Add Fallbacks**: Include fallback mechanisms for when the feature store is unavailable
3. **Handle Errors**: Properly handle errors from the feature store client
4. **Clean Up**: Always close the feature store client when it's no longer needed
5. **Batch Requests**: Batch indicator requests when possible to reduce API calls
6. **Use Async**: Use async/await for feature store client calls to improve performance

## Available Indicators

The following indicators are available in the feature store:

| Indicator | ID | Parameters |
|-----------|-------|------------|
| Simple Moving Average | sma_X | X = period |
| Exponential Moving Average | ema_X | X = period |
| Relative Strength Index | rsi_X | X = period |
| Average True Range | atr_X | X = period |
| Bollinger Bands | bb_X | X = period |
| MACD | macd | fast_period, slow_period, signal_period |
| Stochastic Oscillator | stoch | k_period, d_period, slowing |

## Troubleshooting

### Common Issues

1. **Connection Errors**: Check that the feature store service is running and accessible
2. **Authentication Errors**: Check that the API key is correct
3. **Missing Indicators**: Check that the requested indicators are available in the feature store
4. **Performance Issues**: Check cache settings and consider increasing cache size or TTL

### Logging

The feature store client includes comprehensive logging. To enable debug logging:

```python
import logging
logging.getLogger("feature_store_client").setLevel(logging.DEBUG)
```

## Future Enhancements

1. **Real-time Indicators**: Support for real-time indicator updates
2. **Custom Indicator Registration**: API for registering custom indicators
3. **Feature Store Admin UI**: Web UI for managing the feature store
4. **Enhanced Monitoring**: More detailed monitoring and alerting
5. **ML Feature Integration**: Tighter integration with ML features

## Contact

For questions or issues related to the feature store integration, contact the platform team.
