# Feature Store Integration

## Overview

The feature store integration centralizes the calculation and storage of technical indicators and features used by various services in the forex trading platform. This improves maintainability, reduces duplication, and enhances performance through caching and optimization.

## Components

The feature store integration consists of the following components:

1. **Feature Store Client**: A client library for accessing the feature store from other services
2. **Feature Cache**: A caching layer to reduce load on the feature store and improve performance
3. **Feature Store Metrics**: A metrics collector for monitoring feature store usage
4. **Feature Store Dashboard**: A web dashboard for visualizing feature store metrics

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

## Feature Store Metrics

The feature store metrics collector tracks usage and performance of the feature store client. It includes:

- API call metrics
- Cache hit/miss metrics
- Error metrics
- Performance metrics
- Fallback metrics

### Example Usage

```python
from strategy_execution_engine.monitoring.feature_store_metrics import feature_store_metrics

# Get metrics
metrics = feature_store_metrics.get_metrics()

# Log metrics summary
feature_store_metrics.log_metrics_summary()

# Export metrics as JSON
json_metrics = feature_store_metrics.export_metrics_json()
```

## Feature Store Dashboard

The feature store dashboard provides a web interface for visualizing feature store metrics. It includes:

- API call charts
- Cache performance charts
- Error charts
- Response time charts
- Fallback charts
- Summary statistics

### Running the Dashboard

```bash
python run_feature_store_dashboard.py --port 8050 --debug
```

## Integration with Strategies

Strategies can use the feature store client to retrieve indicators and features instead of calculating them directly. This improves performance and ensures consistent results across all services.

### Example Strategy Integration

```python
from strategy_execution_engine.clients.feature_store_client import FeatureStoreClient

class MyStrategy(BaseStrategy):
    def __init__(self, name, parameters=None):
        super().__init__(name, parameters)
        
        # Initialize feature store client
        self.feature_store_client = FeatureStoreClient(use_cache=True)
    
    async def generate_signals(self, data):
        # Get indicators from feature store
        indicators = await self.feature_store_client.get_indicators(
            symbol=data.get('symbol', 'unknown'),
            start_date=data.index[0],
            end_date=data.index[-1],
            indicators=["sma_50", "sma_200", "rsi_14"]
        )
        
        # Use indicators to generate signals
        # ...
        
        return signals
    
    async def cleanup(self):
        # Close feature store client
        await self.feature_store_client.close()
```

## Best Practices

1. **Use Caching**: Always enable caching to reduce load on the feature store
2. **Add Fallbacks**: Include fallback mechanisms for when the feature store is unavailable
3. **Handle Errors**: Properly handle errors from the feature store client
4. **Clean Up**: Always close the feature store client when it's no longer needed
5. **Batch Requests**: Batch indicator requests when possible to reduce API calls
6. **Use Async**: Use async/await for feature store client calls to improve performance
7. **Monitor Usage**: Use the metrics collector and dashboard to monitor usage and performance

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
