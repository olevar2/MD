# Service Clients

This module provides standardized client implementations for communicating with other services in the Forex Trading Platform.

## Available Clients

- **AnalysisEngineClient**: Client for the Analysis Engine Service
- **MLWorkbenchClient**: Client for the ML Workbench Service

## Usage

### Getting a Client

```python
from ml_integration_service.clients.client_factory import (
    get_analysis_engine_client,
    get_ml_workbench_client
)

# Get clients
analysis_engine_client = get_analysis_engine_client()
ml_workbench_client = get_ml_workbench_client()
```

### Making Requests

```python
# Example: Get technical indicators from Analysis Engine
indicators = [
    {"name": "SMA", "params": {"period": 20}},
    {"name": "RSI", "params": {"period": 14}}
]

result = await analysis_engine_client.get_technical_indicators(
    symbol="EURUSD",
    timeframe="1h",
    indicators=indicators
)

# Example: Get models from ML Workbench
models = await ml_workbench_client.get_models(model_type="classification")
```

### Error Handling

```python
from common_lib.clients.exceptions import ClientError, ClientTimeoutError

try:
    result = await analysis_engine_client.get_technical_indicators(...)
except ClientTimeoutError as e:
    # Handle timeout
    logger.error(f"Request timed out: {str(e)}")
except ClientError as e:
    # Handle other client errors
    logger.error(f"Client error: {str(e)}")
```

## Client Initialization

Clients are initialized during service startup in the `main.py` file:

```python
from ml_integration_service.clients.client_factory import initialize_clients

# Initialize clients
initialize_clients()
```

## Client Configuration

Client configurations are defined in the `client_factory.py` file and use settings from the `settings.py` file:

```python
# Configure Analysis Engine client
analysis_engine_config = {
    "base_url": settings.ANALYSIS_ENGINE_API_URL,
    "service_name": "analysis-engine-service",
    "api_key": settings.ANALYSIS_ENGINE_API_KEY.get_secret_value() if settings.ANALYSIS_ENGINE_API_KEY else None,
    "timeout_seconds": 30.0,
    # ... other settings
}
```

## Examples

See the `examples/client_usage_example.py` file for complete examples of using the clients.
