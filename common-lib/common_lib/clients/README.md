# Service Clients Module

This module provides standardized client implementations for service communication in the Forex Trading Platform.

## Key Components

1. **BaseServiceClient**: Base class for all service clients with built-in resilience patterns
2. **ClientFactory**: Factory functions for creating and managing client instances
3. **ClientConfig**: Configuration model for client settings
4. **Exceptions**: Standardized exceptions for client errors

## Usage

### Creating a Client

```python
from common_lib.clients import get_client
from my_service.clients.other_service_client import OtherServiceClient

# Get a client with default configuration
client = get_client(
    client_class=OtherServiceClient,
    service_name="other-service"
)
```

### Making Requests

```python
# Async request
result = await client.get("endpoint/path")
result = await client.post("endpoint/path", data={"key": "value"})

# Sync request (for services that don't use async)
result = client.sync_get("endpoint/path")
result = client.sync_post("endpoint/path", data={"key": "value"})
```

### Error Handling

```python
from common_lib.clients.exceptions import ClientError, ClientTimeoutError

try:
    result = await client.get("endpoint/path")
except ClientTimeoutError as e:
    # Handle timeout
    logger.error(f"Request timed out: {str(e)}")
except ClientError as e:
    # Handle other client errors
    logger.error(f"Client error: {str(e)}")
```

## Implementing a Service Client

### Creating a New Client

```python
from typing import Dict, Any, Optional, Union
from common_lib.clients import BaseServiceClient, ClientConfig

class MyServiceClient(BaseServiceClient):
    """Client for interacting with My Service."""
    
    def __init__(self, config: Union[ClientConfig, Dict[str, Any]]):
        """Initialize the client."""
        super().__init__(config)
    
    async def get_resource(self, resource_id: str) -> Dict[str, Any]:
        """Get a resource by ID."""
        try:
            return await self.get(f"resources/{resource_id}")
        except Exception as e:
            # Map to a more specific exception if needed
            raise ClientError(f"Failed to get resource {resource_id}", self.config.service_name) from e
```

## Configuration

Client configurations are centralized in `common_lib.clients.config`:

```python
from common_lib.clients import register_client_config, ClientConfig

# Register a client configuration
register_client_config(
    "other-service",
    ClientConfig(
        base_url="http://other-service:8000/api/v1",
        service_name="other-service",
        timeout_seconds=30.0
    )
)
```

## Resilience Patterns

The base client includes the following resilience patterns:

1. **Circuit Breaker**: Prevents cascading failures by stopping calls to failing services
2. **Retry Policy**: Automatically retries temporary failures with exponential backoff
3. **Timeout Handler**: Ensures operations complete within specific time constraints
4. **Bulkhead Pattern**: Isolates failures by partitioning resources

These patterns are configured through the `ClientConfig` model.
