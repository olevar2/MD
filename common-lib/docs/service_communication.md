# Service Communication Guide

This guide explains the standardized approach to service communication in the Forex Trading Platform.

## Overview

The platform uses a standardized client implementation for service-to-service communication, providing:

1. **Resilience Patterns**: Circuit breakers, retry policies, timeouts, and bulkheads
2. **Consistent Error Handling**: Standardized exceptions and error mapping
3. **Metrics Collection**: Performance and error metrics for monitoring
4. **Structured Logging**: Detailed logging of requests and responses

## Client Architecture

### Base Client

The `BaseServiceClient` class in `common_lib.clients` provides a foundation for all service clients with:

- Standard HTTP methods (GET, POST, PUT, DELETE)
- Built-in resilience patterns
- Error handling and mapping
- Metrics collection
- Structured logging

### Client Factory

The client factory pattern centralizes client creation and configuration:

- `create_client()`: Creates a new client instance
- `get_client()`: Gets or creates a singleton client
- `register_client_config()`: Registers default configurations

### Client Configuration

Client configurations are centralized in `common_lib.clients.config`:

- Standard configurations for all services
- Environment variable overrides
- Service-specific settings

## Using Service Clients

### Creating a Service Client

```python
from common_lib.clients import get_client
from my_service.clients.other_service_client import OtherServiceClient

# Get a client with default configuration
client = get_client(
    client_class=OtherServiceClient,
    service_name="other-service"
)

# Get a client with custom configuration
client = get_client(
    client_class=OtherServiceClient,
    service_name="other-service",
    config_override={
        "timeout_seconds": 60.0,
        "max_retries": 5
    }
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

1. Create a new client class that inherits from `BaseServiceClient`
2. Implement service-specific methods using the base HTTP methods
3. Add proper error handling and documentation

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

### Creating a Client Factory

Create a factory module to simplify client creation:

```python
from common_lib.clients import get_client
from my_service.clients.other_service_client import OtherServiceClient

def get_other_service_client(config_override=None):
    """Get a configured Other Service client."""
    return get_client(
        client_class=OtherServiceClient,
        service_name="other-service",
        config_override=config_override
    )
```

### Initializing Clients

Initialize clients during service startup:

```python
def initialize_clients():
    """Initialize all service clients."""
    from common_lib.clients import register_client_config, ClientConfig
    
    # Register client configurations
    register_client_config(
        "other-service",
        ClientConfig(
            base_url="http://other-service:8000/api/v1",
            service_name="other-service",
            timeout_seconds=30.0
        )
    )
```

## Best Practices

1. **Use the Base Client**: Always inherit from `BaseServiceClient` for new clients
2. **Centralize Configuration**: Use the client factory and configuration module
3. **Handle Errors Properly**: Map exceptions to domain-specific errors
4. **Add Proper Logging**: Use structured logging for requests and responses
5. **Include Metrics**: Enable metrics collection for monitoring
6. **Close Clients**: Close clients during service shutdown
7. **Use Dependency Injection**: Provide clients to components through DI
8. **Test with Mocks**: Use mocks for testing client interactions
