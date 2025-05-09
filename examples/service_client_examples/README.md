# Standardized Service Client Examples

This directory contains examples of how to implement and use standardized service clients in the Forex Trading Platform.

## Overview

The examples demonstrate:

1. How to implement a service client using the standardized template
2. How to create a client factory for managing client instances
3. How to use the clients with proper error handling
4. How to propagate correlation IDs across service calls

## Files

### Python Examples

- `market_data_client.py`: Example implementation of a service client
- `client_factory.py`: Example implementation of a client factory
- `usage_example.py`: Example of how to use the service clients

### JavaScript/TypeScript Examples

- `MarketDataClient.ts`: Example implementation of a service client
- `clientFactory.ts`: Example implementation of a client factory
- `usageExample.ts`: Example of how to use the service clients

## Key Features

The standardized service clients provide:

1. **Consistent Interface**: All service clients follow the same pattern
2. **Built-in Resilience**: Retry, circuit breaker, timeout, and bulkhead patterns
3. **Standardized Error Handling**: Consistent error types and handling
4. **Correlation ID Propagation**: Easy tracking of requests across services
5. **Metrics Collection**: Performance tracking for all service calls
6. **Structured Logging**: Consistent logging format with context

## Usage

### Python

```python
from examples.service_client_examples.client_factory import (
    initialize_clients,
    get_market_data_client
)
from common_lib.clients.exceptions import ClientError

# Initialize clients
initialize_clients()

# Get a client
client = get_market_data_client()

# Use the client
try:
    result = await client.get_ohlcv_data(
        symbol="EUR/USD",
        timeframe="1h",
        limit=100
    )
    print(f"Got {len(result.get('data', []))} OHLCV data points")
except ClientError as e:
    print(f"Client error: {str(e)}")
```

### JavaScript/TypeScript

```typescript
import { 
  initializeClients, 
  getMarketDataClient 
} from './clientFactory';
import { ClientError } from '../../common-js-lib/index';

// Initialize clients
initializeClients();

// Get a client
const client = getMarketDataClient();

// Use the client
try {
  const result = await client.getOHLCVData(
    'EUR/USD',
    '1h',
    undefined,
    undefined,
    100
  );
  console.log(`Got ${result.data.length} OHLCV data points`);
} catch (error) {
  if (error instanceof ClientError) {
    console.error(`Client error: ${error.message}`);
  }
}
```

## Implementing a New Service Client

### Python

1. Import the standardized template:

```python
from common_lib.clients.templates.service_client_template import StandardServiceClient
```

2. Create a new client class:

```python
class MyServiceClient(StandardServiceClient):
    """Client for interacting with My Service."""
    
    def __init__(self, config: Union[ClientConfig, Dict[str, Any]]):
        """Initialize the client."""
        super().__init__(config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def get_my_resource(self, resource_id: str) -> Dict[str, Any]:
        """Get a resource by ID."""
        self.logger.debug(f"Getting resource {resource_id}")
        try:
            return await self.get(f"my-resources/{resource_id}")
        except Exception as e:
            self.logger.error(f"Failed to get resource {resource_id}: {str(e)}")
            raise
```

3. Add a factory function:

```python
def get_my_service_client(config_override: Optional[Dict[str, Any]] = None) -> MyServiceClient:
    """Get a configured My Service client."""
    return get_client(
        client_class=MyServiceClient,
        service_name="my-service",
        config_override=config_override
    )
```

### JavaScript/TypeScript

1. Import the standardized template:

```typescript
import { StandardServiceClient } from '../../common-js-lib/templates/ServiceClientTemplate';
```

2. Create a new client class:

```typescript
export class MyServiceClient extends StandardServiceClient<any> {
  constructor(config: ClientConfig) {
    super(config);
  }
  
  async getMyResource(resourceId: string): Promise<any> {
    this.logger.debug(`Getting resource ${resourceId}`);
    try {
      return await this.get(`my-resources/${resourceId}`);
    } catch (error) {
      this.logger.error(`Failed to get resource ${resourceId}: ${error}`);
      throw error;
    }
  }
}
```

3. Add a factory function:

```typescript
export function getMyServiceClient(configOverride?: Partial<ClientConfig>): MyServiceClient {
  return getClient(
    MyServiceClient,
    'my-service',
    configOverride
  );
}
```

## Best Practices

1. **Use the Template**: Always extend the standardized template for new clients
2. **Consistent Method Names**: Follow the naming conventions for methods
3. **Proper Error Handling**: Log errors and provide meaningful error messages
4. **Documentation**: Document all methods with proper docstrings/JSDoc
5. **Type Safety**: Use proper type annotations for parameters and return values
6. **Correlation ID**: Use the correlation ID propagation methods for tracing
7. **Factory Functions**: Create factory functions for all clients
8. **Configuration**: Use the client factory for configuration management