# Service Client Implementation Guide

This guide provides detailed examples and best practices for implementing standardized service clients in the Forex Trading Platform. It covers both Python and JavaScript/TypeScript implementations.

## Table of Contents

1. [Overview](#overview)
2. [Python Client Implementation](#python-client-implementation)
3. [JavaScript/TypeScript Client Implementation](#javascripttypescript-client-implementation)
4. [Error Handling](#error-handling)
5. [Resilience Patterns](#resilience-patterns)
6. [Correlation ID Propagation](#correlation-id-propagation)
7. [Testing Service Clients](#testing-service-clients)

## Overview

Service clients in the Forex Trading Platform follow these principles:

1. **Standardized Interface**: Consistent method signatures and naming
2. **Built-in Resilience**: Circuit breakers, retries, timeouts, and bulkheads
3. **Comprehensive Error Handling**: Domain-specific exceptions and error mapping
4. **Correlation ID Propagation**: For tracing requests across services
5. **Metrics and Logging**: For monitoring and debugging

## Python Client Implementation

### Basic Client Structure

```python
from typing import Dict, Any, Optional, Union, List
import logging
from datetime import datetime

from common_lib.clients.base_client import BaseServiceClient, ClientConfig
from common_lib.clients.exceptions import (
    ClientError,
    ClientConnectionError,
    ClientTimeoutError,
    ClientValidationError,
    ClientAuthenticationError
)

class ExampleServiceClient(BaseServiceClient):
    """Client for interacting with Example Service."""
    
    def __init__(self, config: Union[ClientConfig, Dict[str, Any]]):
        """Initialize the client."""
        super().__init__(config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def get_resource(self, resource_id: str) -> Dict[str, Any]:
        """
        Get a resource by ID.
        
        Args:
            resource_id: ID of the resource to retrieve
            
        Returns:
            Resource data
            
        Raises:
            ClientError: If the request fails
            ClientTimeoutError: If the request times out
            ClientConnectionError: If connection to the service fails
        """
        self.logger.debug(f"Getting resource {resource_id}")
        try:
            return await self.get(f"resources/{resource_id}")
        except Exception as e:
            self.logger.error(f"Failed to get resource {resource_id}: {str(e)}")
            raise ClientError(f"Failed to get resource {resource_id}", 
                             self.config.service_name) from e
    
    async def create_resource(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new resource.
        
        Args:
            data: Resource data
            
        Returns:
            Created resource data
            
        Raises:
            ClientError: If the request fails
            ClientValidationError: If the data is invalid
            ClientTimeoutError: If the request times out
            ClientConnectionError: If connection to the service fails
        """
        self.logger.debug(f"Creating resource: {data}")
        try:
            return await self.post("resources", data=data)
        except Exception as e:
            self.logger.error(f"Failed to create resource: {str(e)}")
            raise ClientError(f"Failed to create resource", 
                             self.config.service_name) from e
```

### Client Factory

```python
from typing import Dict, Optional
from common_lib.clients import get_client, register_client_config, ClientConfig
from .example_service_client import ExampleServiceClient

# Client instances
_example_service_client: Optional[ExampleServiceClient] = None

def initialize_clients(config: Optional[Dict] = None):
    """
    Initialize all service clients.
    
    Args:
        config: Optional configuration overrides
    """
    config = config or {}
    
    # Register client configurations
    register_client_config(
        "example-service",
        ClientConfig(
            base_url=config.get("EXAMPLE_SERVICE_URL", "http://example-service:8000/api/v1"),
            service_name="example-service",
            timeout_seconds=config.get("EXAMPLE_SERVICE_TIMEOUT", 30.0),
            circuit_breaker_failure_threshold=config.get("EXAMPLE_SERVICE_CB_THRESHOLD", 5),
            circuit_breaker_reset_timeout_seconds=config.get("EXAMPLE_SERVICE_CB_RESET", 30.0),
            retry_max_attempts=config.get("EXAMPLE_SERVICE_RETRY_MAX", 3),
            retry_initial_delay_seconds=config.get("EXAMPLE_SERVICE_RETRY_DELAY", 0.1),
            retry_backoff_factor=config.get("EXAMPLE_SERVICE_RETRY_BACKOFF", 2.0)
        )
    )

def get_example_service_client() -> ExampleServiceClient:
    """
    Get the Example Service client.
    
    Returns:
        ExampleServiceClient instance
    """
    global _example_service_client
    
    if _example_service_client is None:
        _example_service_client = get_client(
            client_class=ExampleServiceClient,
            service_name="example-service"
        )
    
    return _example_service_client
```

### Usage Example

```python
from .clients import get_example_service_client
from common_lib.clients.exceptions import ClientError, ClientTimeoutError

async def get_resource_data(resource_id: str):
    """Get resource data from Example Service."""
    client = get_example_service_client()
    
    try:
        resource = await client.get_resource(resource_id)
        return resource
    except ClientTimeoutError as e:
        # Handle timeout specifically
        logger.error(f"Request timed out: {str(e)}")
        # Implement fallback or retry logic
        return None
    except ClientError as e:
        # Handle other client errors
        logger.error(f"Client error: {str(e)}")
        # Implement error handling logic
        return None
```

## JavaScript/TypeScript Client Implementation

### Basic Client Structure

```typescript
import { 
  BaseServiceClient, 
  ClientConfig,
  ClientError,
  ClientTimeoutError,
  ClientConnectionError,
  ClientValidationError
} from 'common-js-lib';

/**
 * Client for interacting with Example Service
 */
export class ExampleServiceClient extends BaseServiceClient {
  /**
   * Initialize the client
   * 
   * @param config Client configuration
   */
  constructor(config: ClientConfig) {
    super(config);
    this.logger = createLogger('ExampleServiceClient');
  }
  
  /**
   * Get a resource by ID
   * 
   * @param resourceId ID of the resource to retrieve
   * @returns Resource data
   * @throws ClientError if the request fails
   * @throws ClientTimeoutError if the request times out
   * @throws ClientConnectionError if connection to the service fails
   */
  async getResource(resourceId: string): Promise<any> {
    this.logger.debug(`Getting resource ${resourceId}`);
    try {
      return await this.get(`resources/${resourceId}`);
    } catch (error) {
      this.logger.error(`Failed to get resource ${resourceId}: ${error}`);
      throw new ClientError(
        `Failed to get resource ${resourceId}`,
        this.config.serviceName,
        { originalError: error }
      );
    }
  }
  
  /**
   * Create a new resource
   * 
   * @param data Resource data
   * @returns Created resource data
   * @throws ClientError if the request fails
   * @throws ClientValidationError if the data is invalid
   * @throws ClientTimeoutError if the request times out
   * @throws ClientConnectionError if connection to the service fails
   */
  async createResource(data: any): Promise<any> {
    this.logger.debug(`Creating resource: ${JSON.stringify(data)}`);
    try {
      return await this.post('resources', data);
    } catch (error) {
      this.logger.error(`Failed to create resource: ${error}`);
      throw new ClientError(
        'Failed to create resource',
        this.config.serviceName,
        { originalError: error }
      );
    }
  }
}
```

### Client Factory

```typescript
import { ClientConfig } from 'common-js-lib';
import { ExampleServiceClient } from './ExampleServiceClient';

// Client instances
let exampleServiceClient: ExampleServiceClient | null = null;

/**
 * Initialize all service clients
 * 
 * @param config Optional configuration overrides
 */
export function initializeClients(config: Record<string, any> = {}): void {
  // Create client configurations
  const exampleServiceConfig: ClientConfig = {
    baseUrl: config.EXAMPLE_SERVICE_URL || 'http://example-service:8000/api/v1',
    serviceName: 'example-service',
    timeoutMs: config.EXAMPLE_SERVICE_TIMEOUT || 30000,
    circuitBreaker: {
      failureThreshold: config.EXAMPLE_SERVICE_CB_THRESHOLD || 5,
      resetTimeoutMs: config.EXAMPLE_SERVICE_CB_RESET || 30000
    },
    retry: {
      maxRetries: config.EXAMPLE_SERVICE_RETRY_MAX || 3,
      initialDelayMs: config.EXAMPLE_SERVICE_RETRY_DELAY || 100,
      backoffFactor: config.EXAMPLE_SERVICE_RETRY_BACKOFF || 2.0
    }
  };
  
  // Create client instances
  exampleServiceClient = new ExampleServiceClient(exampleServiceConfig);
}

/**
 * Get the Example Service client
 * 
 * @returns ExampleServiceClient instance
 */
export function getExampleServiceClient(): ExampleServiceClient {
  if (!exampleServiceClient) {
    throw new Error('Clients not initialized. Call initializeClients() first.');
  }
  
  return exampleServiceClient;
}
```
