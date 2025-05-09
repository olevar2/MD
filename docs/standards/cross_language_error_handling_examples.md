# Cross-Language Error Handling Examples

This document provides practical examples of implementing cross-language error handling in the Forex Trading Platform.

## Table of Contents

1. [Overview](#overview)
2. [Python Service with JavaScript Client](#python-service-with-javascript-client)
3. [JavaScript Service with Python Client](#javascript-service-with-python-client)
4. [Hybrid Service with Both Languages](#hybrid-service-with-both-languages)
5. [API Gateway Error Handling](#api-gateway-error-handling)
6. [Testing Cross-Language Error Handling](#testing-cross-language-error-handling)

## Overview

The Forex Trading Platform uses a standardized approach to error handling across language boundaries:

1. **Common Error Types**: Both Python and JavaScript use the same error hierarchy
2. **Error Mapping**: Errors are mapped between languages while preserving type, code, message, and details
3. **Consistent API Responses**: All API endpoints return errors in the same format
4. **Correlation ID Propagation**: Correlation IDs are used to track errors across services

## Python Service with JavaScript Client

### Python Service Implementation

```python
# service.py
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from common_lib.error import (
    ForexTradingPlatformError,
    ServiceError,
    DataValidationError,
    convert_to_js_error,
    create_error_response
)
from common_lib.correlation import get_correlation_id, with_correlation_id

app = FastAPI()

# Register error handlers
@app.exception_handler(ForexTradingPlatformError)
async def forex_error_handler(request: Request, exc: ForexTradingPlatformError):
    """Handle ForexTradingPlatformError exceptions."""
    # Get correlation ID from request
    correlation_id = request.headers.get("X-Correlation-ID") or get_correlation_id()
    
    # Create error response
    response = create_error_response(exc, correlation_id)
    
    # Determine status code
    status_code = 500
    if isinstance(exc, ServiceError):
        if exc.error_code == "SERVICE_UNAVAILABLE":
            status_code = 503
        elif exc.error_code == "SERVICE_TIMEOUT":
            status_code = 504
    elif isinstance(exc, DataValidationError):
        status_code = 400
    
    return JSONResponse(
        status_code=status_code,
        content=response
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    # Get correlation ID from request
    correlation_id = request.headers.get("X-Correlation-ID") or get_correlation_id()
    
    # Create error response
    response = create_error_response(exc, correlation_id)
    
    return JSONResponse(
        status_code=500,
        content=response
    )

# API endpoint
@app.get("/api/v1/resources/{resource_id}")
@with_correlation_id
async def get_resource(resource_id: str, request: Request):
    """Get a resource by ID."""
    try:
        # Validate input
        if not resource_id:
            raise DataValidationError(
                "Resource ID is required",
                error_code="MISSING_RESOURCE_ID"
            )
        
        # Get resource (might raise exceptions)
        resource = service.get_resource(resource_id)
        
        # Return successful response
        return {"data": resource, "success": True}
    except Exception as e:
        # Let the exception handlers handle the error
        raise
```

### JavaScript Client Implementation

```typescript
// client.ts
import {
  BaseServiceClient,
  ClientConfig,
  ForexTradingPlatformError,
  ServiceError,
  DataValidationError,
  convertFromPythonError,
  handlePythonErrorResponse
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
   * @throws Various errors depending on the response
   */
  async getResource(resourceId: string): Promise<any> {
    this.logger.debug(`Getting resource ${resourceId}`);
    try {
      // Make the request
      const response = await this.get(`resources/${resourceId}`);
      
      // Check for success
      if (response.success === false && response.error) {
        // Handle error response
        throw handlePythonErrorResponse(response);
      }
      
      // Return data
      return response.data;
    } catch (error) {
      // If it's already a platform error, rethrow
      if (error instanceof ForexTradingPlatformError) {
        throw error;
      }
      
      // Otherwise, wrap in a ServiceError
      this.logger.error(`Failed to get resource ${resourceId}: ${error}`);
      throw new ServiceError(
        `Failed to get resource ${resourceId}`,
        this.config.serviceName,
        'REQUEST_FAILED',
        { originalError: error }
      );
    }
  }
}

// Usage example
async function fetchResource() {
  const client = new ExampleServiceClient({
    baseUrl: 'http://example-service:8000/api/v1',
    serviceName: 'example-service'
  });
  
  try {
    const resource = await client.getResource('resource-id');
    console.log('Resource:', resource);
  } catch (error) {
    if (error instanceof DataValidationError) {
      console.error(`Validation error: ${error.message}`);
    } else if (error instanceof ServiceError) {
      console.error(`Service error: ${error.message}`);
    } else {
      console.error(`Unexpected error: ${error}`);
    }
  }
}
```

## JavaScript Service with Python Client

### JavaScript Service Implementation

```typescript
// service.ts
import express from 'express';
import {
  ForexTradingPlatformError,
  ServiceError,
  DataValidationError,
  convertToPythonError,
  createErrorResponse,
  getCorrelationId,
  withCorrelationId
} from 'common-js-lib';

const app = express();

// Parse JSON bodies
app.use(express.json());

// Correlation ID middleware
app.use((req, res, next) => {
  const correlationId = req.headers['x-correlation-id'] as string || getCorrelationId();
  res.setHeader('X-Correlation-ID', correlationId);
  next();
});

// Error handler middleware
app.use((err: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {
  // Get correlation ID
  const correlationId = req.headers['x-correlation-id'] as string || getCorrelationId();
  
  // Create error response
  const response = createErrorResponse(err, correlationId);
  
  // Determine status code
  let statusCode = 500;
  if (err instanceof DataValidationError) {
    statusCode = 400;
  } else if (err instanceof ServiceError) {
    if (err.code === 'SERVICE_UNAVAILABLE') {
      statusCode = 503;
    } else if (err.code === 'SERVICE_TIMEOUT') {
      statusCode = 504;
    }
  }
  
  // Log the error
  console.error(`Error handling request: ${err.message}`, {
    correlationId,
    statusCode,
    errorType: err.constructor.name,
    path: req.path
  });
  
  // Send response
  res.status(statusCode).json(response);
});

// API endpoint
app.get('/api/v1/resources/:resourceId', withCorrelationId(async (req, res, next) => {
  try {
    const resourceId = req.params.resourceId;
    
    // Validate input
    if (!resourceId) {
      throw new DataValidationError(
        'Resource ID is required',
        'MISSING_RESOURCE_ID'
      );
    }
    
    // Get resource (might throw errors)
    const resource = await service.getResource(resourceId);
    
    // Return successful response
    res.json({ data: resource, success: true });
  } catch (error) {
    next(error);
  }
}));

// Start server
app.listen(3000, () => {
  console.log('Server listening on port 3000');
});
```

### Python Client Implementation

```python
# client.py
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from common_lib.clients.base_client import BaseServiceClient, ClientConfig
from common_lib.clients.exceptions import ClientError, ClientTimeoutError
from common_lib.error import (
    ForexTradingPlatformError,
    ServiceError,
    DataValidationError,
    convert_from_js_error,
    handle_js_error_response
)
from common_lib.correlation import get_correlation_id, with_correlation_id

class ExampleServiceClient(BaseServiceClient):
    """Client for interacting with Example Service."""
    
    def __init__(self, config: Union[ClientConfig, Dict[str, Any]]):
        """Initialize the client."""
        super().__init__(config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @with_correlation_id
    async def get_resource(self, resource_id: str) -> Dict[str, Any]:
        """
        Get a resource by ID.
        
        Args:
            resource_id: ID of the resource to retrieve
            
        Returns:
            Resource data
            
        Raises:
            Various exceptions depending on the response
        """
        self.logger.debug(f"Getting resource {resource_id}")
        try:
            # Make the request
            response = await self.get(f"resources/{resource_id}")
            
            # Check for success
            if response.get("success") is False and "error" in response:
                # Handle error response
                raise handle_js_error_response(response)
            
            # Return data
            return response.get("data", {})
        except Exception as e:
            # If it's already a platform error, rethrow
            if isinstance(e, ForexTradingPlatformError):
                raise
            
            # Otherwise, wrap in a ServiceError
            self.logger.error(f"Failed to get resource {resource_id}: {str(e)}")
            raise ServiceError(
                f"Failed to get resource {resource_id}",
                service_name=self.config.service_name,
                error_code="REQUEST_FAILED",
                details={"original_error": str(e)}
            )

# Usage example
async def fetch_resource():
    client = ExampleServiceClient({
        "base_url": "http://example-service:3000/api/v1",
        "service_name": "example-service"
    })
    
    try:
        resource = await client.get_resource("resource-id")
        print(f"Resource: {resource}")
    except DataValidationError as e:
        print(f"Validation error: {e.message}")
    except ServiceError as e:
        print(f"Service error: {e.message}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
```
