# Error Handling in Trading Gateway Service

This document explains the error handling improvements implemented in the Trading Gateway Service.

## Overview

The Trading Gateway Service now has a comprehensive error handling system that:

1. Provides consistent error handling across both JavaScript and Python components
2. Uses common-lib exceptions for standardized error types
3. Includes proper logging and context for all errors
4. Ensures consistent error responses from all API endpoints

## Key Components

### 1. Python Error Handling

- **exceptions_bridge.py**: Bridges common-lib exceptions with service-specific exceptions
- **exception_handlers.py**: Provides FastAPI exception handlers for consistent error responses

### 2. JavaScript Error Handling

- **errorBridge.js**: Provides bidirectional error conversion between JavaScript and Python
- **errors.js**: Defines JavaScript error classes that mirror common-lib exceptions
- **errorHandler.js**: Express middleware for consistent error responses

## Using the Error Handling System

### In Python Code

```python
from trading_gateway_service.error import (
    ForexTradingPlatformError,
    ServiceError,
    handle_exception,
    with_exception_handling,
    async_with_exception_handling
)

# Using the decorator for async functions
@async_with_exception_handling
async def my_function():
    # Function implementation
    pass

# Using the decorator for sync functions
@with_exception_handling
def another_function():
    # Function implementation
    pass

# Manually handling exceptions
try:
    # Code that might raise an exception
    pass
except Exception as e:
    handle_exception(e, context={"operation": "my_operation"})
```

### In JavaScript Code

```javascript
const { 
  DataValidationError, 
  ServiceError 
} = require('./utils/errors');

const { 
  withErrorHandling, 
  withAsyncErrorHandling,
  convertPythonError
} = require('./utils/errorBridge');

// Using the decorator for async functions
router.get('/endpoint', withAsyncErrorHandling(async (req, res) => {
  // Implementation
}));

// Using the decorator for sync functions
const myFunction = withErrorHandling(function() {
  // Implementation
});

// Converting Python errors from API responses
try {
  const response = await axios.get('/api/endpoint');
  return response.data;
} catch (error) {
  if (error.response && error.response.data && error.response.data.error_type) {
    throw convertPythonError(error.response.data);
  }
  throw error;
}
```

## Error Types

The error handling system supports the following error types:

| Error Type | HTTP Status | Description |
|------------|-------------|-------------|
| ForexTradingPlatformError | 500 | Base error type for all platform errors |
| DataValidationError | 400 | Invalid input data |
| DataFetchError | 500 | Error fetching data |
| DataStorageError | 500 | Error storing data |
| ServiceError | 500 | Generic service error |
| ServiceUnavailableError | 503 | Service is unavailable |
| ServiceTimeoutError | 504 | Service request timed out |
| TradingError | 400 | Error related to trading operations |
| OrderExecutionError | 400 | Error executing an order |
| AuthenticationError | 401 | Authentication failed |
| AuthorizationError | 403 | Authorization failed |
| BrokerConnectionError | 503 | Failed to connect to broker |
| OrderValidationError | 400 | Order validation failed |
| MarketDataError | 500 | Error fetching market data |

## Testing the Error Handling

You can test the error handling system using the `/api/analysis/error-demo` endpoint:

```
GET /api/analysis/error-demo?errorType=validation
GET /api/analysis/error-demo?errorType=service
GET /api/analysis/error-demo?errorType=python
GET /api/analysis/error-demo?errorType=standard
```

Each endpoint will trigger a different type of error, demonstrating how the error handling system works.

## Benefits

- **Consistency**: All errors are handled consistently across the service
- **Context**: Errors include detailed context information for debugging
- **Logging**: All errors are properly logged with appropriate severity
- **Client-Friendly**: Error responses are formatted in a client-friendly way
- **Bridge**: Seamless error handling between JavaScript and Python components
