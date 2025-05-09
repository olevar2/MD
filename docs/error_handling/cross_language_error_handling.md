# Cross-Language Error Handling

This document explains the standardized error handling patterns across language boundaries in the Forex Trading Platform.

## Overview

The platform implements a comprehensive error handling system that:

1. Provides consistent error handling across both Python and JavaScript/TypeScript components
2. Uses common-lib exceptions for standardized error types
3. Includes proper logging and context for all errors
4. Ensures consistent error responses from all API endpoints
5. Supports correlation ID propagation for tracing errors across services

## Key Components

### Python Components

- **exceptions.py**: Defines standardized exception classes for the platform
- **error_bridge.py**: Provides bidirectional error conversion between Python and JavaScript

### JavaScript/TypeScript Components

- **errors.ts**: Defines standardized error classes for the platform
- **errorBridge.ts**: Provides bidirectional error conversion between JavaScript and Python

## Error Hierarchy

Both Python and JavaScript implementations follow the same error hierarchy:

```
ForexTradingPlatformError
├── ConfigurationError
├── DataError
│   ├── DataValidationError
│   ├── DataFetchError
│   ├── DataStorageError
│   └── DataTransformationError
├── ServiceError
│   ├── ServiceUnavailableError
│   └── ServiceTimeoutError
├── AuthenticationError
├── AuthorizationError
├── NetworkError
├── TradingError
│   └── OrderExecutionError
├── AnalysisError
└── MLError
```

## Error Structure

All errors in both languages include:

1. **Message**: Human-readable error message
2. **Error Code**: Machine-readable error code for programmatic handling
3. **Details**: Additional error details as a dictionary/object
4. **Correlation ID**: Optional ID for tracing errors across services

## Using the Error Handling System

### In Python Code

```python
from common_lib.error.exceptions import (
    ForexTradingPlatformError,
    ServiceError,
    DataValidationError
)

# Raising a platform exception
raise ServiceError(
    message="Failed to connect to service",
    error_code="SERVICE_UNAVAILABLE",
    details={"service_name": "market-data-service"}
)

# Handling platform exceptions
try:
    # Code that might raise an exception
    result = service.get_data()
except DataValidationError as e:
    logger.error(f"Validation error: {e.message}", extra={"error_code": e.error_code})
    # Handle validation error
except ServiceError as e:
    logger.error(f"Service error: {e.message}", extra={"error_code": e.error_code})
    # Handle service error
except ForexTradingPlatformError as e:
    logger.error(f"Platform error: {e.message}", extra={"error_code": e.error_code})
    # Handle any platform error
except Exception as e:
    logger.error(f"Unexpected error: {str(e)}")
    # Handle unexpected error
```

### In JavaScript/TypeScript Code

```typescript
import { 
  ForexTradingPlatformError,
  ServiceError,
  DataValidationError
} from 'common-js-lib/errors';

// Raising a platform error
throw new ServiceError(
  "Failed to connect to service",
  "SERVICE_UNAVAILABLE",
  { serviceName: "market-data-service" }
);

// Handling platform errors
try {
  // Code that might throw an error
  const result = await service.getData();
} catch (error) {
  if (error instanceof DataValidationError) {
    console.error(`Validation error: ${error.message}`, { errorCode: error.code });
    // Handle validation error
  } else if (error instanceof ServiceError) {
    console.error(`Service error: ${error.message}`, { errorCode: error.code });
    // Handle service error
  } else if (error instanceof ForexTradingPlatformError) {
    console.error(`Platform error: ${error.message}`, { errorCode: error.code });
    // Handle any platform error
  } else {
    console.error(`Unexpected error: ${error}`);
    // Handle unexpected error
  }
}
```

## Cross-Language Error Conversion

### Python to JavaScript

```python
from common_lib.error.error_bridge import convert_to_js_error, create_error_response

# Convert a Python exception to a JavaScript error format
try:
    # Code that might raise an exception
    result = service.get_data()
except Exception as e:
    # Convert to JavaScript error format
    js_error = convert_to_js_error(e, correlation_id="example-correlation-id")
    
    # Create a standardized error response
    response = create_error_response(e, correlation_id="example-correlation-id")
    return response
```

### JavaScript to Python

```typescript
import { 
  convertToPythonError, 
  createErrorResponse 
} from 'common-js-lib/errorBridge';

// Convert a JavaScript error to a Python error format
try {
  // Code that might throw an error
  const result = await service.getData();
} catch (error) {
  // Convert to Python error format
  const pythonError = convertToPythonError(error, 'example-correlation-id');
  
  // Create a standardized error response
  const response = createErrorResponse(error, 'example-correlation-id');
  return response;
}
```

## API Error Responses

All API endpoints return errors in a consistent format:

```json
{
  "error": {
    "type": "ServiceError",
    "code": "SERVICE_UNAVAILABLE",
    "message": "Failed to connect to service",
    "details": {
      "service_name": "market-data-service"
    },
    "correlationId": "example-correlation-id",
    "timestamp": "2023-06-01T12:34:56.789Z"
  },
  "success": false
}
```

## Handling API Error Responses

### In Python Code

```python
from common_lib.error.error_bridge import handle_js_error_response

# Make a request to a JavaScript service
response = await client.get("https://js-service.example.com/api/resource")

# Check if the response contains an error
if "error" in response:
    # Convert the error response to a Python exception
    exception = handle_js_error_response(response)
    # Handle the exception
    logger.error(f"API error: {exception.message}", extra={"error_code": exception.error_code})
    raise exception
```

### In JavaScript/TypeScript Code

```typescript
import { handlePythonErrorResponse } from 'common-js-lib/errorBridge';

// Make a request to a Python service
try {
  const response = await fetch('https://python-service.example.com/api/resource');
  const data = await response.json();
  
  // Check if the response contains an error
  if (data.error) {
    // Convert the error response to a JavaScript error
    const error = handlePythonErrorResponse(data);
    // Handle the error
    console.error(`API error: ${error.message}`, { errorCode: error.code });
    throw error;
  }
  
  return data;
} catch (error) {
  // Handle fetch errors
  console.error(`Fetch error: ${error}`);
  throw error;
}
```

## Error Decorators

### Python Decorators

```python
from common_lib.error.exceptions import async_with_exception_handling, with_exception_handling

# Decorate async functions
@async_with_exception_handling
async def my_async_function():
    # Function implementation
    pass

# Decorate sync functions
@with_exception_handling
def my_sync_function():
    # Function implementation
    pass
```

### JavaScript/TypeScript Decorators

```typescript
import { withAsyncErrorHandling, withErrorHandling } from 'common-js-lib/errorBridge';

// Decorate async functions
const myAsyncFunction = withAsyncErrorHandling(async function() {
  // Function implementation
});

// Decorate sync functions
const mySyncFunction = withErrorHandling(function() {
  // Function implementation
});
```

## Best Practices

1. **Use Platform Exceptions**: Always use platform-specific exceptions instead of generic ones
2. **Provide Context**: Include relevant context in error details
3. **Consistent Error Codes**: Use consistent error codes across the platform
4. **Proper Logging**: Log errors with appropriate severity and context
5. **Correlation IDs**: Use correlation IDs for tracing errors across services
6. **Graceful Degradation**: Handle errors gracefully and provide fallbacks when possible
7. **User-Friendly Messages**: Provide user-friendly error messages for client-facing errors
8. **Don't Swallow Exceptions**: Always handle or propagate exceptions, don't ignore them

## Implementation Details

### File Structure

```
common-lib/
├── common_lib/
│   ├── error/
│   │   ├── __init__.py
│   │   ├── exceptions.py
│   │   └── error_bridge.py
│   └── clients/
│       ├── __init__.py
│       ├── exceptions.py
│       └── ...

common-js-lib/
├── errors.ts
├── errorBridge.ts
└── ...
```

### Error Mapping

Both Python and JavaScript implementations include mappings between error types:

```python
# Python to JavaScript error mapping
PYTHON_TO_JS_ERROR_MAPPING = {
    "ForexTradingPlatformError": "ForexTradingPlatformError",
    "ConfigurationError": "ConfigurationError",
    "DataError": "DataError",
    # ...
}

# JavaScript to Python error mapping
JS_TO_PYTHON_ERROR_MAPPING = {
    "ForexTradingPlatformError": ForexTradingPlatformError,
    "ConfigurationError": ConfigurationError,
    "DataError": DataError,
    # ...
}
```

```typescript
// JavaScript to Python error mapping
const JS_TO_PYTHON_ERROR_MAPPING: Record<string, string> = {
  'ForexTradingPlatformError': 'ForexTradingPlatformError',
  'ConfigurationError': 'ConfigurationError',
  'DataError': 'DataError',
  // ...
};

// Python to JavaScript error mapping
const PYTHON_TO_JS_ERROR_MAPPING: Record<string, any> = {
  'ForexTradingPlatformError': ForexTradingPlatformError,
  'ConfigurationError': ConfigurationError,
  'DataError': DataError,
  // ...
};
```

## Conclusion

The standardized error handling system ensures consistent error handling across language boundaries in the Forex Trading Platform. By following these patterns, we can provide a better developer experience and make it easier to debug and troubleshoot issues across services.