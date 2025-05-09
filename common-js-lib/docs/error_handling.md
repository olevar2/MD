# Error Handling in Common JavaScript Library

This document provides detailed information about the error handling implementation in the common-js-lib.

## Overview

The error handling system in common-js-lib provides:

1. **Standardized Error Classes**: A comprehensive hierarchy of error classes that mirror the Python exceptions in common-lib
2. **Error Handling Utilities**: Functions for logging, formatting, and handling errors in a consistent way
3. **Correlation ID Tracking**: Automatic generation and propagation of correlation IDs for tracking errors across services
4. **Resilience Patterns**: Circuit breaker, retry, timeout, and bulkhead patterns to improve service reliability
5. **API Client Integration**: A standardized API client with built-in error handling and resilience

## Error Class Hierarchy

The error class hierarchy mirrors the Python exceptions in common-lib to ensure consistent error handling across the platform:

```
ForexTradingPlatformError (base error)
├── ConfigurationError
│   ├── ConfigNotFoundError
│   └── ConfigValidationError
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
├── TradingError
│   └── OrderExecutionError
├── UIError
│   └── RenderingError
└── NetworkError
```

Additionally, there are specialized errors for resilience patterns:

```
ForexTradingPlatformError
├── ServiceUnavailableError
│   ├── CircuitBreakerOpenError
│   └── BulkheadFullError
└── ServiceError
    └── RetryExhaustedError
```

## Using Error Classes

All error classes extend the base `ForexTradingPlatformError` class, which provides common functionality:

```javascript
const { errors } = require('common-js-lib');

// Create a basic error
const error = new errors.ForexTradingPlatformError(
  'Something went wrong',
  'CUSTOM_ERROR_CODE',
  { additionalInfo: 'Some details' }
);

// Create a specific error
const validationError = new errors.DataValidationError(
  'Invalid data',
  { field: 'amount', value: -100 },
  { source: 'user-input' }
);

// Convert error to JSON
const errorJson = validationError.toJSON();
/*
{
  error_type: 'DataValidationError',
  error_code: 'DATA_VALIDATION_ERROR',
  message: 'Invalid data',
  details: {
    data: { field: 'amount', value: -100 },
    source: 'user-input'
  },
  timestamp: '2025-05-13T12:34:56.789Z'
}
*/
```

## Error Handling Utilities

The `errorHandler` module provides utilities for handling errors:

### Handling Errors

```javascript
const { errorHandler } = require('common-js-lib');

try {
  // Some code that might throw an error
  throw new Error('Something went wrong');
} catch (error) {
  // Handle the error
  const correlationId = errorHandler.handleError(error, {
    component: 'MyComponent',
    operation: 'processData'
  });
  
  console.log(`Error handled with correlation ID: ${correlationId}`);
}
```

### Formatting Error Messages

```javascript
const { errorHandler } = require('common-js-lib');

// Format an error for display to the user
const message = errorHandler.formatErrorMessage(error);
```

### Creating Custom Errors

```javascript
const { errorHandler } = require('common-js-lib');

// Create a custom error
const error = errorHandler.createError(
  'Something went wrong',
  'DataFetchError',
  { url: '/api/data' }
);
```

### Error Middleware for Express

```javascript
const express = require('express');
const { errorHandler } = require('common-js-lib');

const app = express();

// Add error middleware
app.use(errorHandler.errorMiddleware);
```

## Correlation ID Tracking

Correlation IDs are used to track errors across services:

```javascript
const { errorHandler } = require('common-js-lib');

// Generate a correlation ID
const correlationId = errorHandler.generateCorrelationId();

// Get correlation ID from request
const correlationId = errorHandler.getCorrelationId(req);
```

## Resilience Patterns

The `resilience` module provides patterns for improving service reliability:

### Circuit Breaker

```javascript
const { resilience } = require('common-js-lib');

// Create a circuit breaker
const circuitBreaker = new resilience.CircuitBreaker('my-service', {
  failureThreshold: 5,
  resetTimeoutMs: 30000
});

// Use the circuit breaker
async function callService() {
  return circuitBreaker.execute(async () => {
    // Call the service
    return await fetch('https://api.example.com/data');
  });
}
```

### Retry Policy

```javascript
const { resilience } = require('common-js-lib');

// Create a retry policy
const retryPolicy = new resilience.RetryPolicy({
  maxRetries: 3,
  initialDelayMs: 100,
  maxDelayMs: 3000,
  backoffFactor: 2,
  jitter: true
});

// Use the retry policy
async function fetchWithRetry() {
  return retryPolicy.execute(async () => {
    // Call the service
    return await fetch('https://api.example.com/data');
  });
}
```

### Bulkhead

```javascript
const { resilience } = require('common-js-lib');

// Create a bulkhead
const bulkhead = new resilience.Bulkhead('my-service', {
  maxConcurrent: 10,
  maxQueue: 20
});

// Use the bulkhead
async function callService() {
  return bulkhead.execute(async () => {
    // Call the service
    return await fetch('https://api.example.com/data');
  });
}
```

### Timeout

```javascript
const { resilience } = require('common-js-lib');

// Use timeout
async function callWithTimeout() {
  return resilience.withTimeout(
    async () => {
      // Call the service
      return await fetch('https://api.example.com/data');
    },
    5000, // 5 seconds
    { serviceName: 'example-api' }
  );
}
```

### Combined Resilience Patterns

```javascript
const { resilience } = require('common-js-lib');

// Combine resilience patterns
const resilientFetch = resilience.withResilience(
  () => fetch('https://api.example.com/data'),
  {
    circuitBreaker: {
      failureThreshold: 5,
      resetTimeoutMs: 30000
    },
    retryPolicy: {
      maxRetries: 3,
      initialDelayMs: 100
    },
    bulkhead: {
      maxConcurrent: 10,
      maxQueue: 20
    },
    timeoutMs: 5000
  },
  { serviceName: 'example-api' }
);

// Call the resilient function
const response = await resilientFetch();
```

## API Client Integration

The `apiClient` module provides a standardized API client with built-in error handling and resilience:

```javascript
const { apiClient } = require('common-js-lib');

// Create a custom API client
const myClient = new apiClient.ApiClient({
  baseURL: 'https://api.example.com',
  timeout: 5000,
  retry: {
    maxRetries: 5
  },
  circuitBreaker: {
    failureThreshold: 3
  }
});

// Make requests
try {
  const users = await myClient.get('/users');
  console.log(users);
} catch (error) {
  // Error is already handled by the client
  console.error(`Request failed: ${error.message}`);
}
```

## Best Practices

1. **Use Specific Error Classes**: Use the most specific error class for the situation to provide better context
2. **Include Details**: Always include relevant details in the error to aid debugging
3. **Propagate Correlation IDs**: Ensure correlation IDs are propagated across service boundaries
4. **Log Appropriately**: Use the error handling utilities to ensure consistent logging
5. **Implement Resilience Patterns**: Use circuit breakers, retries, and bulkheads to improve service reliability
6. **Handle Errors Gracefully**: Catch and handle errors at appropriate levels
7. **Provide User-Friendly Messages**: Use `formatErrorMessage` to provide user-friendly error messages
8. **Monitor Errors**: Implement error monitoring to track and analyze errors
