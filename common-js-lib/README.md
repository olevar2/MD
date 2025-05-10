# Common JavaScript Library

This library contains shared JavaScript utilities for the Forex Trading Platform.

## Features

- Standardized error handling with custom error classes
- Resilience patterns (circuit breaker, retry, timeout, bulkhead)
- API client with built-in error handling and resilience
- Security middleware for API authentication
- JWT token validation

## Installation

```bash
npm install ../common-js-lib
```

## Usage

### Error Handling

```javascript
const { errors, errorHandler } = require('common-js-lib');

// Using error classes
try {
  // Some code that might throw an error
  throw new errors.DataValidationError('Invalid data', { field: 'amount' });
} catch (error) {
  // Handle the error
  errorHandler.handleError(error, { component: 'MyComponent' });

  // Get a user-friendly error message
  const message = errorHandler.formatErrorMessage(error);
  console.log(message); // "Invalid data"
}

// Creating custom errors
const myError = errorHandler.createError(
  'Something went wrong',
  'DataFetchError',
  { url: '/api/data' }
);

// Using error middleware (Express)
app.use(errorHandler.errorMiddleware);
```

### Resilience Patterns

```javascript
const { resilience } = require('common-js-lib');

// Circuit breaker
const circuitBreaker = new resilience.CircuitBreaker('my-service');

async function callService() {
  return circuitBreaker.execute(async () => {
    // Call the service
    return await fetch('https://api.example.com/data');
  });
}

// Retry policy
const retryPolicy = new resilience.RetryPolicy({
  maxRetries: 3,
  initialDelayMs: 100
});

async function fetchWithRetry() {
  return retryPolicy.execute(async () => {
    // Call the service
    return await fetch('https://api.example.com/data');
  });
}

// Combined resilience patterns
const resilientFetch = resilience.withResilience(
  () => fetch('https://api.example.com/data'),
  {
    circuitBreaker,
    retryPolicy,
    timeoutMs: 5000
  },
  { serviceName: 'example-api' }
);

// Call the resilient function
const response = await resilientFetch();
```

### API Client

```javascript
const { apiClient, ApiClient } = require('common-js-lib');

// Using the default API client
const data = await apiClient.defaultApiClient.get('/users');

// Creating a custom API client
const myClient = new ApiClient({
  baseURL: 'https://api.example.com',
  timeout: 5000,
  retry: {
    maxRetries: 5
  }
});

// Making requests
const users = await myClient.get('/users');
const user = await myClient.post('/users', { name: 'John' });
await myClient.put('/users/1', { name: 'John Doe' });
await myClient.delete('/users/1');
```

### Security

```javascript
const { security } = require('common-js-lib');

// Validate API key
app.use(security.validateApiKey);

// Validate JWT token
app.use(security.validateJwtToken);
```

## Error Classes

The library provides a comprehensive set of error classes:

- `ForexTradingPlatformError` - Base error class
- `ConfigurationError` - Configuration-related errors
- `DataError` - Data-related errors
  - `DataValidationError` - Data validation errors
  - `DataFetchError` - Data fetching errors
  - `DataStorageError` - Data storage errors
  - `DataTransformationError` - Data transformation errors
- `ServiceError` - Service-related errors
  - `ServiceUnavailableError` - Service unavailable errors
  - `ServiceTimeoutError` - Service timeout errors
- `AuthenticationError` - Authentication failures
- `AuthorizationError` - Authorization failures
- `TradingError` - Trading-related errors
  - `OrderExecutionError` - Order execution errors
- `UIError` - UI-related errors
  - `RenderingError` - Rendering errors
- `NetworkError` - Network-related errors
- Resilience-related errors
  - `CircuitBreakerOpenError` - Circuit breaker is open
  - `RetryExhaustedError` - Retry attempts exhausted
  - `BulkheadFullError` - Bulkhead is full

## Testing

```bash
npm test
```
