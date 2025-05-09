# Error Handling in Monitoring & Alerting Service

This document describes the error handling approach used in the Monitoring & Alerting Service.

## Overview

The Monitoring & Alerting Service uses a standardized error handling approach that:

1. Defines domain-specific exceptions that extend the common-lib exception hierarchy
2. Provides decorators for consistent error handling in service methods
3. Implements standardized error responses with correlation IDs
4. Converts exceptions to appropriate HTTP status codes
5. Ensures proper error logging with contextual information

## Exception Hierarchy

The service extends the common-lib exception hierarchy with domain-specific exceptions:

- `MonitoringAlertingError` (base exception)
  - `AlertNotFoundError`
  - `NotificationError`
  - `AlertStorageError`
  - `MetricsExporterError`
  - `DashboardError`
  - `AlertRuleError`
  - `ThresholdValidationError`

## Error Handling Decorators

Two decorators are provided for consistent error handling:

1. `with_exception_handling` - For synchronous functions
2. `async_with_exception_handling` - For asynchronous functions

These decorators:
- Catch and log all exceptions
- Pass through domain-specific exceptions
- Convert generic exceptions to `MonitoringAlertingError`

Example usage:

```python
@with_exception_handling
def add_alert(self, alert: Alert) -> str:
    # Implementation
```

## Error Response Format

All error responses follow a standardized format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "key1": "value1",
      "key2": "value2"
    },
    "correlation_id": "unique-correlation-id",
    "timestamp": "2023-05-22T12:34:56.789Z",
    "service": "monitoring-alerting-service"
  }
}
```

## HTTP Status Code Mapping

Exceptions are mapped to appropriate HTTP status codes:

- 400 Bad Request: `DataValidationError`, `ConfigValidationError`, `ThresholdValidationError`, `AlertRuleError`
- 401 Unauthorized: `AuthenticationError`, `ServiceAuthenticationError`
- 403 Forbidden: `AuthorizationError`
- 404 Not Found: `AlertNotFoundError`
- 408 Request Timeout: `TimeoutError`
- 422 Unprocessable Entity: `AlertStorageError`
- 429 Too Many Requests: `BulkheadFullError`
- 503 Service Unavailable: `ServiceUnavailableError`, `CircuitBreakerOpenError`, `ServiceTimeoutError`, `NotificationError`, `MetricsExporterError`
- 500 Internal Server Error: All other exceptions

## Correlation IDs

Correlation IDs are used to track requests across services:

1. The `CorrelationIdMiddleware` adds a correlation ID to each request
2. The correlation ID is included in all error responses
3. The correlation ID is included in all log messages
4. The correlation ID is propagated to downstream services

## Best Practices

When implementing new functionality:

1. Use the appropriate domain-specific exception
2. Apply the `with_exception_handling` or `async_with_exception_handling` decorator to service methods
3. Include relevant details in exception messages
4. Use the `convert_to_http_exception` function in API endpoints
5. Don't catch exceptions in API endpoints unless necessary
