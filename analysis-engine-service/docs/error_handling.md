# Error Handling in Analysis Engine Service

## Overview

The Analysis Engine Service uses a standardized exception hierarchy that aligns with the platform-wide error handling approach. All exceptions inherit from the common-lib `ForexTradingPlatformError` base class, ensuring consistent error handling and reporting across the platform.

## Exception Hierarchy

```
ForexTradingPlatformError (from common-lib)
├── AnalysisEngineError (service base exception)
│   ├── ValidationError
│   ├── DataFetchError
│   ├── AnalysisError
│   ├── ConfigurationError
│   └── ServiceUnavailableError
```

## Using Exceptions

### Importing Exceptions

Always import exceptions from the bridge module:

```python
from analysis_engine.core.exceptions_bridge import (
    ForexTradingPlatformError,
    DataValidationError,
    DataFetchError,
    ConfigurationError,
    ServiceError
)

# For service-specific exceptions
from analysis_engine.core.errors import (
    AnalysisEngineError,
    ValidationError,
    AnalysisError
)
```

### Raising Exceptions

When raising exceptions, always include a descriptive message and relevant details:

```python
# Common-lib exception
raise DataFetchError(
    message="Failed to fetch market data",
    source="external_api",
    details={"symbol": "EUR/USD", "timeframe": "1h"}
)

# Service-specific exception
raise AnalysisError(
    message="Analysis failed due to insufficient data",
    details={
        "analyzer": "market_regime",
        "required_points": 100,
        "available_points": 50
    }
)
```

### Handling Exceptions

The service automatically handles all exceptions that inherit from `ForexTradingPlatformError`. When catching exceptions, prefer catching specific exception types rather than the base exception:

```python
try:
    result = analyzer.analyze(market_data)
except DataFetchError as e:
    logger.error(f"Failed to fetch data: {e.message}", extra=e.details)
    # Handle data fetch error
except AnalysisError as e:
    logger.error(f"Analysis failed: {e.message}", extra=e.details)
    # Handle analysis error
except ForexTradingPlatformError as e:
    logger.error(f"Unexpected error: {e.message}", extra=e.details)
    # Handle any other platform error
```

## Creating Custom Exceptions

If you need to create a new exception type, extend from the appropriate base class:

```python
from analysis_engine.core.errors import AnalysisEngineError

class MarketRegimeAnalysisError(AnalysisError):
    """Exception raised when market regime analysis fails."""

    def __init__(
        self,
        message: str,
        regime_type: str = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if regime_type:
            details["regime_type"] = regime_type
        super().__init__(message=message, details=details)
```

## Error Response Format

All errors are returned in a standardized JSON format that aligns with the platform-wide approach:

```json
{
  "error_type": "AnalysisError",
  "error_code": "ANALYSIS_ERROR",
  "message": "Analysis failed due to insufficient data",
  "details": {
    "analyzer": "market_regime",
    "required_points": 100,
    "available_points": 50
  }
}
```

The error response includes the following fields:

- `error_type`: The type of error (e.g., "DataFetchError", "ValidationError")
- `error_code`: A unique code that identifies the error (e.g., "DATA_FETCH_ERROR")
- `message`: A human-readable message describing the error
- `details`: Additional details about the error, which vary depending on the error type

## HTTP Status Codes

The service uses appropriate HTTP status codes for different types of errors:

- `400 Bad Request`: Validation errors, invalid input
- `401 Unauthorized`: Authentication errors
- `403 Forbidden`: Authorization errors
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Unexpected errors
- `503 Service Unavailable`: External service unavailable

## Migration Guide

When migrating existing code to use the new exception hierarchy:

1. Replace direct imports from `analysis_engine.core.errors` with imports from `analysis_engine.core.exceptions_bridge`
2. Update exception handling to catch both service-specific and common-lib exceptions
3. Ensure that all raised exceptions include appropriate details for debugging

## Exception Handlers

The service registers exception handlers for all common-lib exceptions and service-specific exceptions. These handlers:

1. Log the error with appropriate context
2. Convert the exception to a standardized JSON response using the `to_dict()` method
3. Return the response with an appropriate HTTP status code

Example of an exception handler:

```python
async def data_fetch_exception_handler(request: Request, exc: DataFetchError) -> JSONResponse:
    """Handle DataFetchError exceptions."""
    logger.error(
        f"Data fetch error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=exc.to_dict(),
    )
```

## Best Practices

1. **Be Specific**: Use the most specific exception type that applies to the error
2. **Include Details**: Always include relevant details in the exception
3. **Document Exceptions**: Document all exceptions that your function might raise
4. **Handle Gracefully**: Catch and handle exceptions at appropriate levels
5. **Log Appropriately**: Log exceptions with appropriate severity levels
6. **Consistent Format**: Ensure all error responses follow the standardized format
7. **Use to_dict()**: Always use the `to_dict()` method to convert exceptions to JSON responses
