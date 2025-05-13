# Error Handling System

This module provides a standardized error handling system for the Forex Trading Platform. It includes error classes, error handling decorators, error handling middleware, and error handling utilities.

## Key Features

1. **Standardized Error Classes**: A comprehensive hierarchy of error classes for different types of errors
2. **Error Handling Decorators**: Decorators for adding standardized error handling to functions
3. **Error Handling Middleware**: Middleware for handling errors in API endpoints
4. **Error Handling Utilities**: Utilities for creating standardized error responses
5. **Correlation ID Tracking**: Automatic generation and propagation of correlation IDs for tracking errors across services

## Error Class Hierarchy

The error class hierarchy is designed to provide a consistent way to represent and handle errors across the platform:

```
BaseError (base class for all errors)
├── ValidationError
├── DatabaseError
├── APIError
├── ServiceError
│   ├── ServiceUnavailableError
│   └── ThirdPartyServiceError
├── DataError
├── BusinessError
├── SecurityError
│   ├── AuthenticationError
│   └── AuthorizationError
├── ForexTradingError
├── TimeoutError
├── NotFoundError
├── ConflictError
└── RateLimitError
```

## Error Handling Decorators

Two decorators are provided for adding standardized error handling to functions:

1. `with_exception_handling`: For synchronous functions
2. `async_with_exception_handling`: For asynchronous functions

These decorators:
- Catch all exceptions
- Log the exception with appropriate context
- Optionally run a cleanup function
- Convert generic exceptions to domain-specific exceptions
- Optionally reraise the exception

Example usage:

```python
from common_lib.errors import with_exception_handling, async_with_exception_handling, ServiceError

# Decorate a synchronous function
@with_exception_handling(error_class=ServiceError)
def process_data(data):
    # Function implementation
    pass

# Decorate an asynchronous function
@async_with_exception_handling(error_class=ServiceError)
async def fetch_data():
    # Function implementation
    pass
```

## Error Handling Middleware

Middleware is provided for handling errors in API endpoints:

1. `FastAPIErrorMiddleware`: Middleware for FastAPI applications

Example usage:

```python
from fastapi import FastAPI
from common_lib.errors import fastapi_error_handler

app = FastAPI()

# Add error handling middleware
fastapi_error_handler(app, include_traceback=False)
```

## Error Handling Utilities

Utilities are provided for creating standardized error responses:

1. `create_error_response`: Create a standardized error response for API endpoints
2. `fastapi_exception_handler`: Exception handler for FastAPI applications

Example usage:

```python
from fastapi import FastAPI, Request, HTTPException
from common_lib.errors import create_error_response, fastapi_exception_handler

app = FastAPI()

# Register exception handler
app.add_exception_handler(Exception, fastapi_exception_handler)

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    try:
        # Function implementation
        pass
    except Exception as e:
        # Create error response
        error_response, status_code = create_error_response(e)
        return JSONResponse(status_code=status_code, content=error_response)
```

## Correlation ID Tracking

Correlation IDs are automatically generated and propagated for tracking errors across services. They are:

1. Generated when an error is first encountered
2. Included in error responses
3. Included in log messages
4. Propagated through HTTP headers

## Best Practices

1. **Use Domain-Specific Error Classes**: Create domain-specific error classes for your service by extending the appropriate base error class.

2. **Use Error Handling Decorators**: Decorate functions with the appropriate error handling decorator to ensure consistent error handling.

3. **Use Error Handling Middleware**: Add error handling middleware to your API endpoints to ensure consistent error responses.

4. **Include Correlation IDs**: Always include correlation IDs in error responses and log messages to enable tracking errors across services.

5. **Log Errors Appropriately**: Use the appropriate log level for different types of errors.

## Example Implementation

Here's an example of how to implement error handling in a service:

```python
from fastapi import FastAPI, Depends, HTTPException
from common_lib.errors import (
    with_exception_handling,
    async_with_exception_handling,
    fastapi_error_handler,
    create_error_response,
    ServiceError,
    ValidationError
)

# Create FastAPI application
app = FastAPI()

# Add error handling middleware
fastapi_error_handler(app, include_traceback=False)

# Define domain-specific error class
class UserServiceError(ServiceError):
    """Base error class for user service."""
    pass

# Define service function with error handling
@async_with_exception_handling(error_class=UserServiceError)
async def get_user(user_id: int):
    # Function implementation
    if user_id < 0:
        raise ValidationError(f"Invalid user ID: {user_id}")
    # ...
    return {"id": user_id, "name": "John Doe"}

# Define API endpoint
@app.get("/users/{user_id}")
async def read_user(user_id: int):
    return await get_user(user_id)
```

## Error Response Format

All error responses follow a standardized format:

```json
{
  "error": {
    "message": "Error message",
    "error_code": "ERROR_CODE",
    "correlation_id": "correlation-id",
    "details": {
      "additional": "details"
    },
    "timestamp": "2023-05-15T10:00:00.000Z"
  }
}
```

This format ensures that error responses are consistent across all services and provides all the information needed to diagnose and track errors.
