# Trading Gateway Service Cleanup Plan

This document outlines the cleanup plan for the Trading Gateway Service after implementing the error handling improvements.

## Files to Keep

### Python Files
- `trading_gateway_service/error/exceptions_bridge.py` - New centralized error handling module
- `trading_gateway_service/error/exception_handlers.py` - New FastAPI exception handlers
- `trading_gateway_service/error/__init__.py` - Updated package initialization

### JavaScript Files
- `src/utils/errorBridge.js` - New JavaScript-Python error bridge
- `src/utils/errors.js` - Existing error classes (keep and use with the bridge)
- `src/middleware/errorHandler.js` - Existing error handler middleware (keep and update to use the bridge)

### Test Files
- `tests/test_error_handling.py` - Tests for Python error handling
- `tests/test_error_handlers.py` - Tests for FastAPI exception handlers
- `tests/test_error_bridge.js` - Tests for JavaScript-Python error bridge
- `tests/test_error_handler_middleware.js` - Tests for error handler middleware

## Integration Plan

### 1. JavaScript Error Handling Integration

Update `src/middleware/errorHandler.js` to use the new error bridge:

```javascript
const {
  convertPythonError,
  handleError
} = require('../utils/errorBridge');

// In the error handler function:
function errorHandler(err, req, res, next) {
  // Use the error bridge to handle the error
  const handledError = handleError(err, {
    path: req.path,
    method: req.method
  });
  
  // Rest of the function remains the same...
}
```

### 2. Python Error Handling Integration

Ensure all Python components use the new error handling:

1. Update imports in all files to use the new error module:
   ```python
   from trading_gateway_service.error import (
       ForexTradingPlatformError,
       ServiceError,
       # Other exceptions as needed
       handle_exception,
       with_exception_handling,
       async_with_exception_handling
   )
   ```

2. Apply the exception handling decorators to critical functions:
   ```python
   @async_with_exception_handling
   async def critical_function():
       # Function implementation
   ```

## Testing Plan

1. Run all tests to ensure they pass with the new error handling:
   ```bash
   # Python tests
   python -m pytest tests/

   # JavaScript tests
   npm test
   ```

2. Manually test error scenarios to ensure they're handled correctly:
   - Test API endpoints with invalid input
   - Test API endpoints with server errors
   - Test JavaScript-Python error conversion
   - Test error logging and reporting
