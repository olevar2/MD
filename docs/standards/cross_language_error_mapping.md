# Cross-Language Error Mapping

This document provides detailed examples and best practices for implementing error mapping between Python and JavaScript/TypeScript in the Forex Trading Platform.

## Table of Contents

1. [Overview](#overview)
2. [Python to JavaScript Error Mapping](#python-to-javascript-error-mapping)
3. [JavaScript to Python Error Mapping](#javascript-to-python-error-mapping)
4. [API Error Responses](#api-error-responses)
5. [Implementation Examples](#implementation-examples)
6. [Testing Error Mapping](#testing-error-mapping)

## Overview

The Forex Trading Platform implements a bidirectional error mapping system that:

1. Converts Python exceptions to JavaScript errors and vice versa
2. Preserves error types, codes, messages, and details across language boundaries
3. Ensures consistent error handling in hybrid services
4. Provides standardized error responses from all API endpoints

## Python to JavaScript Error Mapping

### Error Mapping Function

```python
from typing import Dict, Any, Optional, Type
import traceback

from common_lib.error import (
    ForexTradingPlatformError,
    ServiceError,
    ServiceUnavailableError,
    ServiceTimeoutError,
    DataError,
    DataValidationError,
    AuthenticationError,
    AuthorizationError,
    TradingError,
    OrderExecutionError,
    AnalysisError,
    MLError
)

def convert_to_js_error(
    exception: Exception,
    correlation_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convert a Python exception to a JavaScript error format.
    
    Args:
        exception: The Python exception to convert
        correlation_id: Optional correlation ID for tracking
        
    Returns:
        Dictionary representing the error in JavaScript format
    """
    # Get error type
    if isinstance(exception, ForexTradingPlatformError):
        error_type = exception.__class__.__name__
        error_code = exception.error_code
        message = exception.message
        details = exception.details.copy() if exception.details else {}
    else:
        error_type = exception.__class__.__name__
        error_code = "UNKNOWN_ERROR"
        message = str(exception)
        details = {"original_error": error_type}
    
    # Add correlation ID if provided
    if correlation_id:
        details["correlation_id"] = correlation_id
    
    # Add timestamp
    from datetime import datetime
    details["timestamp"] = datetime.utcnow().isoformat()
    
    # Create error object
    error_obj = {
        "error_type": error_type,
        "error_code": error_code,
        "message": message,
        "details": details
    }
    
    return error_obj

def create_error_response(
    exception: Exception,
    correlation_id: Optional[str] = None,
    include_traceback: bool = False
) -> Dict[str, Any]:
    """
    Create a standardized error response for API endpoints.
    
    Args:
        exception: The exception to convert
        correlation_id: Optional correlation ID for tracking
        include_traceback: Whether to include traceback in the response
        
    Returns:
        Standardized error response
    """
    # Convert exception to JavaScript error format
    error_obj = convert_to_js_error(exception, correlation_id)
    
    # Add traceback if requested (and not in production)
    if include_traceback:
        error_obj["details"]["traceback"] = traceback.format_exc()
    
    # Create response
    response = {
        "error": error_obj,
        "success": False
    }
    
    return response
```

### Usage Example

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from common_lib.error import (
    ForexTradingPlatformError,
    ServiceError,
    convert_to_js_error,
    create_error_response
)

app = FastAPI()

@app.exception_handler(ForexTradingPlatformError)
async def forex_error_handler(request: Request, exc: ForexTradingPlatformError):
    """Handle ForexTradingPlatformError exceptions."""
    # Get correlation ID from request
    correlation_id = request.headers.get("X-Correlation-ID")
    
    # Create error response
    response = create_error_response(exc, correlation_id)
    
    # Determine status code
    status_code = 500
    if isinstance(exc, ServiceError):
        if exc.error_code == "SERVICE_UNAVAILABLE":
            status_code = 503
        elif exc.error_code == "SERVICE_TIMEOUT":
            status_code = 504
    
    return JSONResponse(
        status_code=status_code,
        content=response
    )

@app.get("/api/v1/resources/{resource_id}")
async def get_resource(resource_id: str, request: Request):
    """Get a resource by ID."""
    try:
        # Code that might raise an exception
        resource = service.get_resource(resource_id)
        return {"data": resource, "success": True}
    except Exception as e:
        # Convert to JavaScript error format
        correlation_id = request.headers.get("X-Correlation-ID")
        response = create_error_response(e, correlation_id)
        return JSONResponse(
            status_code=500,
            content=response
        )
```

## JavaScript to Python Error Mapping

### Error Mapping Function

```typescript
import {
  ForexTradingPlatformError,
  ServiceError,
  ServiceUnavailableError,
  ServiceTimeoutError,
  DataError,
  DataValidationError,
  AuthenticationError,
  AuthorizationError,
  TradingError,
  OrderExecutionError,
  AnalysisError,
  MLError
} from './errors';

/**
 * Map of Python error types to JavaScript error classes
 */
const PYTHON_TO_JS_ERROR_MAPPING: Record<string, any> = {
  'ForexTradingPlatformError': ForexTradingPlatformError,
  'ServiceError': ServiceError,
  'ServiceUnavailableError': ServiceUnavailableError,
  'ServiceTimeoutError': ServiceTimeoutError,
  'DataError': DataError,
  'DataValidationError': DataValidationError,
  'AuthenticationError': AuthenticationError,
  'AuthorizationError': AuthorizationError,
  'TradingError': TradingError,
  'OrderExecutionError': OrderExecutionError,
  'AnalysisError': AnalysisError,
  'MLError': MLError,
  // Add more mappings as needed
};

/**
 * Convert a JavaScript error to a Python error format
 * 
 * @param error JavaScript error to convert
 * @param correlationId Optional correlation ID for tracking
 * @returns Object representing the error in Python format
 */
export function convertToPythonError(
  error: Error,
  correlationId?: string
): Record<string, any> {
  // Get error type and details
  let errorType = 'ForexTradingPlatformError';
  let errorCode = 'UNKNOWN_ERROR';
  let message = error.message || 'Unknown error';
  let details: Record<string, any> = {};
  
  if (error instanceof ForexTradingPlatformError) {
    errorType = error.constructor.name;
    errorCode = error.code;
    message = error.message;
    details = error.details || {};
  }
  
  // Add correlation ID if provided
  if (correlationId) {
    details.correlation_id = correlationId;
  }
  
  // Add timestamp
  details.timestamp = new Date().toISOString();
  
  // Create error object
  const errorObj = {
    error_type: errorType,
    error_code: errorCode,
    message: message,
    details: details
  };
  
  return errorObj;
}

/**
 * Convert a Python error to a JavaScript error
 * 
 * @param errorData Object representing the error in Python format
 * @returns JavaScript error
 */
export function convertFromPythonError(errorData: Record<string, any>): Error {
  // Extract error information
  const pythonErrorType = errorData.error_type || 'ForexTradingPlatformError';
  const errorCode = errorData.error_code || 'UNKNOWN_ERROR';
  const message = errorData.message || 'Unknown error';
  const details = errorData.details || {};
  const correlationId = errorData.correlation_id;
  
  // Map to JavaScript error class
  const ErrorClass = PYTHON_TO_JS_ERROR_MAPPING[pythonErrorType] || ForexTradingPlatformError;
  
  // Create error
  const error = new ErrorClass(message, errorCode, details);
  
  // Add correlation ID if available
  if (correlationId) {
    error.details.correlationId = correlationId;
  }
  
  return error;
}

/**
 * Handle an error response from a Python service
 * 
 * @param responseData Response data containing error information
 * @returns JavaScript error
 */
export function handlePythonErrorResponse(responseData: Record<string, any>): Error {
  // Check if response contains error information
  if (responseData.error) {
    return convertFromPythonError(responseData.error);
  }
  
  // Fallback for unexpected response format
  return new ServiceError(
    'Unexpected error response format',
    'UNEXPECTED_ERROR_FORMAT',
    { responseData }
  );
}

/**
 * Create a standardized error response for API endpoints
 * 
 * @param error Error to convert
 * @param correlationId Optional correlation ID for tracking
 * @returns Standardized error response
 */
export function createErrorResponse(
  error: Error,
  correlationId?: string
): Record<string, any> {
  // Convert error to Python format
  const errorObj = convertToPythonError(error, correlationId);
  
  // Create response
  const response = {
    error: errorObj,
    success: false
  };
  
  return response;
}
```

### Usage Example

```typescript
import express from 'express';
import {
  ForexTradingPlatformError,
  ServiceError,
  convertToPythonError,
  createErrorResponse
} from 'common-js-lib';

const app = express();

// Error handler middleware
app.use((err: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {
  // Get correlation ID from request
  const correlationId = req.headers['x-correlation-id'] as string;
  
  // Create error response
  const response = createErrorResponse(err, correlationId);
  
  // Determine status code
  let statusCode = 500;
  if (err instanceof ServiceError) {
    if (err.code === 'SERVICE_UNAVAILABLE') {
      statusCode = 503;
    } else if (err.code === 'SERVICE_TIMEOUT') {
      statusCode = 504;
    }
  }
  
  res.status(statusCode).json(response);
});

app.get('/api/v1/resources/:resourceId', (req, res, next) => {
  try {
    // Code that might throw an error
    const resource = service.getResource(req.params.resourceId);
    res.json({ data: resource, success: true });
  } catch (error) {
    next(error);
  }
});
```
