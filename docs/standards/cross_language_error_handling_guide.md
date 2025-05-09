# Cross-Language Error Handling Guide

This guide provides detailed examples and best practices for implementing standardized error handling across language boundaries in the Forex Trading Platform. It covers both Python and JavaScript/TypeScript implementations.

## Table of Contents

1. [Overview](#overview)
2. [Error Hierarchy](#error-hierarchy)
3. [Python Error Handling](#python-error-handling)
4. [JavaScript/TypeScript Error Handling](#javascripttypescript-error-handling)
5. [Cross-Language Error Mapping](#cross-language-error-mapping)
6. [API Error Responses](#api-error-responses)
7. [Correlation ID Propagation](#correlation-id-propagation)
8. [Testing Error Handling](#testing-error-handling)

## Overview

The Forex Trading Platform implements a comprehensive error handling system that:

1. Provides consistent error handling across both Python and JavaScript/TypeScript components
2. Uses domain-specific exceptions for different error types
3. Includes proper logging and context for all errors
4. Ensures consistent error responses from all API endpoints
5. Supports correlation ID propagation for tracing errors across services

## Error Hierarchy

The platform uses a consistent error hierarchy across all languages:

```
ForexTradingPlatformError (Base error)
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
├── TradingError
│   ├── OrderExecutionError
│   ├── PositionError
│   └── AccountError
├── AnalysisError
│   ├── IndicatorError
│   ├── SignalError
│   └── ModelError
└── MLError
    ├── ModelTrainingError
    ├── ModelInferenceError
    └── FeatureError
```

## Python Error Handling

### Base Exception Classes

```python
from typing import Dict, Any, Optional

class ForexTradingPlatformError(Exception):
    """Base exception for all Forex Trading Platform errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception."""
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)

class ServiceError(ForexTradingPlatformError):
    """Exception for service-related errors."""
    
    def __init__(
        self,
        message: str,
        service_name: str,
        error_code: str = "SERVICE_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception."""
        details = details or {}
        details["service_name"] = service_name
        super().__init__(message, error_code, details)

class ServiceUnavailableError(ServiceError):
    """Exception for service unavailable errors."""
    
    def __init__(
        self,
        message: str,
        service_name: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception."""
        super().__init__(message, service_name, "SERVICE_UNAVAILABLE", details)

class ServiceTimeoutError(ServiceError):
    """Exception for service timeout errors."""
    
    def __init__(
        self,
        message: str,
        service_name: str,
        timeout_seconds: float,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception."""
        details = details or {}
        details["timeout_seconds"] = timeout_seconds
        super().__init__(message, service_name, "SERVICE_TIMEOUT", details)
```

### Using Exceptions

```python
from common_lib.error import (
    ServiceError,
    ServiceUnavailableError,
    ServiceTimeoutError
)

def get_data_from_service():
    """Get data from a service."""
    try:
        # Code that might raise an exception
        response = requests.get("https://api.example.com/data", timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        # Handle timeout
        raise ServiceTimeoutError(
            "Request to example service timed out",
            service_name="example-service",
            timeout_seconds=30
        )
    except requests.ConnectionError:
        # Handle connection error
        raise ServiceUnavailableError(
            "Failed to connect to example service",
            service_name="example-service"
        )
    except requests.HTTPError as e:
        # Handle HTTP error
        raise ServiceError(
            f"HTTP error from example service: {e}",
            service_name="example-service",
            error_code="HTTP_ERROR",
            details={"status_code": e.response.status_code}
        )
    except Exception as e:
        # Handle unexpected error
        raise ServiceError(
            f"Unexpected error from example service: {e}",
            service_name="example-service",
            error_code="UNEXPECTED_ERROR"
        )
```

## JavaScript/TypeScript Error Handling

### Base Error Classes

```typescript
export class ForexTradingPlatformError extends Error {
  /**
   * Error code
   */
  public readonly code: string;
  
  /**
   * Error details
   */
  public readonly details: Record<string, any>;
  
  /**
   * Create a new ForexTradingPlatformError
   * 
   * @param message Error message
   * @param code Error code
   * @param details Additional error details
   */
  constructor(
    message: string,
    code: string = 'UNKNOWN_ERROR',
    details: Record<string, any> = {}
  ) {
    super(message);
    this.name = this.constructor.name;
    this.code = code;
    this.details = details;
    
    // Ensure instanceof works correctly in TypeScript
    Object.setPrototypeOf(this, ForexTradingPlatformError.prototype);
  }
}

export class ServiceError extends ForexTradingPlatformError {
  /**
   * Create a new ServiceError
   * 
   * @param message Error message
   * @param serviceName Name of the service that caused the error
   * @param code Error code
   * @param details Additional error details
   */
  constructor(
    message: string,
    serviceName: string,
    code: string = 'SERVICE_ERROR',
    details: Record<string, any> = {}
  ) {
    super(
      message,
      code,
      { ...details, serviceName }
    );
    
    // Ensure instanceof works correctly in TypeScript
    Object.setPrototypeOf(this, ServiceError.prototype);
  }
}

export class ServiceUnavailableError extends ServiceError {
  /**
   * Create a new ServiceUnavailableError
   * 
   * @param message Error message
   * @param serviceName Name of the service that is unavailable
   * @param details Additional error details
   */
  constructor(
    message: string,
    serviceName: string,
    details: Record<string, any> = {}
  ) {
    super(message, serviceName, 'SERVICE_UNAVAILABLE', details);
    
    // Ensure instanceof works correctly in TypeScript
    Object.setPrototypeOf(this, ServiceUnavailableError.prototype);
  }
}

export class ServiceTimeoutError extends ServiceError {
  /**
   * Create a new ServiceTimeoutError
   * 
   * @param message Error message
   * @param serviceName Name of the service that timed out
   * @param timeoutMs Timeout in milliseconds
   * @param details Additional error details
   */
  constructor(
    message: string,
    serviceName: string,
    timeoutMs: number,
    details: Record<string, any> = {}
  ) {
    super(
      message,
      serviceName,
      'SERVICE_TIMEOUT',
      { ...details, timeoutMs }
    );
    
    // Ensure instanceof works correctly in TypeScript
    Object.setPrototypeOf(this, ServiceTimeoutError.prototype);
  }
}
```

### Using Errors

```typescript
import {
  ServiceError,
  ServiceUnavailableError,
  ServiceTimeoutError
} from 'common-js-lib';

async function getDataFromService(): Promise<any> {
  try {
    // Code that might throw an error
    const response = await fetch('https://api.example.com/data', {
      timeout: 30000
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    if (error.name === 'AbortError') {
      // Handle timeout
      throw new ServiceTimeoutError(
        'Request to example service timed out',
        'example-service',
        30000
      );
    } else if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
      // Handle connection error
      throw new ServiceUnavailableError(
        'Failed to connect to example service',
        'example-service'
      );
    } else if (error.message.includes('HTTP error')) {
      // Handle HTTP error
      throw new ServiceError(
        `HTTP error from example service: ${error.message}`,
        'example-service',
        'HTTP_ERROR',
        { statusCode: parseInt(error.message.split(':')[1].trim()) }
      );
    } else {
      // Handle unexpected error
      throw new ServiceError(
        `Unexpected error from example service: ${error.message}`,
        'example-service',
        'UNEXPECTED_ERROR'
      );
    }
  }
}
```
