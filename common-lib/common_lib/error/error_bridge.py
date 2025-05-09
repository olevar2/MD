"""
Error Bridge Module

This module provides utilities for converting errors between Python and JavaScript/TypeScript.
It ensures consistent error handling across language boundaries in the Forex Trading Platform.

Key features:
1. Bidirectional error conversion between Python and JavaScript
2. Standardized error types across languages
3. Consistent error structure and properties
4. Correlation ID propagation
"""

import json
import logging
import traceback
from typing import Dict, Any, Optional, Type, Union, List

from common_lib.error.exceptions import (
    ForexTradingPlatformError,
    ConfigurationError,
    DataError,
    DataValidationError,
    DataFetchError,
    DataStorageError,
    DataTransformationError,
    ServiceError,
    ServiceUnavailableError,
    ServiceTimeoutError,
    AuthenticationError,
    AuthorizationError,
    NetworkError
)

logger = logging.getLogger(__name__)

# Mapping from Python exception types to JavaScript error types
PYTHON_TO_JS_ERROR_MAPPING = {
    "ForexTradingPlatformError": "ForexTradingPlatformError",
    "ConfigurationError": "ConfigurationError",
    "DataError": "DataError",
    "DataValidationError": "DataValidationError",
    "DataFetchError": "DataFetchError",
    "DataStorageError": "DataStorageError",
    "DataTransformationError": "DataTransformationError",
    "ServiceError": "ServiceError",
    "ServiceUnavailableError": "ServiceUnavailableError",
    "ServiceTimeoutError": "ServiceTimeoutError",
    "AuthenticationError": "AuthenticationError",
    "AuthorizationError": "AuthorizationError",
    "NetworkError": "NetworkError",
    # Add more mappings as needed
}

# Mapping from JavaScript error types to Python exception classes
JS_TO_PYTHON_ERROR_MAPPING = {
    "ForexTradingPlatformError": ForexTradingPlatformError,
    "ConfigurationError": ConfigurationError,
    "DataError": DataError,
    "DataValidationError": DataValidationError,
    "DataFetchError": DataFetchError,
    "DataStorageError": DataStorageError,
    "DataTransformationError": DataTransformationError,
    "ServiceError": ServiceError,
    "ServiceUnavailableError": ServiceUnavailableError,
    "ServiceTimeoutError": ServiceTimeoutError,
    "AuthenticationError": AuthenticationError,
    "AuthorizationError": AuthorizationError,
    "NetworkError": NetworkError,
    # Add more mappings as needed
}


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
        details = exception.details
    else:
        error_type = exception.__class__.__name__
        error_code = "UNKNOWN_ERROR"
        message = str(exception)
        details = {"original_error": error_type}
    
    # Map to JavaScript error type
    js_error_type = PYTHON_TO_JS_ERROR_MAPPING.get(error_type, "ForexTradingPlatformError")
    
    # Create error object
    error_obj = {
        "type": js_error_type,
        "code": error_code,
        "message": message,
        "details": details,
        "correlationId": correlation_id,
        "timestamp": None  # Will be set by JavaScript
    }
    
    return error_obj


def convert_from_js_error(
    error_data: Dict[str, Any]
) -> Exception:
    """
    Convert a JavaScript error to a Python exception.
    
    Args:
        error_data: Dictionary representing the error in JavaScript format
        
    Returns:
        Python exception
    """
    # Extract error information
    js_error_type = error_data.get("type", "ForexTradingPlatformError")
    error_code = error_data.get("code", "UNKNOWN_ERROR")
    message = error_data.get("message", "Unknown error")
    details = error_data.get("details", {})
    correlation_id = error_data.get("correlationId")
    
    # Map to Python exception class
    exception_class = JS_TO_PYTHON_ERROR_MAPPING.get(js_error_type, ForexTradingPlatformError)
    
    # Create exception
    if issubclass(exception_class, ForexTradingPlatformError):
        exception = exception_class(
            message=message,
            error_code=error_code,
            details=details
        )
        
        # Add correlation ID if available
        if correlation_id:
            exception.details["correlation_id"] = correlation_id
    else:
        # Fallback for non-platform exceptions
        exception = exception_class(message)
    
    return exception


def handle_js_error_response(
    response_data: Dict[str, Any]
) -> Exception:
    """
    Handle an error response from a JavaScript service.
    
    Args:
        response_data: Response data containing error information
        
    Returns:
        Python exception
    """
    # Check if response contains error information
    if "error" in response_data:
        error_data = response_data["error"]
        return convert_from_js_error(error_data)
    
    # Fallback for unexpected response format
    return ServiceError(
        message="Unexpected error response format",
        error_code="UNEXPECTED_ERROR_FORMAT",
        details={"response_data": response_data}
    )


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