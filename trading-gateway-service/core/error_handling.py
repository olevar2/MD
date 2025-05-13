"""
Error Handling Module

This module provides error handling functionality for the Trading Gateway Service.
"""

import logging
import traceback
import sys
import uuid
from typing import Dict, Any, Optional, Type, List, Callable, Union
from functools import wraps

from common_lib.errors.base_exceptions import (
    BaseError,
    ValidationError,
    DataError,
    ServiceError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    ConflictError,
    RateLimitError,
    ErrorCode
)
from common_lib.errors.error_handler import ErrorHandler


# Create a singleton error handler
error_handler = ErrorHandler()


def handle_error(
    error: Exception,
    operation: str = "unknown",
    correlation_id: Optional[str] = None,
    include_traceback: bool = False
) -> Dict[str, Any]:
    """
    Handle an error and generate a standardized error response.
    
    Args:
        error: The error to handle
        operation: The operation that caused the error
        correlation_id: Optional correlation ID for tracking
        include_traceback: Whether to include traceback in the response
        
    Returns:
        Standardized error response
    """
    return error_handler.handle_error(
        error=error,
        operation=operation,
        correlation_id=correlation_id,
        include_traceback=include_traceback
    )


def handle_exception(
    operation: str = "unknown",
    correlation_id: Optional[str] = None,
    include_traceback: bool = False
) -> Callable:
    """
    Decorator for handling exceptions in functions.
    
    Args:
        operation: The operation being performed
        correlation_id: Optional correlation ID for tracking
        include_traceback: Whether to include traceback in the response
        
    Returns:
        Decorator function
    """
    return error_handler.handle_exception(
        operation=operation,
        correlation_id=correlation_id,
        include_traceback=include_traceback
    )


def handle_async_exception(
    operation: str = "unknown",
    correlation_id: Optional[str] = None,
    include_traceback: bool = False
) -> Callable:
    """
    Decorator for handling exceptions in async functions.
    
    Args:
        operation: The operation being performed
        correlation_id: Optional correlation ID for tracking
        include_traceback: Whether to include traceback in the response
        
    Returns:
        Decorator function
    """
    return error_handler.handle_async_exception(
        operation=operation,
        correlation_id=correlation_id,
        include_traceback=include_traceback
    )


def get_status_code(error: Exception) -> int:
    """
    Get the HTTP status code for an error.
    
    Args:
        error: The error to get the status code for
        
    Returns:
        HTTP status code
    """
    return error_handler.get_status_code(error)
