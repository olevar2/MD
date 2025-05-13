"""
Standardized Error Handling Module for Data Pipeline Service

This module provides error handling functionality for the service using the standardized
error handling system from common-lib.
"""

import logging
import traceback
import sys
import uuid
from typing import Dict, Any, Optional, Type, List, Callable, Union
from functools import wraps

from common_lib.errors import (
    BaseError,
    ValidationError,
    DataError,
    ServiceError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    ConflictError,
    RateLimitError,
    ErrorCode,
    ErrorHandler
)
from common_lib.monitoring.tracing import trace_function

from data_pipeline_service.logging_setup_standardized import get_service_logger


# Create a logger
logger = get_service_logger(__name__)


# Create a singleton error handler
error_handler = ErrorHandler(logger=logger)


# Data Pipeline Service specific errors
class DataPipelineError(BaseError):
    """Base error for Data Pipeline Service."""
    
    error_code = ErrorCode.SERVICE_ERROR
    default_message = "Data Pipeline Service error"
    status_code = 500


class DataSourceError(DataPipelineError):
    """Error when accessing a data source."""
    
    error_code = ErrorCode.EXTERNAL_SERVICE_ERROR
    default_message = "Error accessing data source"
    status_code = 502


class DataProcessingError(DataPipelineError):
    """Error when processing data."""
    
    error_code = ErrorCode.PROCESSING_ERROR
    default_message = "Error processing data"
    status_code = 500


class DataExportError(DataPipelineError):
    """Error when exporting data."""
    
    error_code = ErrorCode.PROCESSING_ERROR
    default_message = "Error exporting data"
    status_code = 500


class FeatureStoreError(DataPipelineError):
    """Error when accessing the feature store."""
    
    error_code = ErrorCode.EXTERNAL_SERVICE_ERROR
    default_message = "Error accessing feature store"
    status_code = 502


# Register custom errors with the error handler
error_handler.register_error(DataPipelineError)
error_handler.register_error(DataSourceError)
error_handler.register_error(DataProcessingError)
error_handler.register_error(DataExportError)
error_handler.register_error(FeatureStoreError)


@trace_function(name="handle_error")
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


@trace_function(name="handle_exception")
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


@trace_function(name="handle_async_exception")
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


@trace_function(name="get_status_code")
def get_status_code(error: Exception) -> int:
    """
    Get the HTTP status code for an error.
    
    Args:
        error: The error to get the status code for
        
    Returns:
        HTTP status code
    """
    return error_handler.get_status_code(error)
