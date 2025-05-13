"""
Standardized Error Handling Module for ML Integration Service

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

from core.logging_setup_standardized import get_service_logger


# Create a logger
logger = get_service_logger(__name__)


# Create a singleton error handler
error_handler = ErrorHandler(logger=logger)


# ML Integration Service specific errors
class MLIntegrationError(BaseError):
    """Base error for ML Integration Service."""
    
    error_code = ErrorCode.SERVICE_ERROR
    default_message = "ML Integration Service error"
    status_code = 500


class ModelNotFoundError(MLIntegrationError):
    """Error when a model is not found."""
    
    error_code = ErrorCode.NOT_FOUND
    default_message = "Model not found"
    status_code = 404


class ModelVersionNotFoundError(MLIntegrationError):
    """Error when a model version is not found."""
    
    error_code = ErrorCode.NOT_FOUND
    default_message = "Model version not found"
    status_code = 404


class ModelTrainingError(MLIntegrationError):
    """Error when training a model."""
    
    error_code = ErrorCode.PROCESSING_ERROR
    default_message = "Error training model"
    status_code = 500


class ModelPredictionError(MLIntegrationError):
    """Error when making predictions with a model."""
    
    error_code = ErrorCode.PROCESSING_ERROR
    default_message = "Error making predictions with model"
    status_code = 500


class ModelEvaluationError(MLIntegrationError):
    """Error when evaluating a model."""
    
    error_code = ErrorCode.PROCESSING_ERROR
    default_message = "Error evaluating model"
    status_code = 500


class ModelOptimizationError(MLIntegrationError):
    """Error when optimizing a model."""
    
    error_code = ErrorCode.PROCESSING_ERROR
    default_message = "Error optimizing model"
    status_code = 500


class FeatureExtractionError(MLIntegrationError):
    """Error when extracting features."""
    
    error_code = ErrorCode.PROCESSING_ERROR
    default_message = "Error extracting features"
    status_code = 500


class ModelRegistryError(MLIntegrationError):
    """Error when accessing the model registry."""
    
    error_code = ErrorCode.EXTERNAL_SERVICE_ERROR
    default_message = "Error accessing model registry"
    status_code = 502


class MLWorkbenchError(MLIntegrationError):
    """Error when accessing the ML Workbench Service."""
    
    error_code = ErrorCode.EXTERNAL_SERVICE_ERROR
    default_message = "Error accessing ML Workbench Service"
    status_code = 502


# Register custom errors with the error handler
error_handler.register_error(MLIntegrationError)
error_handler.register_error(ModelNotFoundError)
error_handler.register_error(ModelVersionNotFoundError)
error_handler.register_error(ModelTrainingError)
error_handler.register_error(ModelPredictionError)
error_handler.register_error(ModelEvaluationError)
error_handler.register_error(ModelOptimizationError)
error_handler.register_error(FeatureExtractionError)
error_handler.register_error(ModelRegistryError)
error_handler.register_error(MLWorkbenchError)


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
