"""
Service-Specific Exceptions Bridge

This module provides a bridge between the common-lib error handling system and service-specific error handling.
It includes decorators for adding standardized exception handling to functions and utilities for
converting between different error types.

To use this template:
1. Copy this file to your service's error directory
2. Replace SERVICE_NAME with your service name
3. Replace FeatureStoreError with your service-specific error class
4. Add any additional service-specific exceptions
"""

import functools
import logging
import traceback
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union, cast

# Import common-lib exceptions bridge
from common_lib.errors.exceptions_bridge import (
    # Base exceptions
    BaseError,
    ErrorCode,
    
    # Error classes
    ValidationError,
    DatabaseError,
    APIError,
    ServiceError,
    DataError,
    BusinessError,
    SecurityError,
    ForexTradingError,
    ServiceUnavailableError,
    ThirdPartyServiceError,
    TimeoutError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    ConflictError,
    RateLimitError,
    
    # Base decorators
    with_exception_handling as base_with_exception_handling,
    async_with_exception_handling as base_async_with_exception_handling
)

# Type variable for function
F = TypeVar('F', bound=Callable[..., Any])

# Create logger
logger = logging.getLogger(__name__)

# Service name
SERVICE_NAME = "feature-store"  # Replace with your service name


# Service-specific error class
class FeatureStoreError(ServiceError):
    """Base exception for service-specific errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.SERVICE_ERROR,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize the service-specific error.
        
        Args:
            message: Human-readable error message
            error_code: Error code from the ErrorCode enum
            details: Additional error details
            correlation_id: Correlation ID for tracking the error across services
            cause: Original exception that caused this error
        """
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            correlation_id=correlation_id,
            cause=cause
        )


# Add additional service-specific exceptions here
class FeatureStoreValidationError(ValidationError):
    """Exception for service-specific validation errors."""
    pass


class FeatureStoreDataError(DataError):
    """Exception for service-specific data errors."""
    pass


class FeatureStoreBusinessError(BusinessError):
    """Exception for service-specific business logic errors."""
    pass


def with_exception_handling(
    func: Optional[F] = None,
    *,
    error_class: Type[BaseError] = FeatureStoreError,
    log_level: int = logging.ERROR,
    include_traceback: bool = True,
    reraise: bool = True,
    correlation_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    cleanup_func: Optional[Callable[[], None]] = None
) -> Union[F, Callable[[F], F]]:
    """
    Decorator to add standardized exception handling to synchronous functions.
    
    This decorator:
    1. Catches all exceptions
    2. Logs the exception with appropriate context
    3. Optionally runs a cleanup function
    4. Converts generic exceptions to service-specific exceptions
    5. Optionally reraises the exception
    
    Args:
        func: The function to decorate
        error_class: The error class to use for wrapping exceptions
        log_level: The logging level to use
        include_traceback: Whether to include traceback in the error details
        reraise: Whether to reraise the exception
        correlation_id: Correlation ID for tracking the error
        context: Additional context information to include in logs
        cleanup_func: Optional function to call for cleanup on error
        
    Returns:
        Decorated function with exception handling
    """
    return base_with_exception_handling(
        func=func,
        error_class=error_class,
        log_level=log_level,
        include_traceback=include_traceback,
        reraise=reraise,
        correlation_id=correlation_id,
        context=context,
        cleanup_func=cleanup_func,
        service_name=SERVICE_NAME
    )


def async_with_exception_handling(
    func: Optional[F] = None,
    *,
    error_class: Type[BaseError] = FeatureStoreError,
    log_level: int = logging.ERROR,
    include_traceback: bool = True,
    reraise: bool = True,
    correlation_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    cleanup_func: Optional[Callable[[], Any]] = None
) -> Union[F, Callable[[F], F]]:
    """
    Decorator to add standardized exception handling to asynchronous functions.
    
    This decorator:
    1. Catches all exceptions
    2. Logs the exception with appropriate context
    3. Optionally runs a cleanup function
    4. Converts generic exceptions to service-specific exceptions
    5. Optionally reraises the exception
    
    Args:
        func: The function to decorate
        error_class: The error class to use for wrapping exceptions
        log_level: The logging level to use
        include_traceback: Whether to include traceback in the error details
        reraise: Whether to reraise the exception
        correlation_id: Correlation ID for tracking the error
        context: Additional context information to include in logs
        cleanup_func: Optional function to call for cleanup on error
        
    Returns:
        Decorated function with exception handling
    """
    return base_async_with_exception_handling(
        func=func,
        error_class=error_class,
        log_level=log_level,
        include_traceback=include_traceback,
        reraise=reraise,
        correlation_id=correlation_id,
        context=context,
        cleanup_func=cleanup_func,
        service_name=SERVICE_NAME
    )


# Re-export all imported exceptions
__all__ = [
    # Error codes
    "ErrorCode",
    
    # Base exceptions
    "BaseError",
    
    # Common error classes
    "ValidationError",
    "DatabaseError",
    "APIError",
    "ServiceError",
    "DataError",
    "BusinessError",
    "SecurityError",
    "ForexTradingError",
    "ServiceUnavailableError",
    "ThirdPartyServiceError",
    "TimeoutError",
    "AuthenticationError",
    "AuthorizationError",
    "NotFoundError",
    "ConflictError",
    "RateLimitError",
    
    # Service-specific error classes
    "FeatureStoreError",
    "FeatureStoreValidationError",
    "FeatureStoreDataError",
    "FeatureStoreBusinessError",
    
    # Decorators
    "with_exception_handling",
    "async_with_exception_handling"
]
