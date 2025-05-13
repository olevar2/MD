"""
Exceptions Bridge Module

This module provides a bridge between the common-lib error handling system and service-specific error handling.
It includes decorators for adding standardized exception handling to functions and utilities for
converting between different error types.

Features:
- Standardized exception handling decorators for synchronous and asynchronous functions
- Service-specific error conversion
- Correlation ID propagation
- Structured logging
- Traceback handling
"""

import functools
import logging
import traceback
import uuid
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union, cast

# Import common-lib exceptions
from common_lib.errors.base_exceptions import (
    ErrorCode,
    BaseError,
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
    RateLimitError
)

# Type variable for function
F = TypeVar('F', bound=Callable[..., Any])

# Create logger
logger = logging.getLogger(__name__)


def with_exception_handling(
    func: Optional[F] = None,
    *,
    error_class: Type[BaseError] = ServiceError,
    log_level: int = logging.ERROR,
    include_traceback: bool = True,
    reraise: bool = True,
    correlation_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    cleanup_func: Optional[Callable[[], None]] = None,
    service_name: Optional[str] = None
) -> Union[F, Callable[[F], F]]:
    """
    Decorator to add standardized exception handling to synchronous functions.

    This decorator:
    1. Catches all exceptions
    2. Logs the exception with appropriate context
    3. Optionally runs a cleanup function
    4. Converts generic exceptions to domain-specific exceptions
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
        service_name: Name of the service for error context

    Returns:
        Decorated function with exception handling
    """
    def decorator(func: F) -> F:
    """
    Decorator.
    
    Args:
        func: Description of func
    
    Returns:
        F: Description of return value
    
    """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

            # Generate correlation ID if not provided
            nonlocal correlation_id
            if correlation_id is None:
                correlation_id = str(uuid.uuid4())

            # Create context dictionary
            context_dict = {
                "correlation_id": correlation_id,
                "function": func.__name__
            }
            if service_name:
                context_dict["service"] = service_name
            if context:
                context_dict.update(context)

            try:
                # Call the function
                return func(*args, **kwargs)
            except Exception as e:
                # Run cleanup function if provided
                if cleanup_func:
                    try:
                        cleanup_func()
                    except Exception as cleanup_error:
                        logger.error(
                            f"Error in cleanup function: {str(cleanup_error)}",
                            extra=context_dict
                        )

                # Handle the exception
                if isinstance(e, BaseError):
                    # Set correlation ID if not already set
                    if not e.correlation_id:
                        e.correlation_id = correlation_id

                    # Log the error
                    log_message = f"{e.__class__.__name__}: {e.message} (Code: {e.error_code.name}, ID: {e.correlation_id})"
                    logger.log(log_level, log_message, extra=context_dict)

                    # Reraise if requested
                    if reraise:
                        raise

                    # Return None if not reraising
                    return None
                else:
                    # Convert to domain-specific error
                    error_message = str(e)
                    error_details = {"original_error": error_message}

                    # Add traceback if requested
                    if include_traceback:
                        error_details["traceback"] = traceback.format_exc()

                    # Create domain-specific error
                    domain_error = error_class(
                        message=f"Error in {func.__name__}: {error_message}",
                        error_code=ErrorCode.UNKNOWN_ERROR,
                        details=error_details,
                        correlation_id=correlation_id,
                        cause=e
                    )

                    # Log the error
                    log_message = f"{domain_error.__class__.__name__}: {domain_error.message} (Code: {domain_error.error_code.name}, ID: {domain_error.correlation_id})"
                    logger.log(log_level, log_message, extra=context_dict)

                    # Reraise if requested
                    if reraise:
                        raise domain_error

                    # Return None if not reraising
                    return None

        return cast(F, wrapper)

    # Handle both @with_exception_handling and @with_exception_handling()
    if func is None:
        return decorator
    return decorator(func)


def async_with_exception_handling(
    func: Optional[F] = None,
    *,
    error_class: Type[BaseError] = ServiceError,
    log_level: int = logging.ERROR,
    include_traceback: bool = True,
    reraise: bool = True,
    correlation_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    cleanup_func: Optional[Callable[[], Any]] = None,
    service_name: Optional[str] = None
) -> Union[F, Callable[[F], F]]:
    """
    Decorator to add standardized exception handling to asynchronous functions.

    This decorator:
    1. Catches all exceptions
    2. Logs the exception with appropriate context
    3. Optionally runs a cleanup function
    4. Converts generic exceptions to domain-specific exceptions
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
        service_name: Name of the service for error context

    Returns:
        Decorated function with exception handling
    """
    def decorator(func: F) -> F:
    """
    Decorator.
    
    Args:
        func: Description of func
    
    Returns:
        F: Description of return value
    
    """

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

            # Generate correlation ID if not provided
            nonlocal correlation_id
            if correlation_id is None:
                correlation_id = str(uuid.uuid4())

            # Create context dictionary
            context_dict = {
                "correlation_id": correlation_id,
                "function": func.__name__
            }
            if service_name:
                context_dict["service"] = service_name
            if context:
                context_dict.update(context)

            try:
                # Call the function
                return await func(*args, **kwargs)
            except Exception as e:
                # Run cleanup function if provided
                if cleanup_func:
                    try:
                        result = cleanup_func()
                        if hasattr(result, "__await__"):
                            await result
                    except Exception as cleanup_error:
                        logger.error(
                            f"Error in cleanup function: {str(cleanup_error)}",
                            extra=context_dict
                        )

                # Handle the exception
                if isinstance(e, BaseError):
                    # Set correlation ID if not already set
                    if not e.correlation_id:
                        e.correlation_id = correlation_id

                    # Log the error
                    log_message = f"{e.__class__.__name__}: {e.message} (Code: {e.error_code.name}, ID: {e.correlation_id})"
                    logger.log(log_level, log_message, extra=context_dict)

                    # Reraise if requested
                    if reraise:
                        raise

                    # Return None if not reraising
                    return None
                else:
                    # Convert to domain-specific error
                    error_message = str(e)
                    error_details = {"original_error": error_message}

                    # Add traceback if requested
                    if include_traceback:
                        error_details["traceback"] = traceback.format_exc()

                    # Create domain-specific error
                    domain_error = error_class(
                        message=f"Error in {func.__name__}: {error_message}",
                        error_code=ErrorCode.UNKNOWN_ERROR,
                        details=error_details,
                        correlation_id=correlation_id,
                        cause=e
                    )

                    # Log the error
                    log_message = f"{domain_error.__class__.__name__}: {domain_error.message} (Code: {domain_error.error_code.name}, ID: {domain_error.correlation_id})"
                    logger.log(log_level, log_message, extra=context_dict)

                    # Reraise if requested
                    if reraise:
                        raise domain_error

                    # Return None if not reraising
                    return None

        return cast(F, wrapper)

    # Handle both @async_with_exception_handling and @async_with_exception_handling()
    if func is None:
        return decorator
    return decorator(func)


# Re-export all imported exceptions
__all__ = [
    # Error codes
    "ErrorCode",
    
    # Base exceptions
    "BaseError",
    
    # Error classes
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
    
    # Decorators
    "with_exception_handling",
    "async_with_exception_handling"
]
