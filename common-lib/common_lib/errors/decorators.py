"""
Error Handling Decorators Module

This module provides standardized decorators for error handling across the platform.
These decorators ensure consistent error handling, logging, and reporting.
"""

import functools
import logging
import traceback
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Type, TypeVar, cast, Union

from common_lib.errors.base_exceptions import BaseError, ErrorCode, ServiceError

# Type variable for function return type
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

# Get logger
logger = logging.getLogger(__name__)


def with_exception_handling(
    error_class: Type[BaseError] = ServiceError,
    log_level: int = logging.ERROR,
    include_traceback: bool = True,
    reraise: bool = True,
    correlation_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    cleanup_func: Optional[Callable[[], None]] = None
) -> Callable[[F], F]:
    """
    Decorator to add standardized exception handling to synchronous functions.

    This decorator:
    1. Catches all exceptions
    2. Logs the exception with appropriate context
    3. Optionally runs a cleanup function
    4. Converts generic exceptions to domain-specific exceptions
    5. Optionally reraises the exception

    Args:
        error_class: The BaseError subclass to use for wrapping exceptions
        log_level: Logging level to use
        include_traceback: Whether to include traceback in logs
        reraise: Whether to reraise the exception after handling
        correlation_id: Optional correlation ID for tracking
        context: Additional context information to include in logs
        cleanup_func: Optional function to call for cleanup on error

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
            context_dict = context.copy() if context else {}
            context_dict.update({
                "function": func.__name__,
                "module": func.__module__,
                "correlation_id": correlation_id,
                "timestamp": datetime.utcnow().isoformat()
            })

            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Run cleanup if provided
                if cleanup_func:
                    try:
                        cleanup_func()
                    except Exception as cleanup_error:
                        logger.error(
                            f"Error during cleanup: {str(cleanup_error)}",
                            extra={"original_error": str(e)}
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

    return decorator


def async_with_exception_handling(
    error_class: Type[BaseError] = ServiceError,
    log_level: int = logging.ERROR,
    include_traceback: bool = True,
    reraise: bool = True,
    correlation_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    cleanup_func: Optional[Union[Callable[[], None], Callable[[], Any]]] = None
) -> Callable[[F], F]:
    """
    Decorator to add standardized exception handling to asynchronous functions.

    This decorator:
    1. Catches all exceptions
    2. Logs the exception with appropriate context
    3. Optionally runs a cleanup function
    4. Converts generic exceptions to domain-specific exceptions
    5. Optionally reraises the exception

    Args:
        error_class: The BaseError subclass to use for wrapping exceptions
        log_level: Logging level to use
        include_traceback: Whether to include traceback in logs
        reraise: Whether to reraise the exception after handling
        correlation_id: Optional correlation ID for tracking
        context: Additional context information to include in logs
        cleanup_func: Optional function to call for cleanup on error

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
            context_dict = context.copy() if context else {}
            context_dict.update({
                "function": func.__name__,
                "module": func.__module__,
                "correlation_id": correlation_id,
                "timestamp": datetime.utcnow().isoformat()
            })

            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Run cleanup if provided
                if cleanup_func:
                    try:
                        if callable(getattr(cleanup_func, "__await__", None)):
                            await cleanup_func()
                        else:
                            cleanup_func()
                    except Exception as cleanup_error:
                        logger.error(
                            f"Error during cleanup: {str(cleanup_error)}",
                            extra={"original_error": str(e)}
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

    return decorator
