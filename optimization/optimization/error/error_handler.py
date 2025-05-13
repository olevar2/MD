"""
Error handling utilities for the Optimization module.

This module provides functions for handling errors in a consistent way
throughout the Optimization module.
"""

import logging
import traceback
from typing import Any, Callable, Dict, Optional, Type, TypeVar, cast

from .exceptions import (
    OptimizationError,
    ParameterValidationError,
    OptimizationConvergenceError,
    ResourceAllocationError,
    CachingError,
    MLOptimizationError
)

# Configure logger
logger = logging.getLogger("optimization.error_handler")

# Type variable for the return type of the wrapped function
T = TypeVar("T")


def handle_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    reraise: bool = True
) -> None:
    """
    Handle an error in a consistent way.
    
    Args:
        error: The exception to handle
        context: Additional context information
        reraise: Whether to reraise the exception after handling
        
    Raises:
        OptimizationError: If reraise is True and the error is not already an OptimizationError
    """
    context = context or {}
    
    # Convert to OptimizationError if it's not already one
    if not isinstance(error, OptimizationError):
        error_message = str(error)
        error_type = error.__class__.__name__
        
        # Create context with traceback
        error_context = {
            "error_type": error_type,
            "traceback": traceback.format_exc(),
            **context
        }
        
        # Log the error
        logger.error(
            f"Error occurred: {error_message}",
            extra=error_context
        )
        
        # Wrap in OptimizationError if reraising
        if reraise:
            raise OptimizationError(
                message=f"{error_type}: {error_message}",
                details=error_context
            )
    else:
        # It's already an OptimizationError, just log and reraise if needed
        logger.error(
            f"{error.__class__.__name__}: {error.message}",
            extra={
                "error_code": error.error_code,
                "details": error.details,
                **context
            }
        )
        
        if reraise:
            raise


def with_error_handling(
    error_class: Type[OptimizationError] = OptimizationError,
    reraise: bool = True,
    cleanup_func: Optional[Callable[[], None]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to add error handling to a function.
    
    Args:
        error_class: The OptimizationError subclass to use for wrapping errors
        reraise: Whether to reraise the exception after handling
        cleanup_func: Optional function to call for cleanup on error
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator.
    
    Args:
        func: Description of func
        T]: Description of T]
    
    Returns:
        Callable[..., T]: Description of return value
    
    """

        def wrapper(*args: Any, **kwargs: Any) -> T:
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        T: Description of return value
    
    """

            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Create context with function information
                context = {
                    "function": func.__name__,
                    "args": str(args),
                    "kwargs": str(kwargs)
                }
                
                # Handle the error
                try:
                    if not isinstance(e, OptimizationError):
                        error_message = str(e)
                        error_type = e.__class__.__name__
                        
                        # Wrap in specified error class
                        e = error_class(
                            message=f"{error_type}: {error_message}",
                            details={"original_error": error_type, "traceback": traceback.format_exc()}
                        )
                    
                    # Run cleanup if provided
                    if cleanup_func:
                        try:
                            cleanup_func()
                        except Exception as cleanup_error:
                            logger.error(
                                f"Error during cleanup: {str(cleanup_error)}",
                                extra={"original_error": str(e)}
                            )
                            # Don't wrap cleanup error, just log it
                    
                    # Log and reraise if needed
                    logger.error(
                        f"{e.__class__.__name__}: {getattr(e, 'message', str(e))}",
                        extra={
                            "error_code": getattr(e, "error_code", "UNKNOWN"),
                            "details": getattr(e, "details", {}),
                            **context
                        }
                    )
                    
                    if reraise:
                        raise e
                    
                    # Return a default value if not reraising
                    # This is a bit of a hack, but it allows the function to "continue"
                    # with a default value when reraise is False
                    return cast(T, None)
                except Exception as wrapped_error:
                    # If error handling itself fails, log and raise
                    logger.critical(
                        f"Error handling failed: {str(wrapped_error)}",
                        extra={"original_error": str(e)}
                    )
                    raise
        
        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__
        
        return wrapper
    
    return decorator
