"""
Exceptions Bridge Module for Trading Gateway Service

This module provides a bridge between the common-lib exceptions and the
Trading Gateway Service's error handling system. It ensures consistent
error handling across both Python and JavaScript components of the service.
"""

from typing import Dict, Any, Optional, Type, List, Callable, Union
import logging
import traceback
import json

# Import common-lib exceptions
from common_lib.exceptions import (
    # Base exception
    ForexTradingPlatformError,
    
    # Configuration exceptions
    ConfigurationError,
    
    # Data exceptions
    DataError,
    DataValidationError,
    DataFetchError,
    DataStorageError,
    DataTransformationError,
    
    # Service exceptions
    ServiceError,
    ServiceUnavailableError,
    ServiceTimeoutError,
    
    # Authentication/Authorization exceptions
    AuthenticationError,
    AuthorizationError,
    
    # Trading exceptions
    TradingError,
    OrderExecutionError,
    
    # Utility function
    get_all_exception_classes
)

# Re-export all imported exceptions
__all__ = [
    # Base exception
    "ForexTradingPlatformError",
    
    # Configuration exceptions
    "ConfigurationError",
    
    # Data exceptions
    "DataError",
    "DataValidationError",
    "DataFetchError",
    "DataStorageError",
    "DataTransformationError",
    
    # Service exceptions
    "ServiceError",
    "ServiceUnavailableError",
    "ServiceTimeoutError",
    
    # Authentication/Authorization exceptions
    "AuthenticationError",
    "AuthorizationError",
    
    # Trading exceptions
    "TradingError",
    "OrderExecutionError",
    
    # Utility function
    "get_all_exception_classes",
    
    # Bridge functions
    "handle_exception",
    "with_exception_handling",
    "async_with_exception_handling",
    "convert_js_error",
    "convert_to_js_error"
]

# Initialize logger
logger = logging.getLogger("trading_gateway_service.error")

# Trading Gateway specific exceptions
class BrokerConnectionError(ServiceUnavailableError):
    """Exception raised when connection to a broker fails."""
    
    def __init__(
        self,
        message: str = "Failed to connect to broker",
        broker_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
    """
      init  .
    
    Args:
        message: Description of message
        broker_name: Description of broker_name
        details: Description of details
        Any]]: Description of Any]]
    
    """

        error_details = details or {}
        if broker_name:
            error_details["broker_name"] = broker_name
        
        super().__init__(
            message=message,
            error_code="BROKER_CONNECTION_ERROR",
            details=error_details
        )

class OrderValidationError(DataValidationError):
    """Exception raised when an order fails validation."""
    
    def __init__(
        self,
        message: str = "Order validation failed",
        order_id: Optional[str] = None,
        validation_errors: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
    """
      init  .
    
    Args:
        message: Description of message
        order_id: Description of order_id
        validation_errors: Description of validation_errors
        details: Description of details
        Any]]: Description of Any]]
    
    """

        error_details = details or {}
        if order_id:
            error_details["order_id"] = order_id
        if validation_errors:
            error_details["validation_errors"] = validation_errors
        
        super().__init__(
            message=message,
            error_code="ORDER_VALIDATION_ERROR",
            details=error_details
        )

class MarketDataError(DataFetchError):
    """Exception raised when there's an issue with market data."""
    
    def __init__(
        self,
        message: str = "Failed to fetch market data",
        symbol: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
    """
      init  .
    
    Args:
        message: Description of message
        symbol: Description of symbol
        details: Description of details
        Any]]: Description of Any]]
    
    """

        error_details = details or {}
        if symbol:
            error_details["symbol"] = symbol
        
        super().__init__(
            message=message,
            error_code="MARKET_DATA_ERROR",
            details=error_details
        )

# Add Trading Gateway specific exceptions to __all__
__all__.extend([
    "BrokerConnectionError",
    "OrderValidationError",
    "MarketDataError"
])

def handle_exception(
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    reraise: bool = True,
    default_error_class: Type[ForexTradingPlatformError] = ServiceError
) -> Optional[ForexTradingPlatformError]:
    """
    Handle an exception by logging it and optionally wrapping it in a ForexTradingPlatformError.
    
    Args:
        exception: The exception to handle
        context: Additional context information to include in logs
        reraise: Whether to reraise the exception after handling
        default_error_class: Default error class to use if the exception is not a ForexTradingPlatformError
        
    Returns:
        The handled exception if reraise is False, otherwise None
        
    Raises:
        ForexTradingPlatformError: If reraise is True
    """
    context = context or {}
    
    if isinstance(exception, ForexTradingPlatformError):
        # It's already a ForexTradingPlatformError, just log it
        logger.error(
            f"{exception.__class__.__name__}: {exception.message}",
            extra={
                "error_code": exception.error_code,
                "details": exception.details,
                **context
            }
        )
        
        if reraise:
            raise exception
        return exception
    else:
        # Convert to ForexTradingPlatformError
        error_message = str(exception)
        error_type = exception.__class__.__name__
        error_context = {
            "original_error": error_type,
            "traceback": traceback.format_exc(),
            **context
        }
        
        # Create a new ForexTradingPlatformError
        forex_error = default_error_class(
            message=f"{error_type}: {error_message}",
            details=error_context
        )
        
        # Log the error
        logger.error(
            f"{forex_error.__class__.__name__}: {forex_error.message}",
            extra={
                "error_code": forex_error.error_code,
                "details": forex_error.details
            }
        )
        
        if reraise:
            raise forex_error
        return forex_error

def with_exception_handling(
    func: Callable,
    error_class: Type[ForexTradingPlatformError] = ServiceError,
    context: Optional[Dict[str, Any]] = None,
    cleanup_func: Optional[Callable] = None
) -> Callable:
    """
    Decorator to handle exceptions in a function.
    
    Args:
        func: The function to wrap
        error_class: The error class to use for wrapping exceptions
        context: Additional context information to include in logs
        cleanup_func: Optional function to call for cleanup on error
        
    Returns:
        Wrapped function with exception handling
    """
    def wrapper(*args, **kwargs):
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

        try:
            return func(*args, **kwargs)
        except Exception as e:
            context_dict = context.copy() if context else {}
            context_dict.update({
                "function": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs)
            })
            
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
            return handle_exception(
                exception=e,
                context=context_dict,
                reraise=True,
                default_error_class=error_class
            )
    
    return wrapper

async def async_with_exception_handling(
    func: Callable,
    error_class: Type[ForexTradingPlatformError] = ServiceError,
    context: Optional[Dict[str, Any]] = None,
    cleanup_func: Optional[Callable] = None
) -> Callable:
    """
    Decorator to handle exceptions in an async function.
    
    Args:
        func: The async function to wrap
        error_class: The error class to use for wrapping exceptions
        context: Additional context information to include in logs
        cleanup_func: Optional function to call for cleanup on error
        
    Returns:
        Wrapped async function with exception handling
    """
    async def wrapper(*args, **kwargs):
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

        try:
            return await func(*args, **kwargs)
        except Exception as e:
            context_dict = context.copy() if context else {}
            context_dict.update({
                "function": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs)
            })
            
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
            return handle_exception(
                exception=e,
                context=context_dict,
                reraise=True,
                default_error_class=error_class
            )
    
    return wrapper

def convert_js_error(error_json: Union[str, Dict[str, Any]]) -> ForexTradingPlatformError:
    """
    Convert a JavaScript error (as JSON) to a Python ForexTradingPlatformError.
    
    Args:
        error_json: JSON string or dictionary representing the error
        
    Returns:
        Appropriate ForexTradingPlatformError instance
    """
    if isinstance(error_json, str):
        try:
            error_data = json.loads(error_json)
        except json.JSONDecodeError:
            # If it's not valid JSON, treat it as a simple error message
            return ServiceError(message=error_json)
    else:
        error_data = error_json
    
    # Extract error information
    error_type = error_data.get("error_type", "ForexTradingPlatformError")
    error_code = error_data.get("error_code", "UNKNOWN_ERROR")
    message = error_data.get("message", "Unknown error")
    details = error_data.get("details", {})
    
    # Map JavaScript error types to Python exception classes
    error_map = {cls.__name__: cls for cls in get_all_exception_classes()}
    
    # Add Trading Gateway specific exceptions
    error_map.update({
        "BrokerConnectionError": BrokerConnectionError,
        "OrderValidationError": OrderValidationError,
        "MarketDataError": MarketDataError
    })
    
    # Get the appropriate exception class
    exception_class = error_map.get(error_type, ForexTradingPlatformError)
    
    # Create and return the exception
    return exception_class(
        message=message,
        error_code=error_code,
        details=details
    )

def convert_to_js_error(exception: Exception) -> Dict[str, Any]:
    """
    Convert a Python exception to a format that can be used in JavaScript.
    
    Args:
        exception: The exception to convert
        
    Returns:
        Dictionary representing the error in a format compatible with JavaScript
    """
    if isinstance(exception, ForexTradingPlatformError):
        return {
            "error_type": exception.__class__.__name__,
            "error_code": exception.error_code,
            "message": exception.message,
            "details": exception.details
        }
    else:
        # Convert standard Python exception
        return {
            "error_type": "ServiceError",
            "error_code": "PYTHON_EXCEPTION",
            "message": str(exception),
            "details": {
                "original_error": exception.__class__.__name__,
                "traceback": traceback.format_exc()
            }
        }
