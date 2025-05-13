"""
Error handling package for the Trading Gateway Service.

This package provides error handling utilities and custom exceptions
for the Trading Gateway Service.
"""

from .exceptions_bridge import (
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
    
    # Trading Gateway specific exceptions
    BrokerConnectionError,
    OrderValidationError,
    MarketDataError,
    
    # Utility functions
    handle_exception,
    with_exception_handling,
    async_with_exception_handling,
    convert_js_error,
    convert_to_js_error
)

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
    
    # Trading Gateway specific exceptions
    "BrokerConnectionError",
    "OrderValidationError",
    "MarketDataError",
    
    # Utility functions
    "handle_exception",
    "with_exception_handling",
    "async_with_exception_handling",
    "convert_js_error",
    "convert_to_js_error"
]
