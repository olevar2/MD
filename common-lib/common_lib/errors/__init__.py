
"""
Error handling module for the forex trading platform.

This module provides a standardized error handling system for the platform.
It includes error classes, error handling decorators, and error utilities.
"""

from .base_exceptions import (
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

from .error_handler import ErrorHandler
from .decorators import with_exception_handling, async_with_exception_handling
from .middleware import FastAPIErrorMiddleware, fastapi_error_handler
from .api import create_error_response, fastapi_exception_handler

__all__ = [
    # Error codes and base classes
    'ErrorCode',
    'BaseError',

    # Error classes
    'ValidationError',
    'DatabaseError',
    'APIError',
    'ServiceError',
    'DataError',
    'BusinessError',
    'SecurityError',
    'ForexTradingError',
    'ServiceUnavailableError',
    'ThirdPartyServiceError',
    'TimeoutError',
    'AuthenticationError',
    'AuthorizationError',
    'NotFoundError',
    'ConflictError',
    'RateLimitError',

    # Error handling utilities
    'ErrorHandler',

    # Error handling decorators
    'with_exception_handling',
    'async_with_exception_handling',

    # Error handling middleware
    'FastAPIErrorMiddleware',
    'fastapi_error_handler',

    # API error handling
    'create_error_response',
    'fastapi_exception_handler'
]
