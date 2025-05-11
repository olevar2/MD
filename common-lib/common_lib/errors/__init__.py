
"""
Error handling module for the forex trading platform.

This module provides a standardized error handling system for the platform.
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

__all__ = [
    'ErrorCode',
    'BaseError',
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
    'ErrorHandler'
]
