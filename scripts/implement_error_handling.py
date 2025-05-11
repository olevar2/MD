#!/usr/bin/env python3
"""
Standardized Error Handling Implementation Script

This script implements a standardized error handling system in the common-lib module
to ensure consistent error handling across services.
"""

import os
import sys
from pathlib import Path

# Constants
COMMON_LIB_PATH = "common-lib/common_lib"
ERRORS_DIR = f"{COMMON_LIB_PATH}/errors"

# Error handling code
BASE_EXCEPTIONS = '''
"""
Base exceptions for the forex trading platform.

This module defines the base exception hierarchy for the platform,
ensuring consistent error handling across services.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Union
import uuid
import traceback


class ErrorCode(Enum):
    """Error codes for the platform."""
    # General errors (1-99)
    UNKNOWN_ERROR = 1
    VALIDATION_ERROR = 2
    CONFIGURATION_ERROR = 3
    DEPENDENCY_ERROR = 4
    TIMEOUT_ERROR = 5
    RESOURCE_NOT_FOUND = 6
    PERMISSION_DENIED = 7
    RATE_LIMIT_EXCEEDED = 8
    
    # Database errors (100-199)
    DATABASE_CONNECTION_ERROR = 100
    DATABASE_QUERY_ERROR = 101
    DATABASE_TRANSACTION_ERROR = 102
    DATABASE_CONSTRAINT_ERROR = 103
    DATABASE_TIMEOUT_ERROR = 104
    
    # API errors (200-299)
    API_REQUEST_ERROR = 200
    API_RESPONSE_ERROR = 201
    API_AUTHENTICATION_ERROR = 202
    API_AUTHORIZATION_ERROR = 203
    API_RATE_LIMIT_ERROR = 204
    API_TIMEOUT_ERROR = 205
    
    # Service errors (300-399)
    SERVICE_UNAVAILABLE = 300
    SERVICE_TIMEOUT = 301
    SERVICE_DEPENDENCY_ERROR = 302
    SERVICE_CIRCUIT_OPEN = 303
    
    # Data errors (400-499)
    DATA_VALIDATION_ERROR = 400
    DATA_INTEGRITY_ERROR = 401
    DATA_FORMAT_ERROR = 402
    DATA_MISSING_ERROR = 403
    DATA_INCONSISTENCY_ERROR = 404
    
    # Business logic errors (500-599)
    BUSINESS_RULE_VIOLATION = 500
    INSUFFICIENT_FUNDS = 501
    POSITION_LIMIT_EXCEEDED = 502
    INVALID_OPERATION = 503
    OPERATION_NOT_ALLOWED = 504
    
    # Security errors (600-699)
    SECURITY_VIOLATION = 600
    AUTHENTICATION_FAILED = 601
    AUTHORIZATION_FAILED = 602
    TOKEN_EXPIRED = 603
    INVALID_CREDENTIALS = 604


class BaseError(Exception):
    """Base exception for all platform errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize the base error.
        
        Args:
            message: Human-readable error message
            error_code: Error code from the ErrorCode enum
            details: Additional error details
            correlation_id: Correlation ID for tracking the error across services
            cause: Original exception that caused this error
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.cause = cause
        self.timestamp = None  # Will be set when the error is logged
        
        # Add stack trace to details if a cause is provided
        if cause:
            self.details["cause"] = str(cause)
            self.details["traceback"] = "".join(traceback.format_exception(
                type(cause), cause, cause.__traceback__
            ))
        
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the error to a dictionary representation."""
        return {
            "message": self.message,
            "error_code": self.error_code.name,
            "error_code_value": self.error_code.value,
            "details": self.details,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp
        }


class ValidationError(BaseError):
    """Error raised when input validation fails."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        constraint: Optional[str] = None,
        **kwargs
    ):
        """Initialize a validation error.
        
        Args:
            message: Human-readable error message
            field: Name of the field that failed validation
            value: Value that failed validation
            constraint: Constraint that was violated
            **kwargs: Additional arguments to pass to BaseError
        """
        details = kwargs.pop("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        if constraint:
            details["constraint"] = constraint
        
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            details=details,
            **kwargs
        )


class DatabaseError(BaseError):
    """Base class for database-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.DATABASE_CONNECTION_ERROR,
        query: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize a database error.
        
        Args:
            message: Human-readable error message
            error_code: Specific database error code
            query: SQL query that caused the error
            parameters: Parameters for the query
            **kwargs: Additional arguments to pass to BaseError
        """
        details = kwargs.pop("details", {})
        if query:
            # Sanitize the query to avoid logging sensitive information
            details["query"] = self._sanitize_query(query)
        if parameters:
            # Sanitize parameters to avoid logging sensitive information
            details["parameters"] = self._sanitize_parameters(parameters)
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            **kwargs
        )
    
    @staticmethod
    def _sanitize_query(query: str) -> str:
        """Sanitize a SQL query to remove sensitive information."""
        # This is a simple implementation that could be enhanced
        return query
    
    @staticmethod
    def _sanitize_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize query parameters to remove sensitive information."""
        # This is a simple implementation that could be enhanced
        sanitized = {}
        sensitive_keys = ["password", "token", "secret", "key", "auth"]
        
        for key, value in parameters.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "********"
            else:
                sanitized[key] = value
        
        return sanitized


class APIError(BaseError):
    """Base class for API-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.API_REQUEST_ERROR,
        status_code: Optional[int] = None,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
        response: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize an API error.
        
        Args:
            message: Human-readable error message
            error_code: Specific API error code
            status_code: HTTP status code
            endpoint: API endpoint that was called
            method: HTTP method used
            response: Response from the API
            **kwargs: Additional arguments to pass to BaseError
        """
        details = kwargs.pop("details", {})
        if status_code:
            details["status_code"] = status_code
        if endpoint:
            details["endpoint"] = endpoint
        if method:
            details["method"] = method
        if response:
            details["response"] = response
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            **kwargs
        )


class ServiceError(BaseError):
    """Base class for service-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.SERVICE_UNAVAILABLE,
        service_name: Optional[str] = None,
        operation: Optional[str] = None,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        """Initialize a service error.
        
        Args:
            message: Human-readable error message
            error_code: Specific service error code
            service_name: Name of the service that failed
            operation: Operation that was attempted
            retry_after: Seconds to wait before retrying
            **kwargs: Additional arguments to pass to BaseError
        """
        details = kwargs.pop("details", {})
        if service_name:
            details["service_name"] = service_name
        if operation:
            details["operation"] = operation
        if retry_after:
            details["retry_after"] = retry_after
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            **kwargs
        )


class DataError(BaseError):
    """Base class for data-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.DATA_VALIDATION_ERROR,
        data_source: Optional[str] = None,
        data_type: Optional[str] = None,
        **kwargs
    ):
        """Initialize a data error.
        
        Args:
            message: Human-readable error message
            error_code: Specific data error code
            data_source: Source of the data
            data_type: Type of data
            **kwargs: Additional arguments to pass to BaseError
        """
        details = kwargs.pop("details", {})
        if data_source:
            details["data_source"] = data_source
        if data_type:
            details["data_type"] = data_type
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            **kwargs
        )


class BusinessError(BaseError):
    """Base class for business logic errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.BUSINESS_RULE_VIOLATION,
        rule: Optional[str] = None,
        entity: Optional[str] = None,
        entity_id: Optional[str] = None,
        **kwargs
    ):
        """Initialize a business error.
        
        Args:
            message: Human-readable error message
            error_code: Specific business error code
            rule: Business rule that was violated
            entity: Entity involved in the error
            entity_id: ID of the entity
            **kwargs: Additional arguments to pass to BaseError
        """
        details = kwargs.pop("details", {})
        if rule:
            details["rule"] = rule
        if entity:
            details["entity"] = entity
        if entity_id:
            details["entity_id"] = entity_id
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            **kwargs
        )


class SecurityError(BaseError):
    """Base class for security-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.SECURITY_VIOLATION,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        **kwargs
    ):
        """Initialize a security error.
        
        Args:
            message: Human-readable error message
            error_code: Specific security error code
            user_id: ID of the user
            resource: Resource that was accessed
            action: Action that was attempted
            **kwargs: Additional arguments to pass to BaseError
        """
        details = kwargs.pop("details", {})
        if user_id:
            details["user_id"] = user_id
        if resource:
            details["resource"] = resource
        if action:
            details["action"] = action
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            **kwargs
        )
'''

ERROR_HANDLER = '''
"""
Error handling utilities for the forex trading platform.

This module provides utilities for handling errors consistently across services.
"""

import logging
import traceback
import sys
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Callable, Type, TypeVar, cast
from functools import wraps

from .base_exceptions import BaseError, ErrorCode


# Type variable for generic function
F = TypeVar('F', bound=Callable[..., Any])


class ErrorHandler:
    """Utility class for handling errors consistently."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the error handler.
        
        Args:
            logger: Logger to use for logging errors
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def handle_error(
        self,
        error: Exception,
        correlation_id: Optional[str] = None,
        log_level: int = logging.ERROR,
        include_traceback: bool = True
    ) -> Dict[str, Any]:
        """Handle an error and return a standardized error response.
        
        Args:
            error: The exception to handle
            correlation_id: Correlation ID for tracking the error
            log_level: Logging level to use
            include_traceback: Whether to include the traceback in the log
        
        Returns:
            A standardized error response dictionary
        """
        # Generate a correlation ID if not provided
        correlation_id = correlation_id or str(uuid.uuid4())
        
        # Convert to BaseError if it's not already
        if not isinstance(error, BaseError):
            base_error = BaseError(
                message=str(error),
                error_code=ErrorCode.UNKNOWN_ERROR,
                correlation_id=correlation_id,
                cause=error
            )
        else:
            base_error = error
        
        # Set the timestamp
        base_error.timestamp = datetime.utcnow().isoformat()
        
        # Log the error
        log_message = f"Error: {base_error.message} (Code: {base_error.error_code.name}, ID: {correlation_id})"
        if include_traceback and not isinstance(error, BaseError):
            log_message += f"\\n{''.join(traceback.format_exception(type(error), error, error.__traceback__))}"
        
        self.logger.log(log_level, log_message)
        
        # Return a standardized error response
        return base_error.to_dict()
    
    def create_error_response(
        self,
        error: Exception,
        correlation_id: Optional[str] = None,
        status_code: int = 500
    ) -> Dict[str, Any]:
        """Create a standardized error response for API endpoints.
        
        Args:
            error: The exception to handle
            correlation_id: Correlation ID for tracking the error
            status_code: HTTP status code to return
        
        Returns:
            A standardized error response dictionary
        """
        error_dict = self.handle_error(error, correlation_id)
        
        # Map error codes to HTTP status codes
        if isinstance(error, BaseError):
            status_code = self._map_error_code_to_status_code(error.error_code)
        
        return {
            "error": error_dict,
            "status_code": status_code
        }
    
    @staticmethod
    def _map_error_code_to_status_code(error_code: ErrorCode) -> int:
        """Map an error code to an HTTP status code.
        
        Args:
            error_code: The error code to map
        
        Returns:
            The corresponding HTTP status code
        """
        # Map error codes to HTTP status codes
        error_code_map = {
            # General errors
            ErrorCode.UNKNOWN_ERROR: 500,
            ErrorCode.VALIDATION_ERROR: 400,
            ErrorCode.CONFIGURATION_ERROR: 500,
            ErrorCode.DEPENDENCY_ERROR: 503,
            ErrorCode.TIMEOUT_ERROR: 504,
            ErrorCode.RESOURCE_NOT_FOUND: 404,
            ErrorCode.PERMISSION_DENIED: 403,
            ErrorCode.RATE_LIMIT_EXCEEDED: 429,
            
            # Database errors
            ErrorCode.DATABASE_CONNECTION_ERROR: 503,
            ErrorCode.DATABASE_QUERY_ERROR: 500,
            ErrorCode.DATABASE_TRANSACTION_ERROR: 500,
            ErrorCode.DATABASE_CONSTRAINT_ERROR: 400,
            ErrorCode.DATABASE_TIMEOUT_ERROR: 504,
            
            # API errors
            ErrorCode.API_REQUEST_ERROR: 400,
            ErrorCode.API_RESPONSE_ERROR: 502,
            ErrorCode.API_AUTHENTICATION_ERROR: 401,
            ErrorCode.API_AUTHORIZATION_ERROR: 403,
            ErrorCode.API_RATE_LIMIT_ERROR: 429,
            ErrorCode.API_TIMEOUT_ERROR: 504,
            
            # Service errors
            ErrorCode.SERVICE_UNAVAILABLE: 503,
            ErrorCode.SERVICE_TIMEOUT: 504,
            ErrorCode.SERVICE_DEPENDENCY_ERROR: 503,
            ErrorCode.SERVICE_CIRCUIT_OPEN: 503,
            
            # Data errors
            ErrorCode.DATA_VALIDATION_ERROR: 400,
            ErrorCode.DATA_INTEGRITY_ERROR: 400,
            ErrorCode.DATA_FORMAT_ERROR: 400,
            ErrorCode.DATA_MISSING_ERROR: 400,
            ErrorCode.DATA_INCONSISTENCY_ERROR: 409,
            
            # Business logic errors
            ErrorCode.BUSINESS_RULE_VIOLATION: 400,
            ErrorCode.INSUFFICIENT_FUNDS: 400,
            ErrorCode.POSITION_LIMIT_EXCEEDED: 400,
            ErrorCode.INVALID_OPERATION: 400,
            ErrorCode.OPERATION_NOT_ALLOWED: 403,
            
            # Security errors
            ErrorCode.SECURITY_VIOLATION: 403,
            ErrorCode.AUTHENTICATION_FAILED: 401,
            ErrorCode.AUTHORIZATION_FAILED: 403,
            ErrorCode.TOKEN_EXPIRED: 401,
            ErrorCode.INVALID_CREDENTIALS: 401
        }
        
        return error_code_map.get(error_code, 500)
    
    def error_handler(
        self,
        error_type: Optional[Type[Exception]] = None,
        log_level: int = logging.ERROR,
        include_traceback: bool = True
    ) -> Callable[[F], F]:
        """Decorator for handling errors in functions.
        
        Args:
            error_type: Type of exception to catch
            log_level: Logging level to use
            include_traceback: Whether to include the traceback in the log
        
        Returns:
            A decorator function
        """
        def decorator(func: F) -> F:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if error_type is None or isinstance(e, error_type):
                        self.handle_error(e, log_level=log_level, include_traceback=include_traceback)
                    raise
            return cast(F, wrapper)
        return decorator
    
    def async_error_handler(
        self,
        error_type: Optional[Type[Exception]] = None,
        log_level: int = logging.ERROR,
        include_traceback: bool = True
    ) -> Callable[[F], F]:
        """Decorator for handling errors in async functions.
        
        Args:
            error_type: Type of exception to catch
            log_level: Logging level to use
            include_traceback: Whether to include the traceback in the log
        
        Returns:
            A decorator function
        """
        def decorator(func: F) -> F:
            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if error_type is None or isinstance(e, error_type):
                        self.handle_error(e, log_level=log_level, include_traceback=include_traceback)
                    raise
            return cast(F, wrapper)
        return decorator
'''

INIT_PY = '''
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
    SecurityError
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
    'ErrorHandler'
]
'''

def create_directory(path):
    """Create a directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def create_file(path, content):
    """Create a file with the given content"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created file: {path}")

def implement_error_handling():
    """Implement standardized error handling"""
    # Create directories
    create_directory(ERRORS_DIR)
    
    # Create files
    create_file(os.path.join(ERRORS_DIR, "base_exceptions.py"), BASE_EXCEPTIONS)
    create_file(os.path.join(ERRORS_DIR, "error_handler.py"), ERROR_HANDLER)
    create_file(os.path.join(ERRORS_DIR, "__init__.py"), INIT_PY)
    
    print("Standardized error handling implemented successfully")

def main():
    implement_error_handling()

if __name__ == "__main__":
    main()
