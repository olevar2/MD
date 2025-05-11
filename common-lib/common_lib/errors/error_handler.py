
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
            log_message += f"\n{''.join(traceback.format_exception(type(error), error, error.__traceback__))}"
        
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
