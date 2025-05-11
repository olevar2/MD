"""
Error Handler Module

This module provides error handling functionality for the platform.
"""

import logging
import traceback
import sys
from typing import Dict, Any, Optional, List, Type, Union

from pydantic import BaseModel

from common_lib.errors.base import BaseError, ErrorCode


class ErrorContext(BaseModel):
    """
    Context for error handling.
    """
    
    correlation_id: Optional[str] = None
    service_name: Optional[str] = None
    operation: Optional[str] = None
    request_id: Optional[str] = None
    request_path: Optional[str] = None
    request_method: Optional[str] = None
    request_params: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None


class ErrorResponse(BaseModel):
    """
    Error response.
    """
    
    code: int
    message: str
    details: Optional[str] = None
    correlation_id: Optional[str] = None
    service: Optional[str] = None
    operation: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the error response to a dictionary.
        
        Returns:
            Dictionary representation of the error response
        """
        result = {
            "code": self.code,
            "message": self.message
        }
        
        if self.details:
            result["details"] = self.details
        
        if self.correlation_id:
            result["correlation_id"] = self.correlation_id
        
        if self.service:
            result["service"] = self.service
        
        if self.operation:
            result["operation"] = self.operation
        
        if self.data:
            result["data"] = self.data
        
        return result


class ErrorHandler:
    """
    Error handler for the platform.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the error handler.
        
        Args:
            logger: Logger to use (if None, creates a new logger)
        """
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None
    ) -> ErrorResponse:
        """
        Handle an error.
        
        Args:
            error: Error to handle
            context: Error context
            
        Returns:
            Error response
        """
        # Get error details
        if isinstance(error, BaseError):
            # Use error code and message from the error
            code = error.code
            message = error.message
            details = error.details
            data = error.data
        else:
            # Use generic error code and message
            code = ErrorCode.UNKNOWN_ERROR.value
            message = str(error)
            details = None
            data = None
        
        # Get stack trace
        stack_trace = traceback.format_exception(*sys.exc_info())
        
        # Log error
        self.logger.error(
            f"Error: {message} (code={code})"
        )
        self.logger.debug(
            f"Stack trace: {''.join(stack_trace)}"
        )
        
        # Create error response
        response = ErrorResponse(
            code=code,
            message=message,
            details=details,
            correlation_id=getattr(context, "correlation_id", None),
            service=getattr(context, "service_name", None),
            operation=getattr(context, "operation", None),
            data=data
        )
        
        return response