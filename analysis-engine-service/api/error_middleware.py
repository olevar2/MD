"""
Error Handling Middleware for Analysis Engine Service.

This module provides middleware for handling errors in API endpoints.
It ensures consistent error handling across the service.
"""

import logging
import traceback
from typing import Dict, Any, Optional, Callable, Union

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException

from analysis_engine.core.exceptions_bridge import (
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    generate_correlation_id,
    get_correlation_id_from_request
)

# Initialize logger
logger = logging.getLogger("analysis_engine")


def create_error_response(
    error: Exception,
    correlation_id: Optional[str] = None,
    include_traceback: bool = False
) -> Dict[str, Any]:
    """
    Create a standardized error response.
    
    Args:
        error: The error to handle
        correlation_id: Optional correlation ID for tracking
        include_traceback: Whether to include traceback in the response
        
    Returns:
        Standardized error response
    """
    # Generate a correlation ID if not provided
    correlation_id = correlation_id or generate_correlation_id()
    
    # Handle ForexTradingPlatformError
    if isinstance(error, ForexTradingPlatformError):
        response = {
            "error": {
                "code": error.error_code,
                "message": error.message,
                "correlation_id": getattr(error, "correlation_id", correlation_id),
                "details": error.details or {}
            }
        }
    # Handle FastAPI RequestValidationError
    elif isinstance(error, RequestValidationError):
        response = {
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation error",
                "correlation_id": correlation_id,
                "details": {
                    "errors": error.errors()
                }
            }
        }
    # Handle FastAPI HTTPException
    elif isinstance(error, HTTPException):
        response = {
            "error": {
                "code": "HTTP_ERROR",
                "message": error.detail,
                "correlation_id": correlation_id,
                "details": {
                    "status_code": error.status_code
                }
            }
        }
    # Handle other exceptions
    else:
        response = {
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": str(error),
                "correlation_id": correlation_id,
                "details": {}
            }
        }
    
    # Include traceback if requested
    if include_traceback:
        response["error"]["details"]["traceback"] = traceback.format_exc()
    
    return response


async def exception_handler(
    request: Request,
    error: Exception
) -> JSONResponse:
    """
    Handle exceptions and return a standardized error response.
    
    Args:
        request: FastAPI request
        error: The error to handle
        
    Returns:
        JSONResponse with standardized error format
    """
    # Get correlation ID from request or generate a new one
    correlation_id = get_correlation_id_from_request(request)
    
    # Determine status code based on error type
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    if isinstance(error, ForexTradingPlatformError):
        # Use status code from error if available
        status_code = getattr(error, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Map error types to status codes
        if isinstance(error, ValidationError):
            status_code = status.HTTP_400_BAD_REQUEST
        elif isinstance(error, AuthenticationError):
            status_code = status.HTTP_401_UNAUTHORIZED
        elif isinstance(error, AuthorizationError):
            status_code = status.HTTP_403_FORBIDDEN
        elif isinstance(error, NotFoundError):
            status_code = status.HTTP_404_NOT_FOUND
        elif isinstance(error, ServiceError):
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif isinstance(error, RequestValidationError):
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    elif isinstance(error, HTTPException):
        status_code = error.status_code
    
    # Create error response
    response = create_error_response(
        error=error,
        correlation_id=correlation_id,
        include_traceback=False  # Don't include traceback in production
    )
    
    # Log the error
    log_level = logging.ERROR
    if status_code >= 500:
        log_level = logging.ERROR
    elif status_code >= 400:
        log_level = logging.WARNING
    
    logger.log(
        log_level,
        f"Error handling request: {str(error)}",
        extra={
            "correlation_id": correlation_id,
            "status_code": status_code,
            "error_type": error.__class__.__name__,
            "path": request.url.path,
            "method": request.method
        },
        exc_info=True
    )
    
    # Return the response
    return JSONResponse(
        status_code=status_code,
        content=response
    )


def setup_error_handlers(app: FastAPI) -> None:
    """
    Set up error handlers for a FastAPI application.
    
    Args:
        app: FastAPI application
    """
    # Handle ForexTradingPlatformError
    @app.exception_handler(ForexTradingPlatformError)
    async def handle_forex_trading_platform_error(request: Request, error: ForexTradingPlatformError):
    """
    Handle forex trading platform error.
    
    Args:
        request: Description of request
        error: Description of error
    
    """

        return await exception_handler(request, error)
    
    # Handle RequestValidationError
    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, error: RequestValidationError):
    """
    Handle validation error.
    
    Args:
        request: Description of request
        error: Description of error
    
    """

        return await exception_handler(request, error)
    
    # Handle HTTPException
    @app.exception_handler(HTTPException)
    async def handle_http_exception(request: Request, error: HTTPException):
    """
    Handle http exception.
    
    Args:
        request: Description of request
        error: Description of error
    
    """

        return await exception_handler(request, error)
    
    # Handle generic Exception
    @app.exception_handler(Exception)
    async def handle_generic_exception(request: Request, error: Exception):
        return await exception_handler(request, error)


class ErrorHandlingMiddleware:
    """
    Middleware for handling errors in FastAPI applications.
    
    This middleware ensures that all errors are handled consistently,
    with proper logging and correlation ID tracking.
    """
    
    def __init__(
        self,
        app: FastAPI,
        include_traceback: bool = False
    ):
        """
        Initialize the middleware.
        
        Args:
            app: FastAPI application
            include_traceback: Whether to include traceback in error responses
        """
        self.app = app
        self.include_traceback = include_traceback
        
        # Set up error handlers
        setup_error_handlers(app)
    
    async def __call__(self, scope, receive, send):
        """
        Process a request.
        
        Args:
            scope: ASGI scope
            receive: ASGI receive function
            send: ASGI send function
        """
        # Add correlation ID to the scope
        if scope["type"] == "http":
            # Generate a correlation ID
            correlation_id = generate_correlation_id()
            
            # Add it to the scope
            scope["correlation_id"] = correlation_id
        
        # Process the request
        await self.app(scope, receive, send)


def add_error_handling_middleware(
    app: FastAPI,
    include_traceback: bool = False
) -> None:
    """
    Add error handling middleware to a FastAPI application.
    
    Args:
        app: FastAPI application
        include_traceback: Whether to include traceback in error responses
    """
    # Add the middleware
    app.add_middleware(
        ErrorHandlingMiddleware,
        include_traceback=include_traceback
    )
    
    # Set up error handlers
    setup_error_handlers(app)
