"""
Error handlers for FastAPI.
"""

from typing import Dict, Any, Callable
import uuid
import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from common_lib.exceptions import (
    ForexBaseException,
    ServiceException,
    DataException,
    TradingException,
    ConfigurationException,
    SecurityException,
    InfrastructureException
)

logger = logging.getLogger(__name__)


def generate_correlation_id() -> str:
    """
    Generate a correlation ID for error tracking.
    
    Returns:
        Correlation ID
    """
    return str(uuid.uuid4())


def get_correlation_id(request: Request) -> str:
    """
    Get the correlation ID from the request or generate a new one.
    
    Args:
        request: FastAPI request
        
    Returns:
        Correlation ID
    """
    if hasattr(request.state, 'correlation_id') and request.state.correlation_id:
        return request.state.correlation_id
    
    # Check if it's in the headers
    correlation_id = request.headers.get('X-Correlation-ID')
    if correlation_id:
        return correlation_id
    
    # Generate a new one
    return generate_correlation_id()


def setup_error_handlers(app):
    """
    Set up error handlers for the FastAPI application.
    
    Args:
        app: FastAPI application
    """
    
    @app.exception_handler(ForexBaseException)
    async def forex_exception_handler(request: Request, exc: ForexBaseException):
        """
        Handle ForexBaseException and its subclasses.
        
        Args:
            request: FastAPI request
            exc: Exception
            
        Returns:
            JSON response with error details
        """
        correlation_id = exc.correlation_id or get_correlation_id(request)
        
        # Log the error
        logger.error(
            f"ForexBaseException: {exc.message}",
            extra={
                'correlation_id': correlation_id,
                'exception_type': exc.__class__.__name__,
                'details': exc.details
            }
        )
        
        # Determine status code based on exception type
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        
        if isinstance(exc, ServiceException):
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif isinstance(exc, DataException):
            status_code = status.HTTP_400_BAD_REQUEST
        elif isinstance(exc, TradingException):
            status_code = status.HTTP_400_BAD_REQUEST
        elif isinstance(exc, ConfigurationException):
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        elif isinstance(exc, SecurityException):
            status_code = status.HTTP_401_UNAUTHORIZED
        elif isinstance(exc, InfrastructureException):
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        
        # Create response
        response = {
            'error': exc.__class__.__name__,
            'message': exc.message,
            'correlation_id': correlation_id
        }
        
        if exc.details:
            response['details'] = exc.details
        
        return JSONResponse(
            status_code=status_code,
            content=response,
            headers={'X-Correlation-ID': correlation_id}
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """
        Handle FastAPI validation errors.
        
        Args:
            request: FastAPI request
            exc: Exception
            
        Returns:
            JSON response with error details
        """
        correlation_id = get_correlation_id(request)
        
        # Log the error
        logger.error(
            f"RequestValidationError: {str(exc)}",
            extra={
                'correlation_id': correlation_id,
                'exception_type': 'RequestValidationError',
                'details': exc.errors()
            }
        )
        
        # Create response
        response = {
            'error': 'RequestValidationError',
            'message': 'Request validation failed',
            'correlation_id': correlation_id,
            'details': exc.errors()
        }
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=response,
            headers={'X-Correlation-ID': correlation_id}
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """
        Handle Starlette HTTP exceptions.
        
        Args:
            request: FastAPI request
            exc: Exception
            
        Returns:
            JSON response with error details
        """
        correlation_id = get_correlation_id(request)
        
        # Log the error
        logger.error(
            f"HTTPException: {exc.detail}",
            extra={
                'correlation_id': correlation_id,
                'exception_type': 'HTTPException',
                'status_code': exc.status_code
            }
        )
        
        # Create response
        response = {
            'error': 'HTTPException',
            'message': exc.detail,
            'correlation_id': correlation_id
        }
        
        return JSONResponse(
            status_code=exc.status_code,
            content=response,
            headers={'X-Correlation-ID': correlation_id}
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """
        Handle all other exceptions.
        
        Args:
            request: FastAPI request
            exc: Exception
            
        Returns:
            JSON response with error details
        """
        correlation_id = get_correlation_id(request)
        
        # Log the error
        logger.error(
            f"Unhandled exception: {str(exc)}",
            extra={
                'correlation_id': correlation_id,
                'exception_type': exc.__class__.__name__
            },
            exc_info=True
        )
        
        # Create response
        response = {
            'error': 'InternalServerError',
            'message': 'An unexpected error occurred',
            'correlation_id': correlation_id
        }
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response,
            headers={'X-Correlation-ID': correlation_id}
        )
