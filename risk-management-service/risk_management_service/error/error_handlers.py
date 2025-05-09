"""
Error Handlers Module

This module provides FastAPI exception handlers for standardized error responses.
"""
from typing import Dict, Any, List, Union
import uuid
from datetime import datetime
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from fastapi.responses import JSONResponse
import traceback

from common_lib.exceptions import (
    ForexTradingPlatformError,
    DataError,
    ServiceError,
    ModelError,
    SecurityError,
    ResilienceError
)

from risk_management_service.error.exceptions_bridge import (
    RiskManagementError,
    RiskCalculationError,
    RiskParameterError,
    CircuitBreakerError,
    RiskLimitBreachError,
    RiskProfileNotFoundError
)

from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


def get_correlation_id(request: Request) -> str:
    """
    Get correlation ID from request or generate a new one.
    
    Args:
        request: The FastAPI request
        
    Returns:
        Correlation ID string
    """
    # Try to get from request state (set by middleware)
    correlation_id = getattr(request.state, "correlation_id", None)
    
    # If not in state, try to get from headers
    if not correlation_id:
        correlation_id = request.headers.get("X-Correlation-ID")
    
    # If still not found, generate a new one
    if not correlation_id:
        correlation_id = str(uuid.uuid4())
    
    return correlation_id


def format_error_response(
    error_code: str,
    message: str,
    details: Dict[str, Any] = None,
    correlation_id: str = None,
    service: str = "risk-management-service"
) -> Dict[str, Any]:
    """
    Format a standardized error response.
    
    Args:
        error_code: Error code
        message: Error message
        details: Additional error details
        correlation_id: Request correlation ID
        service: Service name
        
    Returns:
        Formatted error response dictionary
    """
    return {
        "error": {
            "code": error_code,
            "message": message,
            "details": details or {},
            "correlation_id": correlation_id or str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "service": service
        }
    }


def format_validation_errors(errors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Format validation errors into a structured format.
    
    Args:
        errors: List of validation errors
        
    Returns:
        Formatted validation errors
    """
    formatted_errors = {}
    
    for error in errors:
        # Get the field name from the location
        loc = error.get("loc", [])
        field = ".".join(str(item) for item in loc) if loc else "request"
        
        # Get the error message
        msg = error.get("msg", "Validation error")
        
        # Add to formatted errors
        formatted_errors[field] = msg
    
    return formatted_errors


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register all exception handlers with the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    # Handle ForexTradingPlatformError (base exception for all platform errors)
    @app.exception_handler(ForexTradingPlatformError)
    async def forex_platform_exception_handler(request: Request, exc: ForexTradingPlatformError):
        """Handle custom ForexTradingPlatformError exceptions."""
        correlation_id = get_correlation_id(request)
        
        logger.error(
            f"ForexTradingPlatformError: {exc.message}",
            extra={
                "error_code": exc.error_code,
                "details": exc.details,
                "path": request.url.path,
                "method": request.method,
                "correlation_id": correlation_id
            },
        )

        # Determine status code based on error type
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        
        if isinstance(exc, DataError):
            status_code = status.HTTP_400_BAD_REQUEST
        elif isinstance(exc, SecurityError):
            status_code = status.HTTP_401_UNAUTHORIZED
        elif isinstance(exc, ServiceError):
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif isinstance(exc, ModelError):
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        elif isinstance(exc, ResilienceError):
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
        # Risk management specific errors
        if isinstance(exc, RiskProfileNotFoundError):
            status_code = status.HTTP_404_NOT_FOUND
        elif isinstance(exc, RiskParameterError):
            status_code = status.HTTP_400_BAD_REQUEST
        elif isinstance(exc, RiskLimitBreachError):
            status_code = status.HTTP_403_FORBIDDEN
        elif isinstance(exc, CircuitBreakerError):
            status_code = status.HTTP_403_FORBIDDEN
        elif isinstance(exc, RiskCalculationError):
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY

        return JSONResponse(
            status_code=status_code,
            content=format_error_response(
                exc.error_code,
                exc.message,
                exc.details,
                correlation_id,
                "risk-management-service"
            ),
        )

    # Handle RequestValidationError and ValidationError
    @app.exception_handler(RequestValidationError)
    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request: Request, exc: Union[RequestValidationError, ValidationError]):
        """Handle validation errors from FastAPI and Pydantic."""
        correlation_id = get_correlation_id(request)
        
        # Extract errors from the exception
        errors = exc.errors() if hasattr(exc, 'errors') else [{"msg": str(exc)}]

        logger.warning(
            f"Validation error for {request.method} {request.url.path}",
            extra={
                "errors": errors,
                "correlation_id": correlation_id
            },
        )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=format_error_response(
                "VALIDATION_ERROR",
                "Request validation failed",
                format_validation_errors(errors),
                correlation_id,
                "risk-management-service"
            ),
        )

    # Handle generic exceptions
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """Handle all other unhandled exceptions."""
        correlation_id = get_correlation_id(request)
        
        logger.error(
            f"Unhandled exception: {str(exc)}",
            extra={
                "path": request.url.path,
                "method": request.method,
                "traceback": traceback.format_exc(),
                "correlation_id": correlation_id
            },
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=format_error_response(
                "INTERNAL_SERVER_ERROR",
                "An unexpected error occurred",
                # Only include exception details in debug mode
                {"error": str(exc)} if logger.level <= 10 else None,  # 10 is DEBUG level
                correlation_id,
                "risk-management-service"
            ),
        )
