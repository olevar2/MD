"""
Exception handlers for the Trading Gateway Service.

This module provides custom exception handlers for the FastAPI application
to ensure consistent error responses and proper error logging.
"""

import logging
import traceback
from typing import Dict, Any, Optional, Union

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from common_lib.exceptions import (
    ForexTradingPlatformError,
    ConfigurationError,
    DataError,
    DataValidationError,
    DataFetchError,
    DataStorageError,
    DataTransformationError,
    ServiceError,
    ServiceUnavailableError,
    ServiceTimeoutError,
    TradingError,
    OrderExecutionError,
    AuthenticationError,
    AuthorizationError
)

from .exceptions_bridge import (
    BrokerConnectionError,
    OrderValidationError as TGOrderValidationError,
    MarketDataError
)

# Initialize logger
logger = logging.getLogger("trading_gateway_service")

async def forex_platform_exception_handler(request: Request, exc: ForexTradingPlatformError):
    """Handle custom ForexTradingPlatformError exceptions."""
    logger.error(
        f"ForexTradingPlatformError: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
        },
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=exc.to_dict(),
    )

async def data_validation_exception_handler(request: Request, exc: DataValidationError):
    """Handle DataValidationError exceptions."""
    logger.warning(
        f"Data validation error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
        },
    )
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=exc.to_dict(),
    )

async def data_fetch_exception_handler(request: Request, exc: DataFetchError):
    """Handle DataFetchError exceptions."""
    logger.error(
        f"Data fetch error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
        },
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=exc.to_dict(),
    )

async def data_storage_exception_handler(request: Request, exc: DataStorageError):
    """Handle DataStorageError exceptions."""
    logger.error(
        f"Data storage error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
        },
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=exc.to_dict(),
    )

async def service_exception_handler(request: Request, exc: ServiceError):
    """Handle ServiceError exceptions."""
    logger.error(
        f"Service error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
        },
    )
    
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=exc.to_dict(),
    )

async def trading_exception_handler(request: Request, exc: TradingError):
    """Handle TradingError exceptions."""
    logger.error(
        f"Trading error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
        },
    )
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=exc.to_dict(),
    )

async def order_execution_exception_handler(request: Request, exc: OrderExecutionError):
    """Handle OrderExecutionError exceptions."""
    logger.error(
        f"Order execution error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
        },
    )
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=exc.to_dict(),
    )

async def broker_connection_exception_handler(request: Request, exc: BrokerConnectionError):
    """Handle BrokerConnectionError exceptions."""
    logger.error(
        f"Broker connection error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
        },
    )
    
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=exc.to_dict(),
    )

async def market_data_exception_handler(request: Request, exc: MarketDataError):
    """Handle MarketDataError exceptions."""
    logger.error(
        f"Market data error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
        },
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=exc.to_dict(),
    )

async def validation_exception_handler(
    request: Request, 
    exc: Union[RequestValidationError, ValidationError]
):
    """Handle FastAPI and Pydantic validation errors."""
    errors = []
    
    if isinstance(exc, RequestValidationError):
        for error in exc.errors():
            errors.append({
                "loc": error.get("loc", []),
                "msg": error.get("msg", ""),
                "type": error.get("type", "")
            })
    else:
        # Handle Pydantic ValidationError
        for error in exc.errors():
            errors.append({
                "loc": error.get("loc", []),
                "msg": error.get("msg", ""),
                "type": error.get("type", "")
            })
    
    logger.warning(
        "Validation error",
        extra={
            "path": request.url.path,
            "method": request.method,
            "errors": errors
        },
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error_type": "DataValidationError",
            "error_code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "details": {"errors": errors}
        },
    )

async def generic_exception_handler(request: Request, exc: Exception):
    """Handle any unhandled exceptions."""
    logger.error(
        f"Unhandled exception: {str(exc)}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "exception_type": exc.__class__.__name__,
            "traceback": traceback.format_exc()
        },
    )
    
    # Don't expose internal details in production
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_type": "ServiceError",
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred",
            "details": {"exception_type": exc.__class__.__name__}
        },
    )

def register_exception_handlers(app):
    """Register all exception handlers with the FastAPI app."""
    # Register the base ForexTradingPlatformError handler (handles all derived exceptions)
    app.add_exception_handler(ForexTradingPlatformError, forex_platform_exception_handler)
    
    # Register specific exception handlers for more tailored responses
    app.add_exception_handler(DataValidationError, data_validation_exception_handler)
    app.add_exception_handler(DataFetchError, data_fetch_exception_handler)
    app.add_exception_handler(DataStorageError, data_storage_exception_handler)
    app.add_exception_handler(ServiceError, service_exception_handler)
    app.add_exception_handler(TradingError, trading_exception_handler)
    app.add_exception_handler(OrderExecutionError, order_execution_exception_handler)
    app.add_exception_handler(BrokerConnectionError, broker_connection_exception_handler)
    app.add_exception_handler(MarketDataError, market_data_exception_handler)
    
    # Register validation exception handlers
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ValidationError, validation_exception_handler)
    
    # Register generic exception handler as fallback
    app.add_exception_handler(Exception, generic_exception_handler)
