"""
Error handlers for the Feature Store Service.

This module provides exception handlers for the FastAPI application to handle
custom exceptions from common-lib in a standardized way.
"""
import logging
import traceback
from typing import Union

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

# Import common-lib exceptions
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
    ServiceTimeoutError
)

# Initialize logger
logger = logging.getLogger("feature_store_service")

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
            "path": request.url.path,
            "method": request.method,
            "data": str(exc.data) if hasattr(exc, 'data') else None,
        },
    )
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error_type": "DataValidationError",
            "message": exc.message,
            "details": str(exc.data) if hasattr(exc, 'data') else None,
        },
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

async def data_transformation_exception_handler(request: Request, exc: DataTransformationError):
    """Handle DataTransformationError exceptions."""
    logger.error(
        f"Data transformation error: {exc.message}",
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

async def validation_exception_handler(request: Request, exc: Union[RequestValidationError, ValidationError]):
    """Handle validation errors from FastAPI and Pydantic."""
    # Extract errors from the exception
    errors = exc.errors() if hasattr(exc, 'errors') else [{"msg": str(exc)}]
    
    logger.warning(
        f"Validation error for {request.method} {request.url.path}",
        extra={
            "errors": errors,
        },
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error_type": "ValidationError",
            "message": "Request validation failed",
            "details": errors,
        },
    )

async def generic_exception_handler(request: Request, exc: Exception):
    """Handle all other unhandled exceptions."""
    logger.error(
        f"Unhandled exception: {str(exc)}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "traceback": traceback.format_exc(),
        },
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_type": "InternalServerError",
            "message": "An unexpected error occurred",
            # Only include exception details in debug mode
            "details": str(exc) if logger.level <= logging.DEBUG else None,
        },
    )
