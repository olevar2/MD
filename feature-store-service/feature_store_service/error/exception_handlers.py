"""
Exception handlers for the Feature Store Service.

This module provides custom exception handlers for the FastAPI application
to ensure consistent error responses and proper error logging.
"""

import logging
import traceback
from typing import Dict, Any, Optional, Union, List

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

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
from feature_store_service.error.error_manager import IndicatorError

logger = logging.getLogger(__name__)


async def forex_platform_exception_handler(
    request: Request, exc: ForexTradingPlatformError
) -> JSONResponse:
    """
    Handle custom ForexTradingPlatformError exceptions.

    Args:
        request: The request that caused the exception
        exc: The exception instance

    Returns:
        JSONResponse with appropriate error details
    """
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


async def indicator_error_handler(
    request: Request, exc: IndicatorError
) -> JSONResponse:
    """
    Handle IndicatorError exceptions.

    Args:
        request: The request that caused the exception
        exc: The exception instance

    Returns:
        JSONResponse with appropriate error details
    """
    logger.error(
        f"IndicatorError: {exc.error_type}",
        extra={
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_type": exc.error_type,
            "message": str(exc),
            "details": exc.details,
        },
    )


async def validation_exception_handler(
    request: Request, exc: Union[RequestValidationError, ValidationError]
) -> JSONResponse:
    """
    Handle validation errors from FastAPI and Pydantic.

    Args:
        request: The request that caused the exception
        exc: The exception instance

    Returns:
        JSONResponse with validation error details
    """
    errors: List[Dict[str, Any]] = []

    if isinstance(exc, RequestValidationError):
        errors = exc.errors()
    elif isinstance(exc, ValidationError):
        errors = exc.errors()

    logger.warning(
        f"Validation error for {request.method} {request.url.path}",
        extra={
            "errors": errors,
            "body": await request.body(),
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


async def sqlalchemy_exception_handler(
    request: Request, exc: SQLAlchemyError
) -> JSONResponse:
    """
    Handle SQLAlchemy database errors.

    Args:
        request: The request that caused the exception
        exc: The exception instance

    Returns:
        JSONResponse with database error details
    """
    logger.error(
        f"Database error: {str(exc)}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "traceback": traceback.format_exc(),
        },
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_type": "DatabaseError",
            "message": "A database error occurred",
            # Don't expose internal database details in production
            "details": str(exc) if logger.level <= logging.DEBUG else None,
        },
    )


async def data_validation_exception_handler(
    request: Request, exc: DataValidationError
) -> JSONResponse:
    """
    Handle DataValidationError exceptions.

    Args:
        request: The request that caused the exception
        exc: The exception instance

    Returns:
        JSONResponse with appropriate error details
    """
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


async def data_fetch_exception_handler(
    request: Request, exc: DataFetchError
) -> JSONResponse:
    """
    Handle DataFetchError exceptions.

    Args:
        request: The request that caused the exception
        exc: The exception instance

    Returns:
        JSONResponse with appropriate error details
    """
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


async def data_storage_exception_handler(
    request: Request, exc: DataStorageError
) -> JSONResponse:
    """
    Handle DataStorageError exceptions.

    Args:
        request: The request that caused the exception
        exc: The exception instance

    Returns:
        JSONResponse with appropriate error details
    """
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


async def data_transformation_exception_handler(
    request: Request, exc: DataTransformationError
) -> JSONResponse:
    """
    Handle DataTransformationError exceptions.

    Args:
        request: The request that caused the exception
        exc: The exception instance

    Returns:
        JSONResponse with appropriate error details
    """
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


async def service_exception_handler(
    request: Request, exc: ServiceError
) -> JSONResponse:
    """
    Handle ServiceError exceptions.

    Args:
        request: The request that caused the exception
        exc: The exception instance

    Returns:
        JSONResponse with appropriate error details
    """
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


async def general_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """
    Handle all other unhandled exceptions.

    Args:
        request: The request that caused the exception
        exc: The exception instance

    Returns:
        JSONResponse with error details
    """
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
