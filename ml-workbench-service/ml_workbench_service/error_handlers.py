"""
Standardized Error Handling Module for ML Workbench Service

This module provides standardized error handling that follows the
common-lib pattern for error management.
"""

import logging
import traceback
from typing import Dict, Any, Optional, Callable, Type, Union, List
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError as PydanticValidationError

from common_lib.errors import (
    BaseError,
    ServiceError,
    NotFoundError,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    TimeoutError,
    ConnectionError,
    DatabaseError,
)
from ml_workbench_service.logging_setup import get_logger, get_correlation_id, log_exception

# Get logger
logger = get_logger(__name__)


def get_error_response(
    status_code: int,
    message: str,
    error_code: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get standardized error response.

    Args:
        status_code: HTTP status code
        message: Error message
        error_code: Error code
        details: Error details

    Returns:
        Error response
    """
    response = {
        "status": "error",
        "message": message,
    }

    if error_code is not None:
        response["error_code"] = error_code

    if details is not None:
        response["details"] = details

    # Add correlation ID if available
    correlation_id = get_correlation_id()
    if correlation_id:
        response["correlation_id"] = correlation_id

    return response


def setup_error_handlers(app: FastAPI) -> None:
    """
    Set up error handlers for the FastAPI application.

    Args:
        app: FastAPI application
    """

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """
        Handle request validation errors.

        Args:
            request: HTTP request
            exc: Exception

        Returns:
            JSON response
        """
        # Log the error
        log_exception(
            logger,
            exc,
            "Request validation error",
            extra={
                "path": request.url.path,
                "method": request.method,
                "client_host": request.client.host if request.client else None,
            },
        )

        # Get error details
        details = []
        for error in exc.errors():
            details.append(
                {
                    "loc": error.get("loc", []),
                    "msg": error.get("msg", ""),
                    "type": error.get("type", ""),
                }
            )

        # Return error response
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=get_error_response(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                message="Request validation error",
                error_code=422,
                details={"errors": details},
            ),
        )

    @app.exception_handler(PydanticValidationError)
    async def pydantic_validation_exception_handler(
        request: Request, exc: PydanticValidationError
    ) -> JSONResponse:
        """
        Handle Pydantic validation errors.

        Args:
            request: HTTP request
            exc: Exception

        Returns:
            JSON response
        """
        # Log the error
        log_exception(
            logger,
            exc,
            "Pydantic validation error",
            extra={
                "path": request.url.path,
                "method": request.method,
                "client_host": request.client.host if request.client else None,
            },
        )

        # Get error details
        details = []
        for error in exc.errors():
            details.append(
                {
                    "loc": error.get("loc", []),
                    "msg": error.get("msg", ""),
                    "type": error.get("type", ""),
                }
            )

        # Return error response
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=get_error_response(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                message="Validation error",
                error_code=422,
                details={"errors": details},
            ),
        )

    @app.exception_handler(ValidationError)
    async def custom_validation_exception_handler(
        request: Request, exc: ValidationError
    ) -> JSONResponse:
        """
        Handle custom validation errors.

        Args:
            request: HTTP request
            exc: Exception

        Returns:
            JSON response
        """
        # Log the error
        log_exception(
            logger,
            exc,
            "Validation error",
            extra={
                "path": request.url.path,
                "method": request.method,
                "client_host": request.client.host if request.client else None,
                "service_name": exc.service_name,
                "operation": exc.operation,
            },
        )

        # Return error response
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=get_error_response(
                status_code=status.HTTP_400_BAD_REQUEST,
                message=str(exc),
                error_code=400,
            ),
        )

    @app.exception_handler(NotFoundError)
    async def not_found_exception_handler(
        request: Request, exc: NotFoundError
    ) -> JSONResponse:
        """
        Handle not found errors.

        Args:
            request: HTTP request
            exc: Exception

        Returns:
            JSON response
        """
        # Log the error
        log_exception(
            logger,
            exc,
            "Not found error",
            extra={
                "path": request.url.path,
                "method": request.method,
                "client_host": request.client.host if request.client else None,
                "service_name": exc.service_name,
                "operation": exc.operation,
            },
        )

        # Return error response
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=get_error_response(
                status_code=status.HTTP_404_NOT_FOUND,
                message=str(exc),
                error_code=404,
            ),
        )

    @app.exception_handler(AuthenticationError)
    async def authentication_exception_handler(
        request: Request, exc: AuthenticationError
    ) -> JSONResponse:
        """
        Handle authentication errors.

        Args:
            request: HTTP request
            exc: Exception

        Returns:
            JSON response
        """
        # Log the error
        log_exception(
            logger,
            exc,
            "Authentication error",
            extra={
                "path": request.url.path,
                "method": request.method,
                "client_host": request.client.host if request.client else None,
                "service_name": exc.service_name,
                "operation": exc.operation,
            },
        )

        # Return error response
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content=get_error_response(
                status_code=status.HTTP_401_UNAUTHORIZED,
                message=str(exc),
                error_code=401,
            ),
        )

    @app.exception_handler(AuthorizationError)
    async def authorization_exception_handler(
        request: Request, exc: AuthorizationError
    ) -> JSONResponse:
        """
        Handle authorization errors.

        Args:
            request: HTTP request
            exc: Exception

        Returns:
            JSON response
        """
        # Log the error
        log_exception(
            logger,
            exc,
            "Authorization error",
            extra={
                "path": request.url.path,
                "method": request.method,
                "client_host": request.client.host if request.client else None,
                "service_name": exc.service_name,
                "operation": exc.operation,
            },
        )

        # Return error response
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content=get_error_response(
                status_code=status.HTTP_403_FORBIDDEN,
                message=str(exc),
                error_code=403,
            ),
        )

    @app.exception_handler(RateLimitError)
    async def rate_limit_exception_handler(
        request: Request, exc: RateLimitError
    ) -> JSONResponse:
        """
        Handle rate limit errors.

        Args:
            request: HTTP request
            exc: Exception

        Returns:
            JSON response
        """
        # Log the error
        log_exception(
            logger,
            exc,
            "Rate limit error",
            extra={
                "path": request.url.path,
                "method": request.method,
                "client_host": request.client.host if request.client else None,
                "service_name": exc.service_name,
                "operation": exc.operation,
            },
        )

        # Return error response
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=get_error_response(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                message=str(exc),
                error_code=429,
            ),
        )

    @app.exception_handler(TimeoutError)
    async def timeout_exception_handler(
        request: Request, exc: TimeoutError
    ) -> JSONResponse:
        """
        Handle timeout errors.

        Args:
            request: HTTP request
            exc: Exception

        Returns:
            JSON response
        """
        # Log the error
        log_exception(
            logger,
            exc,
            "Timeout error",
            extra={
                "path": request.url.path,
                "method": request.method,
                "client_host": request.client.host if request.client else None,
                "service_name": exc.service_name,
                "operation": exc.operation,
            },
        )

        # Return error response
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content=get_error_response(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                message=str(exc),
                error_code=504,
            ),
        )

    @app.exception_handler(ConnectionError)
    async def connection_exception_handler(
        request: Request, exc: ConnectionError
    ) -> JSONResponse:
        """
        Handle connection errors.

        Args:
            request: HTTP request
            exc: Exception

        Returns:
            JSON response
        """
        # Log the error
        log_exception(
            logger,
            exc,
            "Connection error",
            extra={
                "path": request.url.path,
                "method": request.method,
                "client_host": request.client.host if request.client else None,
                "service_name": exc.service_name,
                "operation": exc.operation,
            },
        )

        # Return error response
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=get_error_response(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                message=str(exc),
                error_code=503,
            ),
        )

    @app.exception_handler(DatabaseError)
    async def database_exception_handler(
        request: Request, exc: DatabaseError
    ) -> JSONResponse:
        """
        Handle database errors.

        Args:
            request: HTTP request
            exc: Exception

        Returns:
            JSON response
        """
        # Log the error
        log_exception(
            logger,
            exc,
            "Database error",
            extra={
                "path": request.url.path,
                "method": request.method,
                "client_host": request.client.host if request.client else None,
                "service_name": exc.service_name,
                "operation": exc.operation,
            },
        )

        # Return error response
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=get_error_response(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message="Database error occurred",
                error_code=500,
            ),
        )

    @app.exception_handler(ServiceError)
    async def service_exception_handler(
        request: Request, exc: ServiceError
    ) -> JSONResponse:
        """
        Handle service errors.

        Args:
            request: HTTP request
            exc: Exception

        Returns:
            JSON response
        """
        # Log the error
        log_exception(
            logger,
            exc,
            "Service error",
            extra={
                "path": request.url.path,
                "method": request.method,
                "client_host": request.client.host if request.client else None,
                "service_name": exc.service_name,
                "operation": exc.operation,
            },
        )

        # Return error response
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=get_error_response(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message=str(exc),
                error_code=500,
            ),
        )

    @app.exception_handler(BaseError)
    async def base_exception_handler(
        request: Request, exc: BaseError
    ) -> JSONResponse:
        """
        Handle base errors.

        Args:
            request: HTTP request
            exc: Exception

        Returns:
            JSON response
        """
        # Log the error
        log_exception(
            logger,
            exc,
            "Base error",
            extra={
                "path": request.url.path,
                "method": request.method,
                "client_host": request.client.host if request.client else None,
                "service_name": getattr(exc, "service_name", None),
                "operation": getattr(exc, "operation", None),
            },
        )

        # Return error response
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=get_error_response(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message=str(exc),
                error_code=500,
            ),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """
        Handle generic exceptions.

        Args:
            request: HTTP request
            exc: Exception

        Returns:
            JSON response
        """
        # Log the error
        log_exception(
            logger,
            exc,
            "Unhandled exception",
            extra={
                "path": request.url.path,
                "method": request.method,
                "client_host": request.client.host if request.client else None,
                "traceback": traceback.format_exc(),
            },
        )

        # Return error response
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=get_error_response(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message="An unexpected error occurred",
                error_code=500,
            ),
        )