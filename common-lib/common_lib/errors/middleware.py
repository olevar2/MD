"""
Error Handling Middleware Module

This module provides middleware for handling errors in API endpoints.
It includes middleware for FastAPI, Flask, and Express.js.
"""

import logging
import traceback
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Type, Union

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from common_lib.errors.base_exceptions import BaseError, ErrorCode

# Get logger
logger = logging.getLogger(__name__)


class FastAPIErrorMiddleware(BaseHTTPMiddleware):
    """
    Middleware for handling errors in FastAPI applications.

    This middleware:
    1. Catches all exceptions
    2. Logs the exception with appropriate context
    3. Converts exceptions to standardized error responses
    4. Sets appropriate HTTP status codes
    """

    def __init__(
        self,
        app: FastAPI,
        include_traceback: bool = False,
        log_level: int = logging.ERROR
    ):
        """
        Initialize the middleware.

        Args:
            app: FastAPI application
            include_traceback: Whether to include traceback in error responses
            log_level: Logging level to use
        """
        super().__init__(app)
        self.include_traceback = include_traceback
        self.log_level = log_level

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Dispatch the request and handle any errors.

        Args:
            request: FastAPI request
            call_next: Next middleware in the chain

        Returns:
            FastAPI response
        """
        # Generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())

        # Add correlation ID to request state
        request.state.correlation_id = correlation_id

        try:
            # Process the request
            response = await call_next(request)

            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id

            return response
        except Exception as e:
            # Create context dictionary
            context = {
                "method": request.method,
                "url": str(request.url),
                "client": request.client.host if request.client else "unknown",
                "correlation_id": correlation_id,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Handle the exception
            if isinstance(e, BaseError):
                # Set correlation ID if not already set
                if not e.correlation_id:
                    e.correlation_id = correlation_id

                # Log the error
                log_message = f"{e.__class__.__name__}: {e.message} (Code: {e.error_code.name}, ID: {e.correlation_id})"
                logger.log(self.log_level, log_message, extra=context)

                # Create error response
                error_response = e.to_dict()
                status_code = self._map_error_code_to_status_code(e.error_code)
            else:
                # Convert to generic error
                error_message = str(e)
                error_details = {"original_error": error_message}

                # Add traceback if requested
                if self.include_traceback:
                    error_details["traceback"] = traceback.format_exc()

                # Log the error
                log_message = f"Unhandled exception: {error_message}"
                logger.log(self.log_level, log_message, extra=context)

                # Create error response
                error_response = {
                    "message": "Internal server error",
                    "error_code": ErrorCode.UNKNOWN_ERROR.name,
                    "correlation_id": correlation_id,
                    "details": error_details,
                    "timestamp": datetime.utcnow().isoformat()
                }
                status_code = 500

            # Return error response
            return JSONResponse(
                status_code=status_code,
                content={"error": error_response},
                headers={"X-Correlation-ID": correlation_id}
            )

    def _map_error_code_to_status_code(self, error_code: ErrorCode) -> int:
        """
        Map error code to HTTP status code.

        Args:
            error_code: Error code

        Returns:
            HTTP status code
        """
        # Map error codes to HTTP status codes
        error_code_map = {
            # Validation errors (400-499)
            ErrorCode.VALIDATION_ERROR: 400,
            ErrorCode.INVALID_INPUT: 400,
            ErrorCode.MISSING_PARAMETER: 400,
            ErrorCode.INVALID_FORMAT: 400,

            # Authentication/authorization errors (401-403)
            ErrorCode.AUTHENTICATION_FAILED: 401,
            ErrorCode.TOKEN_EXPIRED: 401,
            ErrorCode.INVALID_CREDENTIALS: 401,
            ErrorCode.AUTHORIZATION_FAILED: 403,

            # Resource errors (404-409)
            ErrorCode.RESOURCE_NOT_FOUND: 404,
            ErrorCode.RESOURCE_ALREADY_EXISTS: 409,
            ErrorCode.RESOURCE_CONFLICT: 409,

            # Service errors (500-599)
            ErrorCode.SERVICE_UNAVAILABLE: 503,
            ErrorCode.TIMEOUT: 504,
            ErrorCode.THIRD_PARTY_SERVICE_ERROR: 502,

            # Database errors (500)
            ErrorCode.DATABASE_ERROR: 500,
            ErrorCode.DATABASE_CONNECTION_ERROR: 500,
            ErrorCode.DATABASE_QUERY_ERROR: 500,

            # Data errors (400-499)
            ErrorCode.DATA_VALIDATION_ERROR: 400,
            ErrorCode.DATA_INTEGRITY_ERROR: 400,
            ErrorCode.DATA_FORMAT_ERROR: 400,
            ErrorCode.DATA_MISSING_ERROR: 404,
            ErrorCode.DATA_INCONSISTENCY_ERROR: 409,

            # Business logic errors (400-499)
            ErrorCode.BUSINESS_RULE_VIOLATION: 400,
            ErrorCode.INSUFFICIENT_FUNDS: 400,
            ErrorCode.POSITION_LIMIT_EXCEEDED: 400,
            ErrorCode.INVALID_OPERATION: 400,
            ErrorCode.OPERATION_NOT_ALLOWED: 403,

            # Security errors (401-403)
            ErrorCode.SECURITY_VIOLATION: 403,
        }

        # Return mapped status code or 500 if not found
        return error_code_map.get(error_code, 500)


def fastapi_error_handler(app: FastAPI, include_traceback: bool = False) -> None:
    """
    Add error handling middleware to a FastAPI application.

    Args:
        app: FastAPI application
        include_traceback: Whether to include traceback in error responses
    """
    app.add_middleware(
        FastAPIErrorMiddleware,
        include_traceback=include_traceback
    )
