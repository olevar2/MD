"""
API Error Handling Module

This module provides utilities for handling errors in API endpoints.
It includes functions for creating standardized error responses.
"""

import logging
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Type, Union

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from common_lib.errors.base_exceptions import BaseError, ErrorCode

# Get logger
logger = logging.getLogger(__name__)


def create_error_response(
    error: Union[Exception, BaseError],
    status_code: Optional[int] = None,
    correlation_id: Optional[str] = None,
    include_traceback: bool = False,
    log_level: int = logging.ERROR
) -> Tuple[Dict[str, Any], int]:
    """
    Create a standardized error response for API endpoints.

    Args:
        error: The exception to handle
        status_code: HTTP status code to return (overrides automatic mapping)
        correlation_id: Correlation ID for tracking the error
        include_traceback: Whether to include traceback in the response
        log_level: Logging level to use

    Returns:
        Tuple of (error response dict, HTTP status code)
    """
    # Generate correlation ID if not provided
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())

    # Create context dictionary
    context = {
        "correlation_id": correlation_id,
        "timestamp": datetime.utcnow().isoformat()
    }

    # Handle the exception
    if isinstance(error, BaseError):
        # Set correlation ID if not already set
        if not error.correlation_id:
            error.correlation_id = correlation_id

        # Log the error
        log_message = f"{error.__class__.__name__}: {error.message} (Code: {error.error_code.name}, ID: {error.correlation_id})"
        logger.log(log_level, log_message, extra=context)

        # Create error response
        error_response = error.to_dict()

        # Determine status code
        if status_code is None:
            status_code = _map_error_code_to_status_code(error.error_code)
    elif isinstance(error, HTTPException):
        # Convert FastAPI HTTPException to error response
        error_message = error.detail
        error_details = {}

        # Add traceback if requested
        if include_traceback:
            error_details["traceback"] = traceback.format_exc()

        # Log the error
        log_message = f"HTTPException: {error_message} (Status: {error.status_code})"
        logger.log(log_level, log_message, extra=context)

        # Create error response
        error_response = {
            "message": error_message,
            "error_code": f"HTTP_{error.status_code}",
            "correlation_id": correlation_id,
            "details": error_details,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Use status code from HTTPException
        status_code = error.status_code
    else:
        # Convert generic exception to error response
        error_message = str(error)
        error_details = {"original_error": error_message}

        # Add traceback if requested
        if include_traceback:
            error_details["traceback"] = traceback.format_exc()

        # Log the error
        log_message = f"Unhandled exception: {error_message}"
        logger.log(log_level, log_message, extra=context)

        # Create error response
        error_response = {
            "message": "Internal server error",
            "error_code": ErrorCode.UNKNOWN_ERROR.name,
            "correlation_id": correlation_id,
            "details": error_details,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Use default status code
        if status_code is None:
            status_code = 500

    # Return error response and status code
    return {"error": error_response}, status_code


def fastapi_exception_handler(
    request: Request,
    error: Exception
) -> JSONResponse:
    """
    Exception handler for FastAPI applications.

    Args:
        request: FastAPI request
        error: The exception to handle

    Returns:
        FastAPI JSONResponse
    """
    # Get correlation ID from request state or generate a new one
    correlation_id = getattr(request.state, "correlation_id", None) or str(uuid.uuid4())

    # Create error response
    error_response, status_code = create_error_response(
        error=error,
        correlation_id=correlation_id,
        include_traceback=False
    )

    # Return JSON response
    return JSONResponse(
        status_code=status_code,
        content=error_response,
        headers={"X-Correlation-ID": correlation_id}
    )


def _map_error_code_to_status_code(error_code: ErrorCode) -> int:
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
