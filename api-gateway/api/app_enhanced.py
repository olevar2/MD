"""
Enhanced API Gateway Application

This module provides the main application for the enhanced API Gateway.
"""

import logging
import uuid
from typing import Dict, Any, Optional

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from common_lib.config.config_manager import ConfigManager
from common_lib.errors.handler import ErrorHandler, ErrorContext, ErrorResponse

from .routes import proxy_router
from ..core.auth import EnhancedAuthMiddleware
from ..core.rate_limit import EnhancedRateLimitMiddleware
from ..core.response.standard_response import create_error_response


# Create logger
logger = logging.getLogger("api_gateway")


# Create FastAPI application
app = FastAPI(
    title="Forex Trading Platform API",
    description="Enhanced API Gateway for the Forex Trading Platform",
    version="1.0.0"
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add custom middleware
app.add_middleware(EnhancedAuthMiddleware)
app.add_middleware(EnhancedRateLimitMiddleware)


# Include routers
app.include_router(proxy_router, prefix="/api/v1")


# Create error handler
error_handler = ErrorHandler(logger=logger)


@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    """
    Add correlation ID to request and response.

    Args:
        request: FastAPI request
        call_next: Next middleware or route handler

    Returns:
        Response
    """
    # Get correlation ID from header, or generate a new one
    correlation_id = request.headers.get("X-Correlation-ID")
    if not correlation_id:
        correlation_id = str(uuid.uuid4())

        # Add correlation ID to request headers
        request.headers.__dict__["_list"].append(
            (b"x-correlation-id", correlation_id.encode())
        )

    # Get or generate request ID
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())

        # Add request ID to request headers
        request.headers.__dict__["_list"].append(
            (b"x-request-id", request_id.encode())
        )

    # Process the request
    response = await call_next(request)

    # Add correlation ID and request ID to response headers
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Request-ID"] = request_id

    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log requests and responses.

    Args:
        request: FastAPI request
        call_next: Next middleware or route handler

    Returns:
        Response
    """
    # Get correlation ID and request ID
    correlation_id = request.headers.get("X-Correlation-ID", "")
    request_id = request.headers.get("X-Request-ID", "")

    # Log request
    logger.info(
        f"Request: {request.method} {request.url.path}",
        extra={
            "correlation_id": correlation_id,
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params),
            "client_host": request.client.host if request.client else "unknown"
        }
    )

    # Process the request
    response = await call_next(request)

    # Log response
    logger.info(
        f"Response: {response.status_code}",
        extra={
            "correlation_id": correlation_id,
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code
        }
    )

    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler.

    Args:
        request: FastAPI request
        exc: Exception

    Returns:
        JSON response with error details
    """
    # Get correlation ID and request ID
    correlation_id = request.headers.get("X-Correlation-ID", "")
    request_id = request.headers.get("X-Request-ID", "")

    # Log the error
    logger.error(
        f"Unhandled exception: {str(exc)}",
        extra={
            "correlation_id": correlation_id,
            "request_id": request_id,
            "exception_type": exc.__class__.__name__,
            "exception_message": str(exc)
        },
        exc_info=True
    )

    # Create error response
    return JSONResponse(
        status_code=500,
        content=create_error_response(
            code="INTERNAL_SERVER_ERROR",
            message="Internal server error",
            correlation_id=correlation_id,
            request_id=request_id,
            details={
                "error_type": exc.__class__.__name__,
                "error_message": str(exc)
            }
        ).dict(),
        headers={
            "X-Correlation-ID": correlation_id,
            "X-Request-ID": request_id
        }
    )


@app.on_event("startup")
async def startup():
    """
    Startup event handler.
    """
    # Load configuration
    config_manager = ConfigManager()

    try:
        # Load configuration from file
        config_manager.load_config("config/api-gateway-enhanced.yaml")

        # Configure logging
        logging_config = config_manager.get_logging_config()
        logging.basicConfig(
            level=getattr(logging, logging_config.level),
            format=logging_config.format
        )

        logger.info("Enhanced API Gateway started")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown():
    """
    Shutdown event handler.
    """
    logger.info("Enhanced API Gateway shutting down")


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        Health status
    """
    return {"status": "ok"}