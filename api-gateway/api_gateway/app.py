"""
API Gateway Application

This module provides the main application for the API Gateway.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable, Awaitable

from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRouter
import httpx

from common_lib.config.config_manager import ConfigManager
from common_lib.errors.handler import ErrorHandler, ErrorContext, ErrorResponse
from api_gateway.routes import market_data, analysis, trading, feature_store
from api_gateway.middleware.auth import AuthMiddleware
from api_gateway.middleware.logging import LoggingMiddleware
from api_gateway.middleware.rate_limit import RateLimitMiddleware
from api_gateway.middleware.correlation import CorrelationMiddleware
from api_gateway.middleware.xss_protection import XSSProtectionMiddleware
from api_gateway.middleware.csrf_protection import CSRFProtectionMiddleware
from api_gateway.middleware.security_headers import SecurityHeadersMiddleware


# Create logger
logger = logging.getLogger("api_gateway")


# Create FastAPI application
app = FastAPI(
    title="Forex Trading Platform API",
    description="API Gateway for the Forex Trading Platform",
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
app.add_middleware(CorrelationMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(AuthMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(XSSProtectionMiddleware)
app.add_middleware(CSRFProtectionMiddleware)
app.add_middleware(SecurityHeadersMiddleware)


# Include routers
app.include_router(market_data.router, prefix="/api/v1/market-data", tags=["Market Data"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["Analysis"])
app.include_router(trading.router, prefix="/api/v1/trading", tags=["Trading"])
app.include_router(feature_store.router, prefix="/api/v1/features", tags=["Feature Store"])


# Create error handler
error_handler = ErrorHandler(logger=logger)


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
    # Create error context
    context = ErrorContext(
        correlation_id=request.headers.get("X-Correlation-ID", ""),
        service_name="api-gateway",
        operation=f"{request.method} {request.url.path}",
        request_id=request.headers.get("X-Request-ID", ""),
        request_path=request.url.path,
        request_method=request.method,
        request_params=dict(request.query_params),
        user_id=request.headers.get("X-User-ID", "")
    )

    # Handle error
    error_response = error_handler.handle_error(exc, context)

    # Return JSON response
    return JSONResponse(
        status_code=_get_status_code(error_response),
        content=error_response.to_dict()
    )


def _get_status_code(error_response: ErrorResponse) -> int:
    """
    Get HTTP status code for an error response.

    Args:
        error_response: Error response

    Returns:
        HTTP status code
    """
    # Map error codes to HTTP status codes
    code_map = {
        # General errors
        1000: 500,  # UNKNOWN_ERROR
        1001: 400,  # INVALID_INPUT
        1002: 404,  # RESOURCE_NOT_FOUND
        1003: 409,  # RESOURCE_ALREADY_EXISTS
        1004: 405,  # OPERATION_NOT_SUPPORTED
        1005: 403,  # PERMISSION_DENIED
        1006: 401,  # AUTHENTICATION_FAILED
        1007: 429,  # RATE_LIMIT_EXCEEDED
        1008: 504,  # TIMEOUT

        # Service errors
        2000: 503,  # SERVICE_UNAVAILABLE
        2001: 504,  # SERVICE_TIMEOUT
        2002: 401,  # SERVICE_AUTHENTICATION_FAILED
        2003: 403,  # SERVICE_AUTHORIZATION_FAILED
        2004: 400,  # SERVICE_VALIDATION_FAILED
        2005: 404,  # SERVICE_RESOURCE_NOT_FOUND
        2006: 409,  # SERVICE_RESOURCE_CONFLICT
        2007: 500,  # SERVICE_INTERNAL_ERROR

        # Data errors
        3000: 400,  # DATA_VALIDATION_FAILED
        3001: 404,  # DATA_NOT_FOUND
        3002: 409,  # DATA_DUPLICATE
        3003: 500,  # DATA_CORRUPTION
        3004: 500,  # DATA_PROCESSING_FAILED
    }

    # Get status code from map, or default to 500
    return code_map.get(error_response.code, 500)


@app.on_event("startup")
async def startup():
    """
    Startup event handler.
    """
    # Load configuration
    config_manager = ConfigManager()

    try:
        # Load configuration from file
        config_manager.load_config("config/api-gateway.yaml")

        # Configure logging
        logging_config = config_manager.get_logging_config()
        logging.basicConfig(
            level=getattr(logging, logging_config.level),
            format=logging_config.format
        )

        logger.info("API Gateway started")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown():
    """
    Shutdown event handler.
    """
    logger.info("API Gateway shutting down")


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        Health status
    """
    return {"status": "ok"}