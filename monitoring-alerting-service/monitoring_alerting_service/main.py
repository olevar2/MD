"""
Monitoring & Alerting Service - Main Application

This is the entry point for the Monitoring & Alerting Service, which provides monitoring,
alerting, and visualization capabilities for the Forex Trading Platform.
"""

import os
import logging
import traceback
from typing import Union

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from prometheus_client import make_asgi_app

# Import common-lib exceptions
from common_lib.exceptions import (
    ForexTradingPlatformError,
    ConfigurationError,
    DataError,
    DataValidationError,
    DataFetchError,
    DataStorageError,
    ServiceError,
    ServiceUnavailableError,
    ServiceTimeoutError
)

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("monitoring_alerting_service")

# Initialize FastAPI app
app = FastAPI(
    title="Monitoring & Alerting Service",
    description="Monitoring, alerting, and visualization for the Forex Trading Platform",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to Monitoring & Alerting Service", "docs_url": "/docs"}

# Error handlers

# Handle ForexTradingPlatformError (base exception for all platform errors)
@app.exception_handler(ForexTradingPlatformError)
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

# Handle DataValidationError
@app.exception_handler(DataValidationError)
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

# Handle DataFetchError
@app.exception_handler(DataFetchError)
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

# Handle DataStorageError
@app.exception_handler(DataStorageError)
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

# Handle ServiceError
@app.exception_handler(ServiceError)
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

# Handle RequestValidationError and ValidationError
@app.exception_handler(RequestValidationError)
@app.exception_handler(ValidationError)
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

# Handle generic exceptions
@app.exception_handler(Exception)
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

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize service components on startup"""
    logger.info("Starting Monitoring & Alerting Service")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down Monitoring & Alerting Service")

# Main entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8009))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting Monitoring & Alerting Service on {host}:{port}")
    uvicorn.run(
        "monitoring_alerting_service.main:app",
        host=host,
        port=port,
        reload=os.environ.get("DEBUG", "false").lower() == "true",
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )
