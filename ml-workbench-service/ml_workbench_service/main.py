"""
ML Workbench Service - Main Application

This module initializes the FastAPI application for the ML Workbench Service,
configuring all routes, middleware, and dependencies.
"""

import logging
import os
import traceback
from typing import Union
from common_lib.correlation import FastAPICorrelationIdMiddleware
from fastapi import FastAPI, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

# Import API routers
from ml_workbench_service.api.v1 import (
    model_registry_router,
    model_training_router,
    model_serving_router,
    model_monitoring_router,
    # New in Phase 7
    transfer_learning_router
)

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
    ServiceTimeoutError,
    ModelError,
    ModelTrainingError,
    ModelPredictionError
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
API_PREFIX = "/api/v1"

# Create FastAPI app
app = FastAPI(
    title="ML Workbench Service",
    description="Machine Learning operations platform for training, serving, monitoring, and adapting ML models for trading strategies.",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add correlation ID middleware
app.add_middleware(FastAPICorrelationIdMiddleware)

# Include API routers
app.include_router(model_registry_router, prefix=API_PREFIX)
app.include_router(model_training_router, prefix=API_PREFIX)
app.include_router(model_serving_router, prefix=API_PREFIX)
app.include_router(model_monitoring_router, prefix=API_PREFIX)
# New in Phase 7
app.include_router(transfer_learning_router, prefix=API_PREFIX)

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "service": "ml-workbench-service"}

# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to ML Workbench Service", "docs_url": "/docs"}

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

# Handle ModelError
@app.exception_handler(ModelError)
async def model_exception_handler(request: Request, exc: ModelError):
    """Handle ModelError exceptions."""
    logger.error(
        f"Model error: {exc.message}",
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
    logger.info("ML Workbench Service starting up")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("ML Workbench Service shutting down")


if __name__ == "__main__":
    import uvicorn

    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8030))

    # Run the application
    uvicorn.run(
        "ml_workbench_service.main:app",
        host="0.0.0.0",
        port=port,
        reload=os.environ.get("ENVIRONMENT", "development") == "development"
    )
