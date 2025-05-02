"""
Risk Management Service Main Application.

This service handles risk monitoring, limit enforcement, and risk analytics.
"""
import os
import logging
import traceback
from typing import Dict, Optional, Callable, List, Union

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from prometheus_client import make_asgi_app

from core_foundations.utils.logger import get_logger
from core_foundations.api.health_check import add_health_check_to_app
from core_foundations.events.kafka_event_bus import KafkaEventBus
from core_foundations.events.event_schema import Event, EventType
from core_foundations.models.schemas import HealthStatus

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

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
APP_NAME = "Risk Management Service"
APP_VERSION = "0.1.0"

# CORS Configuration
DEFAULT_CORS_ORIGINS = ["http://localhost:3000", "https://forex-trading-platform.example.com"]
CORS_ORIGINS = os.getenv("CORS_ORIGINS")
ALLOWED_ORIGINS = CORS_ORIGINS.split(",") if CORS_ORIGINS else DEFAULT_CORS_ORIGINS

def handle_event(event: Event) -> None:
    """
    Process incoming events from the event bus.

    Args:
        event: The received event
    """
    logger.info(f"Received event: {event.event_type} from {event.source_service}")

    # Handle specific event types
    if event.event_type == EventType.POSITION_OPENED:
        logger.info(f"New position opened for {event.data.get('instrument')}, evaluating risk")
        # Add risk evaluation logic here
    elif event.event_type == EventType.MARKET_VOLATILITY_CHANGE:
        logger.info(f"Market volatility changed for {event.data.get('instrument')}")
        # Add volatility response logic
    elif event.event_type == EventType.RISK_LIMIT_BREACH:
        logger.warning(f"Risk limit breach detected: {event.data.get('details')}")
        # Add risk limit breach handling
    elif event.event_type == EventType.SERVICE_COMMAND:
        command = event.data.get('command')
        logger.info(f"Received service command: {command}")
# Will add these imports as we implement them
# from risk_management_service.api.router import api_router
# from risk_management_service.db.connection import check_db_connection, init_db

# Initialize logger
logger = get_logger("risk-management-service")

# Initialize application
app = FastAPI(
    title=APP_NAME,
    description="Service for risk monitoring, limit enforcement, and risk analytics",
    version=APP_VERSION,
)

# Configure CORS with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Setup health check endpoints
health_checks = [
    {
        "name": "risk_models_loaded",
        "check_func": lambda: True,  # Will be replaced at startup
        "critical": True,
    },
    {
        "name": "kafka_connection",
        "check_func": lambda: True,  # Will be replaced at startup
        "critical": True,
    }
]

# Setup dependencies
dependencies: Dict[str, Callable] = {}

# Add health check endpoints
add_health_check_to_app(
    app=app,
    service_name=APP_NAME,
    version=APP_VERSION,
    checks=health_checks,
    dependencies=dependencies,
)

# Add exception handlers

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

# Add API routes
from risk_management_service.api import dynamic_risk_routes
app.include_router(dynamic_risk_routes.router)

@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint returning service information."""
    return {
        "service": "risk-management-service",
        "version": "0.1.0",
        "status": "in development"
    }

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    logger.info(f"Starting {APP_NAME}")
    try:
        # Initialize risk models
        app.state.risk_models_loaded = True
        logger.info("Risk models loaded successfully")

        # Update health check for risk models
        app.state.health.checks[0]["check_func"] = lambda: app.state.risk_models_loaded

        # Initialize Kafka event bus
        try:
            app.state.event_bus = KafkaEventBus(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                service_name=APP_NAME,
                auto_create_topics=True
            )

            # Subscribe to relevant events
            app.state.event_bus.subscribe(
                event_types=[
                    EventType.POSITION_OPENED,
                    EventType.MARKET_VOLATILITY_CHANGE,
                    EventType.RISK_LIMIT_BREACH,
                    EventType.SERVICE_COMMAND,
                ],
                handler=handle_event
            )

            # Start consuming events in non-blocking mode
            app.state.event_bus.start_consuming(blocking=False)

            # Update health check function for Kafka
            app.state.health.checks[1]["check_func"] = lambda: app.state.event_bus is not None

            logger.info("Kafka event bus initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka event bus: {e}")
            # Don't fail startup, service can run in degraded mode

        logger.info("Service initialization complete")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info(f"Shutting down {APP_NAME}")

    # Close event bus connection if it exists
    if hasattr(app.state, "event_bus"):
        try:
            app.state.event_bus.flush()  # Ensure pending messages are sent
            app.state.event_bus.close()
            logger.info("Kafka event bus closed successfully")
        except Exception as e:
            logger.error(f"Error closing Kafka event bus: {e}")

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.getenv("PORT", "8004"))

    # Run the application
    uvicorn.run(
        "risk_management_service.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,  # Set to False in production
    )