"""
Main entry point for the Data Pipeline Service.
"""
import os
from typing import Dict

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from prometheus_client import make_asgi_app

from core_foundations.api.health_check import add_health_check_to_app
from core_foundations.events.kafka_event_bus import KafkaEventBus
from core_foundations.events.event_schema import EventType
from core_foundations.models.schemas import HealthStatus
from core_foundations.utils.logger import get_logger

# Import common-lib exceptions
from common_lib.exceptions import (
    ForexTradingPlatformError,
    DataValidationError,
    DataFetchError,
    DataStorageError,
    ServiceError,
    ServiceUnavailableError
)

# Import local modules
from data_pipeline_service.api.router import router as api_router
from data_pipeline_service.config.settings import get_settings
from data_pipeline_service.db.engine import create_db_engine, dispose_db_engine
# Import error handlers
from data_pipeline_service.error_handlers import (
    forex_platform_exception_handler,
    data_validation_exception_handler,
    data_fetch_exception_handler,
    data_storage_exception_handler,
    service_exception_handler,
    validation_exception_handler,
    generic_exception_handler
)

# Initialize logger
logger = get_logger("data-pipeline-service")


def handle_event(event):
    """
    Process incoming events from the event bus.

    Args:
        event: The received event
    """
    logger.info(f"Received event: {event.event_type} from {event.source_service}")

    # Handle specific event types
    if event.event_type == EventType.MARKET_DATA_UPDATED:
        # Handle market data update events
        logger.debug(f"Market data updated for {event.data.get('instrument', 'unknown')}")
    elif event.event_type == EventType.DATA_QUALITY_ALERT:
        # Handle data quality alerts
        logger.warning(f"Data quality alert: {event.data.get('message')}")
    elif event.event_type == EventType.SERVICE_COMMAND:
        # Handle service commands
        command = event.data.get('command')
        logger.info(f"Received service command: {command}")


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        description="Data Pipeline Service for Forex Trading Platform",
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Register custom exception handlers
    app.add_exception_handler(ForexTradingPlatformError, forex_platform_exception_handler)
    app.add_exception_handler(DataValidationError, data_validation_exception_handler)
    app.add_exception_handler(DataFetchError, data_fetch_exception_handler)
    app.add_exception_handler(DataStorageError, data_storage_exception_handler)
    app.add_exception_handler(ServiceError, service_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add Prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # Setup health check endpoints
    health_checks = [
        {
            "name": "database_connection",
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
    dependencies: Dict[str, callable] = {}

    # Add health check endpoints
    add_health_check_to_app(
        app=app,
        service_name=settings.app_name,
        version=settings.app_version,
        checks=health_checks,
        dependencies=dependencies,
    )

    # Include API router
    app.include_router(api_router)

    # Add startup and shutdown event handlers
    @app.on_event("startup")
    async def startup_service():
        """Initialize service dependencies and connections."""
        logger.info("Starting up data pipeline service")

        # Initialize database connection
        app.state.db_engine = await create_db_engine(settings)

        # Update health check function to use actual DB connection check
        app.state.health.checks[0]["check_func"] = lambda: app.state.db_engine is not None        # Initialize Kafka event bus
        try:
            app.state.event_bus = KafkaEventBus(
                bootstrap_servers=settings.kafka_bootstrap_servers,
                service_name=settings.app_name,
                auto_create_topics=True
            )

            # Subscribe to relevant events
            app.state.event_bus.subscribe(
                event_types=[
                    EventType.MARKET_DATA_UPDATED,
                    EventType.DATA_QUALITY_ALERT,
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

    @app.on_event("shutdown")
    async def shutdown_service():
        """Close service connections."""
        logger.info("Shutting down data pipeline service")

        # Close database connection
        if hasattr(app.state, "db_engine"):
            await dispose_db_engine(app.state.db_engine)

        # Close event bus connection
        if hasattr(app.state, "event_bus"):
            app.state.event_bus.close()

    return app


app = create_application()

if __name__ == "__main__":
    """Run the application directly for development."""
    uvicorn.run(
        "data_pipeline_service.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )