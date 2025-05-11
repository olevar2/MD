"""
Main entry point for the Data Pipeline Service.
"""
import os
from typing import Dict

import uvicorn
from common_lib.correlation import FastAPICorrelationIdMiddleware
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

# Import common-lib exceptions
from common_lib.errors.base_exceptions import (
    BaseError,
    ValidationError as CommonValidationError,
    DataError,
    ServiceError,
    NotFoundError,
    AuthenticationError,
    AuthorizationError
)

# Import local modules
from data_pipeline_service.api.router import router as api_router
from data_pipeline_service.api.metrics_integration import setup_metrics
from data_pipeline_service.config import get_service_config
from data_pipeline_service.database import database
from data_pipeline_service.logging_setup import setup_logging
from data_pipeline_service.service_clients import service_clients
from data_pipeline_service.error_handling import (
    handle_error,
    handle_exception,
    handle_async_exception,
    get_status_code
)
from data_pipeline_service.optimization import get_index_manager, initialize_optimized_pool, close_optimized_pool
from data_pipeline_service.adapters.adapter_factory import adapter_factory
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
logger = setup_logging("data-pipeline-service")


# Dependency functions for providing adapters to API endpoints
async def get_market_data_provider(request: Request):
    """
    Dependency function to provide the market data provider adapter.

    Args:
        request: The FastAPI request object

    Returns:
        An instance of IMarketDataProvider
    """
    if not hasattr(request.app.state, "adapter_factory"):
        raise ServiceUnavailableError("Adapter factory not initialized")

    return request.app.state.adapter_factory.get_market_data_provider()


async def get_market_data_cache(request: Request):
    """
    Dependency function to provide the market data cache adapter.

    Args:
        request: The FastAPI request object

    Returns:
        An instance of IMarketDataCache
    """
    if not hasattr(request.app.state, "adapter_factory"):
        raise ServiceUnavailableError("Adapter factory not initialized")

    return request.app.state.adapter_factory.get_market_data_cache()


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
    service_config = get_service_config()

    # Create FastAPI app
    app = FastAPI(
        title="Data Pipeline Service",
        description="Data Pipeline Service for Forex Trading Platform",
        version="0.1.0",
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
        allow_origins=service_config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add correlation ID middleware
    app.add_middleware(FastAPICorrelationIdMiddleware)

    # Set up metrics with standardized middleware
    setup_metrics(app, service_name="data-pipeline-service")

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
        },
        {
            "name": "adapter_factory",
            "check_func": lambda: True,  # Will be replaced at startup
            "critical": False,
        }
    ]

    # Setup dependencies
    dependencies: Dict[str, callable] = {
        "market_data_provider": get_market_data_provider,
        "market_data_cache": get_market_data_cache
    }

    # Add health check endpoints
    add_health_check_to_app(
        app=app,
        service_name="data-pipeline-service",
        version="0.1.0",
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
        await database.connect()

        # Update health check function to use actual DB connection check
        app.state.health.checks[0]["check_func"] = lambda: database.pool is not None

        # Initialize optimized connection pool
        try:
            await initialize_optimized_pool()
            logger.info("Optimized connection pool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize optimized connection pool: {e}")
            # Don't fail startup, service can run in degraded mode

        # Initialize database indexes
        try:
            # Get a database session
            async with get_db_session() as session:
                # Create index manager
                index_manager = await get_index_manager(session)

                # Ensure indexes for time series tables
                for table_name in [
                    "ohlcv", "ohlcv_1m", "ohlcv_5m", "ohlcv_15m", "ohlcv_30m",
                    "ohlcv_1h", "ohlcv_4h", "ohlcv_1d", "ohlcv_1w", "tick_data"
                ]:
                    await index_manager.ensure_indexes(table_name)
                    await index_manager.analyze_table(table_name)

                logger.info("Database indexes initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database indexes: {e}")
            # Don't fail startup, service can run in degraded mode

        # Initialize adapter factory
        try:
            # Store adapter factory in app state for easy access
            app.state.adapter_factory = adapter_factory

            # Get market data provider adapter
            market_data_provider = app.state.adapter_factory.get_market_data_provider()
            market_data_cache = app.state.adapter_factory.get_market_data_cache()

            # Update health check function for adapter factory
            app.state.health.checks[2]["check_func"] = lambda: (
                app.state.adapter_factory is not None and
                app.state.adapter_factory.get_market_data_provider() is not None
            )

            logger.info("Adapter factory initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize adapter factory: {e}")
            # Don't fail startup, service can run in degraded mode

        # Initialize Kafka event bus
        try:
            app.state.event_bus = KafkaEventBus(
                bootstrap_servers=service_config.kafka_bootstrap_servers,
                service_name="data-pipeline-service",
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
        await database.close()

        # Close optimized connection pool
        try:
            await close_optimized_pool()
            logger.info("Optimized connection pool closed")
        except Exception as e:
            logger.error(f"Failed to close optimized connection pool: {e}")

        # Clean up adapter factory
        if hasattr(app.state, "adapter_factory"):
            try:
                app.state.adapter_factory.clear_adapters()
                logger.info("Adapter factory cleaned up")
            except Exception as e:
                logger.error(f"Failed to clean up adapter factory: {e}")

        # Close event bus connection
        if hasattr(app.state, "event_bus"):
            app.state.event_bus.close()

        # Close service clients
        await service_clients.close_all()

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