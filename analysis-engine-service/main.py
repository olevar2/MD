"""
Analysis Engine Service - Main Application

This module initializes the FastAPI application for the Analysis Engine Service,
configuring all routes, middleware, and dependencies.
"""

import os
import sys
import signal
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Optional, Union
from datetime import datetime

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError as PydanticValidationError

# Core imports
from analysis_engine.config import AnalysisEngineSettings as Settings, get_settings
from analysis_engine.core.exceptions_bridge import (
    ForexTradingPlatformError,
    DataValidationError,
    DataFetchError as CommonDataFetchError,
    DataStorageError as CommonDataStorageError,
    DataTransformationError as CommonDataTransformationError,
    ConfigurationError as CommonConfigurationError,
    ServiceError,
    ServiceUnavailableError as CommonServiceUnavailableError,
    ServiceTimeoutError as CommonServiceTimeoutError,
    ModelError as CommonModelError,
    ModelTrainingError as CommonModelTrainingError,
    ModelPredictionError as CommonModelPredictionError
)
from analysis_engine.core.errors import (
    AnalysisEngineError,
    ValidationError,
    DataFetchError,
    AnalysisError,
    ConfigurationError,
    ServiceUnavailableError,
    # Common exception handler
    forex_platform_exception_handler,
    # Service-specific exception handlers
    analysis_engine_exception_handler,
    validation_exception_handler,
    data_fetch_exception_handler,
    analysis_exception_handler,
    configuration_exception_handler,
    service_unavailable_exception_handler,
    # Validation exception handlers
    pydantic_validation_exception_handler,
    fastapi_validation_exception_handler,
    # Common-lib exception handlers
    data_validation_exception_handler,
    common_data_fetch_exception_handler,
    data_storage_exception_handler,
    data_transformation_exception_handler,
    common_configuration_exception_handler,
    service_error_exception_handler,
    common_service_unavailable_exception_handler,
    service_timeout_exception_handler,
    model_error_exception_handler,
    model_training_exception_handler,
    model_prediction_exception_handler,
    # Generic exception handler
    generic_exception_handler
)
from analysis_engine.core.logging import configure_logging, get_logger
from analysis_engine.core.container import ServiceContainer
from analysis_engine.api.routes import setup_routes
from analysis_engine.api.v1.standardized.health import setup_health_routes
from analysis_engine.api.middleware import setup_middleware
from analysis_engine.api.metrics_integration import setup_metrics
from analysis_engine.core.memory_monitor import get_memory_monitor
from analysis_engine.core.monitoring.async_performance_monitor import get_async_monitor
from analysis_engine.monitoring.structured_logging import configure_structured_logging
from analysis_engine.core.exceptions import AnalysisEngineException
from analysis_engine.scheduling.scheduler_factory import initialize_schedulers, cleanup_schedulers

# Optional imports with error handling
try:
    from analysis_engine.causal.services.causal_inference_service import CausalInferenceService
    CAUSAL_INFERENCE_AVAILABLE = True
except ImportError:
    CAUSAL_INFERENCE_AVAILABLE = False
    logger.warning("Causal inference service not available")

try:
    from analysis_engine.analysis.multi_timeframe.multi_timeframe_analyzer import MultiTimeframeAnalyzer
    MULTI_TIMEFRAME_AVAILABLE = True
except ImportError:
    MULTI_TIMEFRAME_AVAILABLE = False
    logger.warning("Multi-timeframe analyzer not available")

# Initialize logger
logger = get_logger(__name__)

# Global shutdown event
shutdown_event = asyncio.Event()

# Constants
APP_NAME = "Analysis Engine Service"
APP_VERSION = "1.0.0"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the application lifecycle, handling setup and teardown of services.

    Args:
        app: FastAPI application instance
    """
    # Create and configure services
    if not hasattr(app.state, 'service_container') or app.state.service_container is None:
        app.state.service_container = ServiceContainer()
        logger.info("Initialized service container in app state.")

    service_container = app.state.service_container

    try:
        # Initialize database connections
        from analysis_engine.db.connection import initialize_database, initialize_async_database, check_async_db_connection

        # Initialize synchronous database
        initialize_database()
        logger.info("Synchronous database initialized")

        # Initialize asynchronous database
        await initialize_async_database()
        logger.info("Asynchronous database initialized")

        # Check database connection
        db_connected = await check_async_db_connection()
        if db_connected:
            logger.info("Database connection verified")
        else:
            logger.warning("Database connection check failed, but continuing startup")

        # Initialize core services
        await service_container.initialize()
        logger.info("Service container initialized successfully")

        # Initialize optional services if available
        if CAUSAL_INFERENCE_AVAILABLE:
            try:
                causal_service = CausalInferenceService()
                app.state.causal_service = causal_service
                logger.info("Causal inference service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize causal inference service: {e}")

        if MULTI_TIMEFRAME_AVAILABLE:
            try:
                multi_timeframe = MultiTimeframeAnalyzer()
                app.state.multi_timeframe_analyzer = multi_timeframe
                logger.info("Multi-timeframe analyzer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize multi-timeframe analyzer: {e}")

        # Initialize memory monitor
        memory_monitor = get_memory_monitor()
        await memory_monitor.start_monitoring()
        logger.info("Memory monitoring started")

        # Initialize async performance monitor
        async_monitor = get_async_monitor()
        await async_monitor.start_reporting(interval=300)  # Report every 5 minutes
        logger.info("Async performance monitoring started")

        # Initialize schedulers
        await initialize_schedulers(service_container)
        logger.info("Schedulers initialized and started")

        logger.info("Service initialization complete")
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise

    yield

    # Cleanup on shutdown
    try:
        # Stop schedulers
        if hasattr(app.state, 'service_container'):
            await cleanup_schedulers(app.state.service_container)
            logger.info("Schedulers stopped")

        # Cleanup service container
        if hasattr(app.state, 'service_container'):
            await app.state.service_container.cleanup()

        # Stop memory monitoring
        memory_monitor = get_memory_monitor()
        await memory_monitor.stop_monitoring()
        logger.info("Memory monitoring stopped")

        # Stop async performance monitoring
        async_monitor = get_async_monitor()
        await async_monitor.stop_reporting()
        logger.info("Async performance monitoring stopped")

        # Dispose database connections
        from analysis_engine.db.connection import dispose_database, dispose_async_database

        # Dispose synchronous database
        dispose_database()
        logger.info("Synchronous database disposed")

        # Dispose asynchronous database
        await dispose_async_database()
        logger.info("Asynchronous database disposed")

        logger.info("Service shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application
    """
    # Configure logging
    configure_logging()
    configure_structured_logging()

    # Create FastAPI app
    app = FastAPI(
        title=APP_NAME,
        description="Provides analytical capabilities for the trading platform",
        version=APP_VERSION,
        lifespan=lifespan,
        debug=get_settings().debug_mode,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add custom middleware for metrics, logging, and request tracking
    setup_middleware(app)

    # Set up standardized metrics
    setup_metrics(app, service_name="analysis-engine-service")

    # Register exception handlers

    # Register the base ForexTradingPlatformError handler (handles all derived exceptions)
    app.add_exception_handler(ForexTradingPlatformError, forex_platform_exception_handler)

    # Register validation exception handlers
    app.add_exception_handler(PydanticValidationError, pydantic_validation_exception_handler)
    app.add_exception_handler(RequestValidationError, fastapi_validation_exception_handler)

    # Register analysis-specific exception handlers
    from analysis_engine.core.exceptions_bridge import (
        AnalysisError,
        AnalyzerNotFoundError,
        InsufficientDataError,
        InvalidAnalysisParametersError,
        AnalysisTimeoutError,
        MarketRegimeError,
        SignalQualityError,
        ToolEffectivenessError,
        NLPAnalysisError,
        CorrelationAnalysisError,
        ManipulationDetectionError
    )

    # All of these will be handled by the forex_platform_exception_handler
    # but we register them explicitly for clarity and potential future customization
    app.add_exception_handler(AnalysisError, forex_platform_exception_handler)
    app.add_exception_handler(AnalyzerNotFoundError, forex_platform_exception_handler)
    app.add_exception_handler(InsufficientDataError, forex_platform_exception_handler)
    app.add_exception_handler(InvalidAnalysisParametersError, forex_platform_exception_handler)
    app.add_exception_handler(AnalysisTimeoutError, forex_platform_exception_handler)
    app.add_exception_handler(MarketRegimeError, forex_platform_exception_handler)
    app.add_exception_handler(SignalQualityError, forex_platform_exception_handler)
    app.add_exception_handler(ToolEffectivenessError, forex_platform_exception_handler)
    app.add_exception_handler(NLPAnalysisError, forex_platform_exception_handler)
    app.add_exception_handler(CorrelationAnalysisError, forex_platform_exception_handler)
    app.add_exception_handler(ManipulationDetectionError, forex_platform_exception_handler)

    # Register generic exception handler as fallback
    app.add_exception_handler(Exception, generic_exception_handler)

    logger.info("Enhanced exception handlers registered")

    # Setup routes (includes standardized health API endpoints)
    setup_routes(app)

    # Create service container for health checks
    service_container = ServiceContainer()

    # Setup health check routes
    setup_health_routes(app)

    logger.info("API routes and health checks configured")

    # Store service container in app state
    app.state.service_container = service_container

    return app

async def handle_shutdown(signal_name: str) -> None:
    """Handle shutdown signals gracefully"""
    logger.info(f"Received {signal_name} signal. Initiating graceful shutdown...")
    shutdown_event.set()

def setup_signal_handlers() -> None:
    """Setup signal handlers for graceful shutdown"""
    loop = asyncio.get_event_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig,
            lambda s=sig: asyncio.create_task(handle_shutdown(signal.Signals(s).name))
        )

async def main():
    """Main entry point for the Analysis Engine Service."""
    try:
        # Create application
        app = create_app()

        # Get settings
        settings = get_settings()

        # Setup signal handlers
        setup_signal_handlers()

        # Configure uvicorn
        config = uvicorn.Config(
            app=app,
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level.lower(),
            reload=settings.debug_mode
        )

        # Create server
        server = uvicorn.Server(config)

        # Start server
        logger.info(f"Starting Analysis Engine Service on {settings.host}:{settings.port}")
        await server.serve()

        # Wait for shutdown signal
        await shutdown_event.wait()

        # Graceful shutdown
        logger.info("Initiating graceful shutdown...")
        server.should_exit = True
        await server.shutdown()

    except Exception as e:
        logger.error(f"Error starting Analysis Engine Service: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
