"""
Strategy Execution Engine - Main Application

This module initializes the FastAPI application for the Strategy Execution Engine,
configuring all routes, middleware, and dependencies.
"""

import os
import sys
import signal
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Optional, Union, List, Any
from datetime import datetime

import uvicorn
from fastapi import FastAPI, Request, status, Depends, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError as PydanticValidationError, Field
from prometheus_client import make_asgi_app

# Import error handling
from strategy_execution_engine.error import (
    ForexTradingPlatformError,
    StrategyExecutionError,
    StrategyConfigurationError,
    StrategyLoadError,
    BacktestError,
    BacktestConfigError,
    BacktestDataError,
    BacktestExecutionError,
    BacktestReportError,
    with_error_handling,
    async_with_error_handling
)

# Import core components
from strategy_execution_engine.core.config import get_settings
from strategy_execution_engine.core.logging import configure_logging, get_logger
from strategy_execution_engine.core.container import ServiceContainer
from strategy_execution_engine.core.monitoring import setup_monitoring
from strategy_execution_engine.api.middleware import setup_middleware
from strategy_execution_engine.api.routes import setup_routes
from strategy_execution_engine.api.health import setup_health_routes
from strategy_execution_engine.api.analysis import setup_analysis_routes
from strategy_execution_engine.strategies.strategy_loader import StrategyLoader
from strategy_execution_engine.backtesting.backtester import backtester
from strategy_execution_engine.analysis.performance_analyzer import performance_analyzer

# Initialize logger
logger = get_logger(__name__)

# Global shutdown event
shutdown_event = asyncio.Event()

# Constants
APP_NAME = "Strategy Execution Engine"
APP_VERSION = "0.1.0"

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
        # Initialize core services
        await service_container.initialize()
        logger.info("Service container initialized successfully")

        # Initialize strategy loader
        strategy_loader = StrategyLoader()
        app.state.strategy_loader = strategy_loader
        await strategy_loader.load_strategies()
        logger.info(f"Strategy loader initialized with {len(strategy_loader.get_available_strategies())} strategies")

        # Initialize backtester
        app.state.backtester = backtester
        logger.info("Backtester initialized")

        # Initialize performance analyzer
        app.state.performance_analyzer = performance_analyzer
        logger.info("Performance analyzer initialized")

        # Initialize monitoring
        await setup_monitoring(app)
        logger.info("Monitoring initialized")

        logger.info("Service initialization complete")

        # Yield control back to FastAPI
        yield

        # Cleanup on shutdown
        logger.info("Shutting down services...")

        # Close any open connections or resources
        await service_container.shutdown()
        logger.info("Service container shutdown complete")

    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application
    """
    # Configure logging
    configure_logging()

    # Create FastAPI app
    app = FastAPI(
        title=APP_NAME,
        description="Executes trading strategies and performs backtesting for the Forex Trading Platform",
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

    # Register exception handlers
    app.add_exception_handler(ForexTradingPlatformError, forex_platform_exception_handler)
    app.add_exception_handler(StrategyExecutionError, strategy_execution_exception_handler)
    app.add_exception_handler(StrategyConfigurationError, strategy_configuration_exception_handler)
    app.add_exception_handler(StrategyLoadError, strategy_load_exception_handler)
    app.add_exception_handler(BacktestError, backtest_exception_handler)
    app.add_exception_handler(PydanticValidationError, pydantic_validation_exception_handler)
    app.add_exception_handler(RequestValidationError, fastapi_validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # Setup routes
    setup_routes(app)

    # Setup health routes
    setup_health_routes(app)

    # Setup analysis routes
    setup_analysis_routes(app)

    logger.info("API routes, health checks, and analysis routes configured")

    # Create service container for health checks
    service_container = ServiceContainer()

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

# Exception handlers
async def forex_platform_exception_handler(request: Request, exc: ForexTradingPlatformError) -> JSONResponse:
    """Handle ForexTradingPlatformError exceptions"""
    logger.error(f"ForexTradingPlatformError: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc), "type": exc.__class__.__name__}
    )

async def strategy_execution_exception_handler(request: Request, exc: StrategyExecutionError) -> JSONResponse:
    """Handle StrategyExecutionError exceptions"""
    logger.error(f"StrategyExecutionError: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc), "type": exc.__class__.__name__}
    )

async def strategy_configuration_exception_handler(request: Request, exc: StrategyConfigurationError) -> JSONResponse:
    """Handle StrategyConfigurationError exceptions"""
    logger.error(f"StrategyConfigurationError: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc), "type": exc.__class__.__name__}
    )

async def strategy_load_exception_handler(request: Request, exc: StrategyLoadError) -> JSONResponse:
    """Handle StrategyLoadError exceptions"""
    logger.error(f"StrategyLoadError: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc), "type": exc.__class__.__name__}
    )

async def backtest_exception_handler(request: Request, exc: BacktestError) -> JSONResponse:
    """Handle BacktestError exceptions"""
    logger.error(f"BacktestError: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc), "type": exc.__class__.__name__}
    )

async def pydantic_validation_exception_handler(request: Request, exc: PydanticValidationError) -> JSONResponse:
    """Handle PydanticValidationError exceptions"""
    logger.error(f"ValidationError: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "type": "ValidationError"}
    )

async def fastapi_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle RequestValidationError exceptions"""
    logger.error(f"RequestValidationError: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "type": "RequestValidationError"}
    )

async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle generic exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred", "type": exc.__class__.__name__}
    )

async def main():
    """Main entry point for the Strategy Execution Engine."""
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
        logger.info(f"Starting {APP_NAME} on {settings.host}:{settings.port}")
        await server.serve()

        # Wait for shutdown signal
        await shutdown_event.wait()

        # Graceful shutdown
        logger.info("Initiating graceful shutdown...")
        server.should_exit = True
        await server.shutdown()

    except Exception as e:
        logger.critical(f"Fatal error during startup: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
