
from .monitoring import setup_monitoring
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
from analysis_engine.config import AnalysisEngineSettings as Settings, get_settings
from analysis_engine.core.exceptions_bridge import ForexTradingPlatformError, DataValidationError, DataFetchError as CommonDataFetchError, DataStorageError as CommonDataStorageError, DataTransformationError as CommonDataTransformationError, ConfigurationError as CommonConfigurationError, ServiceError, ServiceUnavailableError as CommonServiceUnavailableError, ServiceTimeoutError as CommonServiceTimeoutError, ModelError as CommonModelError, ModelTrainingError as CommonModelTrainingError, ModelPredictionError as CommonModelPredictionError
from analysis_engine.core.errors import AnalysisEngineError, ValidationError, DataFetchError, AnalysisError, ConfigurationError, ServiceUnavailableError, forex_platform_exception_handler, analysis_engine_exception_handler, validation_exception_handler, data_fetch_exception_handler, analysis_exception_handler, configuration_exception_handler, service_unavailable_exception_handler, pydantic_validation_exception_handler, fastapi_validation_exception_handler, data_validation_exception_handler, common_data_fetch_exception_handler, data_storage_exception_handler, data_transformation_exception_handler, common_configuration_exception_handler, service_error_exception_handler, common_service_unavailable_exception_handler, service_timeout_exception_handler, model_error_exception_handler, model_training_exception_handler, model_prediction_exception_handler, generic_exception_handler
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
from analysis_engine.core.grpc_server import serve as grpc_serve

try:
    from analysis_engine.causal.services.causal_inference_service import CausalInferenceService
    CAUSAL_INFERENCE_AVAILABLE = True
except ImportError:
    CAUSAL_INFERENCE_AVAILABLE = False
    logger.warning('Causal inference service not available')
try:
    from analysis_engine.analysis.multi_timeframe.multi_timeframe_analyzer import MultiTimeframeAnalyzer
    MULTI_TIMEFRAME_AVAILABLE = True
except ImportError:
    MULTI_TIMEFRAME_AVAILABLE = False
    logger.warning('Multi-timeframe analyzer not available')
logger = get_logger(__name__)
shutdown_event = asyncio.Event()
APP_NAME = 'Analysis Engine Service'
APP_VERSION = '1.0.0'


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@asynccontextmanager
@async_with_exception_handling
async def lifespan(app: FastAPI):
    """
    Manage the application lifecycle, handling setup and teardown of services.

    Args:
        app: FastAPI application instance
    """
    if not hasattr(app.state, 'service_container') or app.state.service_container is None:
        app.state.service_container = ServiceContainer()
        logger.info('Initialized service container in app state.')
    
    app.state.grpc_server_task = None
    service_container = app.state.service_container

    try:
        # Start gRPC server
        logger.info("Attempting to start gRPC server...")
        # Pass the service_container to the grpc_serve function
        app.state.grpc_server_task = asyncio.create_task(grpc_serve(service_container=app.state.service_container))
        logger.info("gRPC server task created.")

        service_template = app.state.service_template
        await service_template.startup()
        logger.info('Service template started')
        
        from analysis_engine.db.connection import initialize_database, initialize_async_database, check_async_db_connection
        initialize_database()
        logger.info('Synchronous database initialized')
        await initialize_async_database()
        logger.info('Asynchronous database initialized')
        
        db_connected = await check_async_db_connection()
        if db_connected:
            logger.info('Database connection verified')
        else:
            logger.warning('Database connection check failed, but continuing startup')
            
        await service_container.initialize()
        logger.info('Service container initialized successfully')
        
        if CAUSAL_INFERENCE_AVAILABLE:
            try:
                causal_service = CausalInferenceService()
                app.state.causal_service = causal_service
                logger.info('Causal inference service initialized')
            except Exception as e:
                logger.error(f'Failed to initialize causal inference service: {e}')
                
        if MULTI_TIMEFRAME_AVAILABLE:
            try:
                multi_timeframe = MultiTimeframeAnalyzer()
                app.state.multi_timeframe_analyzer = multi_timeframe
                logger.info('Multi-timeframe analyzer initialized')
            except Exception as e:
                logger.error(f'Failed to initialize multi-timeframe analyzer: {e}')
                
        memory_monitor = get_memory_monitor()
        await memory_monitor.start_monitoring()
        logger.info('Memory monitoring started')
        
        async_monitor = get_async_monitor()
        await async_monitor.start_reporting(interval=300)
        logger.info('Async performance monitoring started')
        
        await initialize_schedulers(service_container)
        logger.info('Schedulers initialized and started')
        
        logger.info('Service initialization complete')
        
    except Exception as e:
        logger.error(f'Error during startup: {e}', exc_info=True)
        if app.state.grpc_server_task and not app.state.grpc_server_task.done():
            logger.info("Cancelling gRPC server task due to startup error...")
            app.state.grpc_server_task.cancel()
            try:
                await app.state.grpc_server_task
            except asyncio.CancelledError:
                logger.info("gRPC server task cancelled successfully.")
            except Exception as grpc_cancel_err:
                logger.error(f"Error cancelling gRPC server task: {grpc_cancel_err}", exc_info=True)
        raise
        
    yield
    
    logger.info("Initiating application shutdown...")
    try:
        service_template = app.state.service_template
        await service_template.shutdown()
        logger.info('Service template shut down')
        
        if hasattr(app.state, 'service_container'):
            await cleanup_schedulers(app.state.service_container)
            logger.info('Schedulers stopped')
            
        if hasattr(app.state, 'service_container'):
            await app.state.service_container.cleanup()
            
        memory_monitor = get_memory_monitor()
        await memory_monitor.stop_monitoring()
        logger.info('Memory monitoring stopped')
        
        async_monitor = get_async_monitor()
        await async_monitor.stop_reporting()
        logger.info('Async performance monitoring stopped')
        
        from analysis_engine.db.connection import dispose_database, dispose_async_database
        dispose_database()
        logger.info('Synchronous database disposed')
        await dispose_async_database()
        logger.info('Asynchronous database disposed')
        
        if app.state.grpc_server_task and not app.state.grpc_server_task.done():
            logger.info("Attempting to stop gRPC server...")
            app.state.grpc_server_task.cancel()
            try:
                await asyncio.wait_for(app.state.grpc_server_task, timeout=10.0)
                logger.info("gRPC server task awaited successfully.")
            except asyncio.CancelledError:
                logger.info("gRPC server task was cancelled.")
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for gRPC server to shut down.")
            except Exception as e_grpc:
                logger.error(f"Error stopping gRPC server: {e_grpc}", exc_info=True)
        else:
            logger.info("gRPC server task was not running or already completed.")
            
        logger.info('Service shutdown complete')
        
    except Exception as e:
        logger.error(f'Error during shutdown: {e}', exc_info=True)


def create_app() ->FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application
    """
    configure_logging()
    configure_structured_logging()
    app = FastAPI(title=APP_NAME, description=

setup_monitoring(app)
        'Provides analytical capabilities for the trading platform',
        version=APP_VERSION, lifespan=lifespan, debug=get_settings().
        debug_mode, docs_url='/api/docs', redoc_url='/api/redoc',
        openapi_url='/api/openapi.json')
    from analysis_engine.core.service_template import get_service_template
    service_template = get_service_template(app)
    service_template.initialize()
    logger.info('Service template initialized')
    setup_middleware(app)
    setup_metrics(app, service_name='analysis-engine-service')
    from analysis_engine.core.error_middleware import add_error_handling_middleware
    add_error_handling_middleware(app, include_traceback=get_settings().
        debug_mode)
    logger.info('Standardized error handling middleware configured')
    setup_routes(app)
    service_container = ServiceContainer()
    setup_health_routes(app)
    logger.info('API routes and health checks configured')
    app.state.service_container = service_container
    app.state.service_template = service_template
    return app


async def handle_shutdown(signal_name: str) ->None:
    """Handle shutdown signals gracefully"""
    logger.info(
        f'Received {signal_name} signal. Initiating graceful shutdown...')
    shutdown_event.set()


def setup_signal_handlers() ->None:
    """Setup signal handlers for graceful shutdown"""
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(
            handle_shutdown(signal.Signals(s).name)))


@async_with_exception_handling
async def main():
    """Main entry point for the Analysis Engine Service."""
    try:
        app = create_app()
        settings = get_settings()
        setup_signal_handlers()
        config = uvicorn.Config(app=app, host=settings.host, port=settings.
            port, log_level=settings.log_level.lower(), reload=settings.
            debug_mode)
        server = uvicorn.Server(config)
        logger.info(
            f'Starting Analysis Engine Service on {settings.host}:{settings.port}'
            )
        await server.serve()
        await shutdown_event.wait()
        logger.info('Initiating graceful shutdown...')
        server.should_exit = True
        await server.shutdown()
    except Exception as e:
        logger.error(f'Error starting Analysis Engine Service: {e}',
            exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
