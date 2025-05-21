
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

# gRPC specific imports
import grpc.aio
from analysis_engine_service.services.grpc_servicer import AnalysisEngineServicer
# Assuming 'generated_protos' is in PYTHONPATH for this import to work:
from analysis_engine_service.analysis_engine_pb2_grpc import add_AnalysisEngineServiceServicer_to_server
# Import for JWT Interceptor (assuming common-lib is in PYTHONPATH)
from common_lib.security.grpc_interceptors import JwtAuthServerInterceptor

logger = get_logger(__name__) # Already exists, ensure it's fine
GRPC_PORT = 50052 # Define gRPC port
grpc_server_task = None # To hold the asyncio task for the gRPC server
grpc_server_instance = None # To hold the grpc.aio.Server instance

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
    if not hasattr(app.state, 'service_container'
        ) or app.state.service_container is None:
        app.state.service_container = ServiceContainer()
        logger.info('Initialized service container in app state.')
    service_container = app.state.service_container
    try:
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
            logger.warning(
                'Database connection check failed, but continuing startup')
        await service_container.initialize()
        logger.info('Service container initialized successfully')
        if CAUSAL_INFERENCE_AVAILABLE:
            try:
                causal_service = CausalInferenceService()
                app.state.causal_service = causal_service
                logger.info('Causal inference service initialized')
            except Exception as e:
                logger.error(
                    f'Failed to initialize causal inference service: {e}')
        if MULTI_TIMEFRAME_AVAILABLE:
            try:
                multi_timeframe = MultiTimeframeAnalyzer()
                app.state.multi_timeframe_analyzer = multi_timeframe
                logger.info('Multi-timeframe analyzer initialized')
            except Exception as e:
                logger.error(
                    f'Failed to initialize multi-timeframe analyzer: {e}')
        memory_monitor = get_memory_monitor()
        await memory_monitor.start_monitoring()
        logger.info('Memory monitoring started')
        async_monitor = get_async_monitor()
        await async_monitor.start_reporting(interval=300)
        logger.info('Async performance monitoring started')
        await initialize_schedulers(service_container)
        logger.info('Schedulers initialized and started')

        # Start gRPC server
        global grpc_server_task, grpc_server_instance
        
        # Initialize your interceptor
        jwt_interceptor = JwtAuthServerInterceptor(
            # Example placeholder values; replace with actual configuration
            # secret_key="your-analysis-engine-secret",
            # required_audience="analysis-engine-service",
            # issuer="your-auth-issuer"
        )
        interceptors = [jwt_interceptor]
        
        grpc_server_instance = grpc.aio.server(interceptors=interceptors)
        add_AnalysisEngineServiceServicer_to_server(AnalysisEngineServicer(), grpc_server_instance)
        listen_addr = f'[::]:{GRPC_PORT}'
        grpc_server_instance.add_insecure_port(listen_addr)
        logger.info(f"Starting gRPC server on {listen_addr}")
        grpc_server_task = asyncio.create_task(grpc_server_instance.start())
        logger.info("gRPC server startup task created and started.")

        logger.info('Service initialization complete')
    except Exception as e:
        logger.error(f'Error during startup: {e}', exc_info=True)
        # If gRPC server started, try to stop it
        if grpc_server_instance:
            await grpc_server_instance.stop(0)
        if grpc_server_task and not grpc_server_task.done():
            grpc_server_task.cancel()
        raise
    yield
    # Shutdown phase
    try:
        logger.info("Initiating shutdown sequence...")
        # Stop gRPC server first
        global grpc_server_task, grpc_server_instance # Ensure they are accessible
        if grpc_server_instance:
            logger.info("Attempting to stop gRPC server...")
            await grpc_server_instance.stop(5) # 5 seconds grace period
            logger.info("gRPC server stopped.")
        if grpc_server_task and not grpc_server_task.done():
            logger.info("Cancelling gRPC server task...")
            grpc_server_task.cancel()
            try:
                await grpc_server_task
            except asyncio.CancelledError:
                logger.info("gRPC server task cancelled successfully.")
            except Exception as e_task: # pylint: disable=broad-except
                logger.error(f"Error awaiting cancelled gRPC task: {e_task}", exc_info=True)
        
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
    # Similar to trading-gateway-service, adjust sys.path for imports
    # Assumes this script is in analysis-engine-service/core/main.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    service_root = os.path.dirname(current_dir)  # analysis-engine-service directory
    project_root = os.path.dirname(service_root) # Repository root (/app)

    # Path to generated_protos directory (e.g., /app/generated_protos)
    generated_protos_path = os.path.join(project_root, "generated_protos")

    # Add project_root for `from analysis_engine_service...`
    # and generated_protos for proto stubs
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        logger.info(f"Added project root to sys.path: {project_root}")
    if generated_protos_path not in sys.path:
        sys.path.insert(0, generated_protos_path)
        logger.info(f"Added generated_protos to sys.path: {generated_protos_path}")

    # Log the updated sys.path for debugging if needed
    # logger.debug(f"Updated sys.path: {sys.path}")
    
    asyncio.run(main())
