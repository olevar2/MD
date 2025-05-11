"""
Feature Store Service Main Entry Point.

This module initializes and runs the feature store service for the Forex Trading Platform.
"""

import logging
import os
from typing import Dict, List

from common_lib.correlation import FastAPICorrelationIdMiddleware
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from prometheus_client import make_asgi_app

from core_foundations.api.health_check import add_health_check_to_app
from feature_store_service.logging.enhanced_logging import configure_logging, get_logger

from common_lib.exceptions import (
    ForexTradingPlatformError,
    DataValidationError,
    DataFetchError,
    DataStorageError,
    DataTransformationError,
    ServiceError
)
from feature_store_service.error.error_manager import IndicatorError
from feature_store_service.error.exception_handlers import (
    forex_platform_exception_handler,
    indicator_error_handler,
    validation_exception_handler,
    sqlalchemy_exception_handler,
    data_validation_exception_handler,
    data_fetch_exception_handler,
    data_storage_exception_handler,
    data_transformation_exception_handler,
    service_exception_handler,
    general_exception_handler,
)

from feature_store_service.api.feature_computation_api import feature_computation_router
from feature_store_service.api.indicator_api import indicator_router
from feature_store_service.api.realtime_indicators_api import realtime_indicators_router
from feature_store_service.api.incremental_indicators import router as incremental_indicators_router
from feature_store_service.api.v1.reconciliation_api import router as reconciliation_router
from feature_store_service.api.v1.adapter_api import adapter_router
from feature_store_service.api.metrics_integration import setup_metrics
from feature_store_service.caching.config import CacheConfig
from feature_store_service.caching.cache_manager import CacheManager
from feature_store_service.services.enhanced_indicator_service import EnhancedIndicatorService
from feature_store_service.computation.feature_computation_engine import FeatureComputationEngine
from feature_store_service.indicators.indicator_registry import IndicatorRegistry
from feature_store_service.storage.feature_storage import FeatureStorage
from feature_store_service.services.indicator_manager import initialize_indicator_service
from feature_store_service.db import initialize_database, dispose_database, check_connection
from data_pipeline_service.services.ohlcv_service import get_ohlcv_service
from feature_store_service.adapters.adapter_factory import adapter_factory

# Initialize enhanced logging
configure_logging(
    service_name="feature-store-service",
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    use_json=os.getenv("LOG_FORMAT", "").lower() == "json",
    log_file=os.getenv("LOG_FILE")
)

# Get logger instance
logger = get_logger("feature-store-service")

# Create FastAPI application
app = FastAPI(
    title="Forex Trading Platform - Feature Store Service",
    description="Service for calculating, storing, and serving technical indicators and features",
    version="0.1.0",
)

# Register exception handlers
app.add_exception_handler(ForexTradingPlatformError, forex_platform_exception_handler)
app.add_exception_handler(IndicatorError, indicator_error_handler)
app.add_exception_handler(DataValidationError, data_validation_exception_handler)
app.add_exception_handler(DataFetchError, data_fetch_exception_handler)
app.add_exception_handler(DataStorageError, data_storage_exception_handler)
app.add_exception_handler(DataTransformationError, data_transformation_exception_handler)
app.add_exception_handler(ServiceError, service_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(ValidationError, validation_exception_handler)
app.add_exception_handler(SQLAlchemyError, sqlalchemy_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Add middleware
from feature_store_service.middleware.request_tracking import RequestTrackingMiddleware

# Add request tracking middleware (should be first to track all requests)
app.add_middleware(
    RequestTrackingMiddleware,
    exclude_paths=["/health", "/metrics", "/docs", "/redoc", "/openapi.json"],
    request_id_header="X-Request-ID"
)

# Add correlation ID middleware
app.add_middleware(FastAPICorrelationIdMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up metrics
setup_metrics(app, service_name="feature-store-service")

# Initialize components
indicator_registry = IndicatorRegistry()
feature_storage = FeatureStorage()

# Initialize cache system from config file
cache_config_path = os.path.join(os.path.dirname(__file__), "..", "config", "cache_config.json")
cache_config = CacheConfig.load_config(cache_config_path)
logger.info(f"Loaded cache configuration from {cache_config_path}")

# Create cache manager
cache_manager = CacheManager(cache_config)
logger.info(f"Initialized cache manager with {cache_config.get('memory_cache_size', 0)/1024/1024:.1f}MB memory cache")

# Create enhanced indicator service with caching capability
enhanced_indicator_service = EnhancedIndicatorService(config=cache_config)
logger.info("Initialized enhanced indicator service with caching support")

# Initialize feature computation engine
feature_computation_engine = FeatureComputationEngine(
    indicator_registry=indicator_registry,
    feature_storage=feature_storage,
    enhanced_indicator_service=enhanced_indicator_service  # Injecting the enhanced service
)

# Register API routers with caching
app.include_router(indicator_router)
app.include_router(feature_computation_router)
app.include_router(realtime_indicators_router)
app.include_router(incremental_indicators_router)
app.include_router(reconciliation_router, prefix="/api/v1")
app.include_router(adapter_router)

# Add a cache stats endpoint
from common_lib.correlation import FastAPICorrelationIdMiddleware
from fastapi import APIRouter
cache_router = APIRouter(prefix="/api/v1/cache", tags=["Cache"])

@cache_router.get("/stats")
async def get_cache_stats():
    """Get current cache statistics."""
    return enhanced_indicator_service.get_cache_stats()

@cache_router.post("/clear/{symbol}")
async def clear_cache_for_symbol(symbol: str):
    """Clear cache for a specific symbol."""
    cleared_count = enhanced_indicator_service.clear_cache_for_symbol(symbol)
    return {"message": f"Cleared {cleared_count} cache entries for symbol {symbol}"}

@cache_router.post("/clear/indicator/{indicator_type}")
async def clear_cache_for_indicator(indicator_type: str):
    """Clear cache for a specific indicator type."""
    cleared_count = enhanced_indicator_service.clear_cache_for_indicator(indicator_type)
    return {"message": f"Cleared {cleared_count} cache entries for indicator {indicator_type}"}

@cache_router.post("/clear/all")
async def clear_all_cache():
    """Clear all cache entries."""
    return await cache_manager.clear()

app.include_router(cache_router)

# Add health check endpoint
health_check = add_health_check_to_app(
    app=app,
    service_name="feature-store-service",
    version="0.1.0",
    checks=[
        {
            "name": "database",
            "check_func": check_connection,  # Use centralized check_connection function
            "critical": True,
        },
        {
            "name": "indicators",
            "check_func": lambda: len(indicator_registry.get_all_indicators()) > 0,
            "critical": True,
        },
        {
            "name": "adapter_factory",
            "check_func": lambda: adapter_factory._initialized,
            "critical": True,
        },
    ],
)

# Dependency Injection Setup (Example using FastAPI)
async def get_enhanced_indicator_service(
    # ... existing dependencies ...
) -> EnhancedIndicatorService:
    # ... existing setup ...
    # Ensure EnhancedIndicatorService is initialized correctly without CacheAwareIndicatorService
    # service = EnhancedIndicatorService(indicator_registry, cache_manager, storage, config_loader)
    # The EnhancedIndicatorService now takes cache_manager directly
    service = EnhancedIndicatorService(
        registry=indicator_registry, # Assuming registry is available
        cache_manager=cache_manager, # Assuming cache_manager is available
        storage=feature_storage, # Assuming feature_storage is available
        config_loader=config_loader # Assuming config_loader is available
    )
    return service

# Startup event to initialize components
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Feature Store Service")
    try:
        # Initialize centralized database first
        await initialize_database()
        logger.info("Database connection initialized")

        # Initialize indicator registry with all available indicators
        indicator_registry.register_all_indicators()

        # Register advanced technical indicators
        from feature_store_service.indicators.advanced_indicators_registrar import register_advanced_indicators
        register_advanced_indicators(indicator_registry)

        logger.info(f"Registered {len(indicator_registry.get_all_indicators())} indicators")

        # Initialize storage (now using centralized database)
        await feature_storage.initialize()
        logger.info("Feature storage initialized")

        # Initialize incremental indicator service
        ohlcv_service = await get_ohlcv_service()
        await initialize_indicator_service(feature_storage, ohlcv_service)
        logger.info("Incremental indicator service initialized")

        # Register the enhanced indicator service with the API
        from feature_store_service.api.indicator_api import set_indicator_registry
        set_indicator_registry(indicator_registry, enhanced_indicator_service)
        logger.info("Registered enhanced indicator service with API layer")

        # Initialize adapter factory
        adapter_factory.initialize()
        logger.info("Adapter factory initialized")

    except Exception as e:
        logger.error(f"Failed to initialize service: {str(e)}")
        raise

# Shutdown event for cleanup
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Feature Store Service")
    try:
        await feature_storage.close()
        # Dispose centralized database
        await dispose_database()
        logger.info("Database connections closed")

        # Clean up adapter factory
        adapter_factory.cleanup()
        logger.info("Adapter factory cleaned up")
    except Exception as e:
        logger.error(f"Error during service shutdown: {str(e)}")