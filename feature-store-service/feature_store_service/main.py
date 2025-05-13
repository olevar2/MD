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
from common_lib.exceptions import ForexTradingPlatformError, DataValidationError, DataFetchError, DataStorageError, DataTransformationError, ServiceError
from feature_store_service.error.error_manager import IndicatorError
from feature_store_service.error.exception_handlers import forex_platform_exception_handler, indicator_error_handler, validation_exception_handler, sqlalchemy_exception_handler, data_validation_exception_handler, data_fetch_exception_handler, data_storage_exception_handler, data_transformation_exception_handler, service_exception_handler, general_exception_handler
from feature_store_service.api.feature_computation_api import feature_computation_router
from feature_store_service.api.indicator_api import indicator_router
from feature_store_service.api.realtime_indicators_api import realtime_indicators_router
from feature_store_service.api.incremental_indicators import router as incremental_indicators_router
from feature_store_service.api.v1.reconciliation_api import router as reconciliation_router
from feature_store_service.api.v1.adapter_api import adapter_router
from feature_store_service.api.metrics_integration import setup_metrics
from feature_store_service.caching.config import CacheConfig
from common_lib.caching import AdaptiveCacheManager, cached, get_cache_manager
from feature_store_service.services.enhanced_indicator_service import EnhancedIndicatorService
from feature_store_service.computation.feature_computation_engine import FeatureComputationEngine
from feature_store_service.indicators.indicator_registry import IndicatorRegistry
from feature_store_service.storage.feature_storage import FeatureStorage
from feature_store_service.services.indicator_manager import initialize_indicator_service
from feature_store_service.db import initialize_database, dispose_database, check_connection
from data_pipeline_service.services.ohlcv_service import get_ohlcv_service
from feature_store_service.adapters.adapter_factory import adapter_factory
configure_logging(service_name='feature-store-service', log_level=os.getenv
    ('LOG_LEVEL', 'INFO'), use_json=os.getenv('LOG_FORMAT', '').lower() ==
    'json', log_file=os.getenv('LOG_FILE'))
logger = get_logger('feature-store-service')
app = FastAPI(title='Forex Trading Platform - Feature Store Service',
    description=
    'Service for calculating, storing, and serving technical indicators and features'
    , version='0.1.0')
app.add_exception_handler(ForexTradingPlatformError,
    forex_platform_exception_handler)
app.add_exception_handler(IndicatorError, indicator_error_handler)
app.add_exception_handler(DataValidationError,
    data_validation_exception_handler)
app.add_exception_handler(DataFetchError, data_fetch_exception_handler)
app.add_exception_handler(DataStorageError, data_storage_exception_handler)
app.add_exception_handler(DataTransformationError,
    data_transformation_exception_handler)
app.add_exception_handler(ServiceError, service_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(ValidationError, validation_exception_handler)
app.add_exception_handler(SQLAlchemyError, sqlalchemy_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)
from feature_store_service.middleware.request_tracking import RequestTrackingMiddleware
app.add_middleware(RequestTrackingMiddleware, exclude_paths=['/health',
    '/metrics', '/docs', '/redoc', '/openapi.json'], request_id_header=
    'X-Request-ID')
app.add_middleware(FastAPICorrelationIdMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=
    True, allow_methods=['*'], allow_headers=['*'])
setup_metrics(app, service_name='feature-store-service')
indicator_registry = IndicatorRegistry()
feature_storage = FeatureStorage()
cache_config_path = os.path.join(os.path.dirname(__file__), '..', 'config',
    'cache_config.json')
cache_config = CacheConfig.load_config(cache_config_path)
logger.info(f'Loaded cache configuration from {cache_config_path}')
cache_manager = CacheManager(cache_config)
logger.info(
    f"Initialized cache manager with {cache_config_manager.get('memory_cache_size', 0) / 1024 / 1024:.1f}MB memory cache"
    )
enhanced_indicator_service = EnhancedIndicatorService(config=cache_config)
logger.info('Initialized enhanced indicator service with caching support')
feature_computation_engine = FeatureComputationEngine(indicator_registry=
    indicator_registry, feature_storage=feature_storage,
    enhanced_indicator_service=enhanced_indicator_service)
app.include_router(indicator_router)
app.include_router(feature_computation_router)
app.include_router(realtime_indicators_router)
app.include_router(incremental_indicators_router)
app.include_router(reconciliation_router, prefix='/api/v1')
app.include_router(adapter_router)
from common_lib.correlation import FastAPICorrelationIdMiddleware
from fastapi import APIRouter
cache_router = APIRouter(prefix='/api/v1/cache', tags=['Cache'])


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@cache_router.get('/stats')
async def get_cache_stats():
    """Get current cache statistics."""
    return enhanced_indicator_service.get_cache_stats()


@cache_router.post('/clear/{symbol}')
async def clear_cache_for_symbol(symbol: str):
    """Clear cache for a specific symbol."""
    cleared_count = enhanced_indicator_service.clear_cache_for_symbol(symbol)
    return {'message':
        f'Cleared {cleared_count} cache entries for symbol {symbol}'}


@cache_router.post('/clear/indicator/{indicator_type}')
async def clear_cache_for_indicator(indicator_type: str):
    """Clear cache for a specific indicator type."""
    cleared_count = enhanced_indicator_service.clear_cache_for_indicator(
        indicator_type)
    return {'message':
        f'Cleared {cleared_count} cache entries for indicator {indicator_type}'
        }


@cache_router.post('/clear/all')
async def clear_all_cache():
    """Clear all cache entries."""
    return await cache_manager.clear()


app.include_router(cache_router)
health_check = add_health_check_to_app(app=app, service_name=
    'feature-store-service', version='0.1.0', checks=[{'name': 'database',
    'check_func': check_connection, 'critical': True}, {'name':
    'indicators', 'check_func': lambda : len(indicator_registry.
    get_all_indicators()) > 0, 'critical': True}, {'name':
    'adapter_factory', 'check_func': lambda : adapter_factory._initialized,
    'critical': True}])


async def get_enhanced_indicator_service() ->EnhancedIndicatorService:
    """
    Get enhanced indicator service.
    
    Returns:
        EnhancedIndicatorService: Description of return value
    
    """

    service = EnhancedIndicatorService(registry=indicator_registry,
        cache_manager=cache_manager, storage=feature_storage, config_loader
        =config_loader)
    return service


@app.on_event('startup')
@async_with_exception_handling
async def startup_event():
    """
    Startup event.
    
    """

    logger.info('Starting Feature Store Service')
    try:
        await initialize_database()
        logger.info('Database connection initialized')
        indicator_registry.register_all_indicators()
        from feature_store_service.indicators.advanced_indicators_registrar import register_advanced_indicators
        register_advanced_indicators(indicator_registry)
        logger.info(
            f'Registered {len(indicator_registry.get_all_indicators())} indicators'
            )
        await feature_storage.initialize()
        logger.info('Feature storage initialized')
        ohlcv_service = await get_ohlcv_service()
        await initialize_indicator_service(feature_storage, ohlcv_service)
        logger.info('Incremental indicator service initialized')
        from feature_store_service.api.indicator_api import set_indicator_registry
        set_indicator_registry(indicator_registry, enhanced_indicator_service)
        logger.info('Registered enhanced indicator service with API layer')
        adapter_factory.initialize()
        logger.info('Adapter factory initialized')
    except Exception as e:
        logger.error(f'Failed to initialize service: {str(e)}')
        raise


@app.on_event('shutdown')
@async_with_exception_handling
async def shutdown_event():
    """
    Shutdown event.
    
    """

    logger.info('Shutting down Feature Store Service')
    try:
        await feature_storage.close()
        await dispose_database()
        logger.info('Database connections closed')
        adapter_factory.cleanup()
        logger.info('Adapter factory cleaned up')
    except Exception as e:
        logger.error(f'Error during service shutdown: {str(e)}')
