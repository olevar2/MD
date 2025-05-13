"""
Health Check API for Strategy Execution Engine

This module provides health check endpoints for the Strategy Execution Engine.
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, FastAPI, Depends, HTTPException, status
from strategy_execution_engine.core.container import ServiceContainer
logger = logging.getLogger(__name__)
health_router = APIRouter(tags=['health'])


from strategy_execution_engine.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@health_router.get('/health')
async def health_check():
    """
    Health check endpoint.

    Returns:
        Dict: Health status information
    """
    return {'status': 'healthy', 'version': '0.1.0', 'timestamp': datetime.
        utcnow().isoformat()}


@health_router.get('/health/live')
async def liveness_probe():
    """
    Kubernetes liveness probe endpoint.

    This endpoint is used by Kubernetes to determine if the service is alive.
    It should return a 200 OK response if the service is running.

    Returns:
        Dict: Liveness status
    """
    return {'status': 'alive', 'timestamp': datetime.utcnow().isoformat()}


@health_router.get('/health/ready')
async def readiness_probe(service_container: ServiceContainer=Depends()):
    """
    Kubernetes readiness probe endpoint.

    This endpoint is used by Kubernetes to determine if the service is ready
    to receive traffic. It should return a 200 OK response if the service is
    ready, or a 503 Service Unavailable response if the service is not ready.

    Args:
        service_container: Service container with dependencies

    Returns:
        Dict: Readiness status
    """
    health_status = await detailed_health_check(service_container)
    if health_status['status'] != 'healthy':
        return {'status': 'not ready', 'reason': 'Service is not healthy',
            'timestamp': datetime.utcnow().isoformat()}
    return {'status': 'ready', 'timestamp': datetime.utcnow().isoformat()}


@health_router.get('/health/detailed')
@async_with_exception_handling
async def detailed_health_check(service_container: ServiceContainer=Depends()):
    """
    Detailed health check endpoint.

    Args:
        service_container: Service container with dependencies

    Returns:
        Dict: Detailed health status information
    """
    db_status = 'healthy'
    db_message = 'No database connection required'
    services_status = {}
    try:
        analysis_engine_status = await check_analysis_engine_connection(
            service_container)
        services_status['analysis_engine'] = analysis_engine_status
        feature_store_status = await check_feature_store_connection(
            service_container)
        services_status['feature_store'] = feature_store_status
        trading_gateway_status = await check_trading_gateway_connection(
            service_container)
        services_status['trading_gateway'] = trading_gateway_status
        strategy_loader_status = check_strategy_loader(service_container)
        services_status['strategy_loader'] = strategy_loader_status
        backtester_status = check_backtester(service_container)
        services_status['backtester'] = backtester_status
        overall_status = 'healthy'
        for service, status in services_status.items():
            if status['status'] != 'healthy':
                overall_status = 'degraded'
                break
    except Exception as e:
        logger.error(f'Error during health check: {e}', exc_info=True)
        overall_status = 'unhealthy'
        services_status['error'] = str(e)
    return {'status': overall_status, 'version': '0.1.0', 'timestamp':
        datetime.utcnow().isoformat(), 'database': {'status': db_status,
        'message': db_message}, 'services': services_status}


@async_with_exception_handling
async def check_analysis_engine_connection(service_container: ServiceContainer
    ) ->Dict[str, Any]:
    """
    Check connection to Analysis Engine service.

    Args:
        service_container: Service container with dependencies

    Returns:
        Dict: Connection status
    """
    try:
        if service_container.has('analysis_engine_client'):
            client = service_container.get('analysis_engine_client')
        else:
            from strategy_execution_engine.clients.analysis_engine_client import AnalysisEngineClient
            client = AnalysisEngineClient()
        health_status = await client.check_health()
        return health_status
    except Exception as e:
        logger.warning(f'Analysis Engine connection check failed: {e}')
        return {'status': 'unhealthy', 'message': str(e)}


@async_with_exception_handling
async def check_feature_store_connection(service_container: ServiceContainer
    ) ->Dict[str, Any]:
    """
    Check connection to Feature Store service.

    Args:
        service_container: Service container with dependencies

    Returns:
        Dict: Connection status
    """
    try:
        if service_container.has('feature_store_client'):
            client = service_container.get('feature_store_client')
        else:
            from strategy_execution_engine.clients.feature_store_client import FeatureStoreClient
            client = FeatureStoreClient()
        health_status = await client.check_health()
        return health_status
    except Exception as e:
        logger.warning(f'Feature Store connection check failed: {e}')
        return {'status': 'unhealthy', 'message': str(e)}


@async_with_exception_handling
async def check_trading_gateway_connection(service_container: ServiceContainer
    ) ->Dict[str, Any]:
    """
    Check connection to Trading Gateway service.

    Args:
        service_container: Service container with dependencies

    Returns:
        Dict: Connection status
    """
    try:
        if service_container.has('trading_gateway_client'):
            client = service_container.get('trading_gateway_client')
        else:
            from strategy_execution_engine.clients.trading_gateway_client import TradingGatewayClient
            client = TradingGatewayClient()
        health_status = await client.check_health()
        return health_status
    except Exception as e:
        logger.warning(f'Trading Gateway connection check failed: {e}')
        return {'status': 'unhealthy', 'message': str(e)}


@with_exception_handling
def check_strategy_loader(service_container: ServiceContainer) ->Dict[str, Any
    ]:
    """
    Check strategy loader status.

    Args:
        service_container: Service container with dependencies

    Returns:
        Dict: Strategy loader status
    """
    try:
        if service_container.has('strategy_loader'):
            strategy_loader = service_container.get('strategy_loader')
            strategies = strategy_loader.get_available_strategies()
            strategies_count = len(strategies) if strategies else 0
            return {'status': 'healthy', 'message':
                'Strategy loader operational', 'strategies_loaded':
                strategies_count}
        else:
            return {'status': 'degraded', 'message':
                'Strategy loader not initialized'}
    except Exception as e:
        logger.warning(f'Strategy loader check failed: {e}')
        return {'status': 'unhealthy', 'message': str(e)}


@with_exception_handling
def check_backtester(service_container: ServiceContainer) ->Dict[str, Any]:
    """
    Check backtester status.

    Args:
        service_container: Service container with dependencies

    Returns:
        Dict: Backtester status
    """
    try:
        if service_container.has('backtester'):
            from strategy_execution_engine.core.config import get_settings
            import os
            settings = get_settings()
            backtest_data_dir = settings.backtest_data_dir
            if os.path.exists(backtest_data_dir) and os.path.isdir(
                backtest_data_dir):
                return {'status': 'healthy', 'message':
                    'Backtester operational', 'data_dir': backtest_data_dir}
            else:
                return {'status': 'degraded', 'message':
                    f'Backtester data directory not found: {backtest_data_dir}'
                    }
        else:
            return {'status': 'degraded', 'message':
                'Backtester not initialized'}
    except Exception as e:
        logger.warning(f'Backtester check failed: {e}')
        return {'status': 'unhealthy', 'message': str(e)}


def setup_health_routes(app: FastAPI) ->None:
    """
    Set up health check routes for the application.

    Args:
        app: FastAPI application instance
    """
    app.include_router(health_router)
    logger.info('Health check routes configured')
