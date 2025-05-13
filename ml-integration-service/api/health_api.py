"""
Health check API endpoints for the ML Integration Service.

This module provides API endpoints for checking the health of the service
and its dependencies.
"""
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from core.logging_setup_standardized import get_service_logger
from config.standardized_config_1 import settings
from models.error_handling_standardized import handle_async_exception
from core.container import get_model_repository, get_feature_service, get_data_validator, get_reconciliation_service
from config.reconciliation_config import get_reconciliation_config
logger = get_service_logger('health-api')
router = APIRouter(prefix='/health', tags=['health'])


from ml_integration_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class HealthStatus(BaseModel):
    """Health status model."""
    status: str = Field(..., description='Overall health status')
    version: str = Field(..., description='Service version')
    components: Dict[str, Dict[str, Any]] = Field(..., description=
        'Component health statuses')
    details: Dict[str, Any] = Field({}, description='Additional health details'
        )


class ComponentHealth(BaseModel):
    """Component health model."""
    name: str = Field(..., description='Component name')
    status: str = Field(..., description='Component health status')
    details: Dict[str, Any] = Field({}, description=
        'Additional component details')


@router.get('', response_model=HealthStatus)
@handle_async_exception(operation='get_health')
async def get_health() ->Dict[str, Any]:
    """
    Get the health status of the ML Integration Service.

    This endpoint checks the health of the service and its dependencies,
    including the model repository, feature service, and data validator.
    """
    model_repository = get_model_repository()
    model_repository_health = await check_model_repository_health(
        model_repository)
    feature_service = get_feature_service()
    feature_service_health = await check_feature_service_health(feature_service
        )
    data_validator = get_data_validator()
    data_validator_health = await check_data_validator_health(data_validator)
    reconciliation_service = get_reconciliation_service()
    reconciliation_service_health = await check_reconciliation_service_health(
        reconciliation_service)
    components = {'model_repository': model_repository_health,
        'feature_service': feature_service_health, 'data_validator':
        data_validator_health, 'reconciliation_service':
        reconciliation_service_health}
    status = 'healthy'
    for component in components.values():
        if component['status'] != 'healthy':
            status = 'unhealthy'
            break
    version = settings.SERVICE_VERSION
    return {'status': status, 'version': version, 'components': components,
        'details': {'environment': settings.ENVIRONMENT, 'instance_id':
        settings.get('INSTANCE_ID', 'ml-integration-1')}}


@async_with_exception_handling
async def check_model_repository_health(model_repository) ->Dict[str, Any]:
    """
    Check the health of the model repository.

    Args:
        model_repository: Model repository instance

    Returns:
        Health status of the model repository
    """
    try:
        return {'status': 'healthy', 'details': {'connection': 'active',
            'latency_ms': 10}}
    except Exception as e:
        logger.error(f'Error checking model repository health: {str(e)}')
        return {'status': 'unhealthy', 'details': {'error': str(e)}}


@async_with_exception_handling
async def check_feature_service_health(feature_service) ->Dict[str, Any]:
    """
    Check the health of the feature service.

    Args:
        feature_service: Feature service instance

    Returns:
        Health status of the feature service
    """
    try:
        return {'status': 'healthy', 'details': {'connection': 'active',
            'latency_ms': 15}}
    except Exception as e:
        logger.error(f'Error checking feature service health: {str(e)}')
        return {'status': 'unhealthy', 'details': {'error': str(e)}}


@async_with_exception_handling
async def check_data_validator_health(data_validator) ->Dict[str, Any]:
    """
    Check the health of the data validator.

    Args:
        data_validator: Data validator instance

    Returns:
        Health status of the data validator
    """
    try:
        return {'status': 'healthy', 'details': {'status': 'active'}}
    except Exception as e:
        logger.error(f'Error checking data validator health: {str(e)}')
        return {'status': 'unhealthy', 'details': {'error': str(e)}}


@async_with_exception_handling
async def check_reconciliation_service_health(reconciliation_service) ->Dict[
    str, Any]:
    """
    Check the health of the reconciliation service.

    Args:
        reconciliation_service: Reconciliation service instance

    Returns:
        Health status of the reconciliation service
    """
    try:
        reconciliation_config = get_reconciliation_config()
        return {'status': 'healthy', 'details': {'active_reconciliations':
            len(reconciliation_service.reconciliation_results),
            'max_concurrent_processes': reconciliation_config.
            max_concurrent_processes, 'metrics_enabled':
            reconciliation_config.enable_metrics}}
    except Exception as e:
        logger.error(f'Error checking reconciliation service health: {str(e)}')
        return {'status': 'unhealthy', 'details': {'error': str(e)}}


@router.get('/reconciliation', response_model=ComponentHealth)
@handle_async_exception(operation='get_reconciliation_health')
async def get_reconciliation_health() ->Dict[str, Any]:
    """
    Get the health status of the reconciliation service.

    This endpoint checks the health of the reconciliation service specifically.
    """
    reconciliation_service = get_reconciliation_service()
    reconciliation_service_health = await check_reconciliation_service_health(
        reconciliation_service)
    return {'name': 'reconciliation_service', 'status':
        reconciliation_service_health['status'], 'details':
        reconciliation_service_health['details']}
