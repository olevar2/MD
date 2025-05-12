"""
Health check API endpoints for the ML Integration Service.

This module provides API endpoints for checking the health of the service
and its dependencies.
"""

from typing import Dict, Any, List
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ml_integration_service.di.container import (
    get_model_repository,
    get_feature_service,
    get_data_validator,
    get_reconciliation_service
)
from ml_integration_service.config.reconciliation_config import get_reconciliation_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


class HealthStatus(BaseModel):
    """Health status model."""

    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="Service version")
    components: Dict[str, Dict[str, Any]] = Field(..., description="Component health statuses")
    details: Dict[str, Any] = Field({}, description="Additional health details")


class ComponentHealth(BaseModel):
    """Component health model."""

    name: str = Field(..., description="Component name")
    status: str = Field(..., description="Component health status")
    details: Dict[str, Any] = Field({}, description="Additional component details")


@router.get("", response_model=HealthStatus)
async def get_health() -> Dict[str, Any]:
    """
    Get the health status of the ML Integration Service.

    This endpoint checks the health of the service and its dependencies,
    including the model repository, feature service, and data validator.
    """
    try:
        # Check model repository health
        model_repository = get_model_repository()
        model_repository_health = await check_model_repository_health(model_repository)

        # Check feature service health
        feature_service = get_feature_service()
        feature_service_health = await check_feature_service_health(feature_service)

        # Check data validator health
        data_validator = get_data_validator()
        data_validator_health = await check_data_validator_health(data_validator)

        # Check reconciliation service health
        reconciliation_service = get_reconciliation_service()
        reconciliation_service_health = await check_reconciliation_service_health(reconciliation_service)

        # Determine overall health status
        components = {
            "model_repository": model_repository_health,
            "feature_service": feature_service_health,
            "data_validator": data_validator_health,
            "reconciliation_service": reconciliation_service_health
        }

        status = "healthy"
        for component in components.values():
            if component["status"] != "healthy":
                status = "unhealthy"
                break

        # Get service version
        version = "1.0.0"  # Replace with actual version

        return {
            "status": status,
            "version": version,
            "components": components,
            "details": {
                "environment": "production",  # Replace with actual environment
                "instance_id": "ml-integration-1"  # Replace with actual instance ID
            }
        }
    except Exception as e:
        logger.exception(f"Error checking health: {str(e)}")
        # Don't expose the actual exception details to the client
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while checking service health"
        )


async def check_model_repository_health(model_repository) -> Dict[str, Any]:
    """
    Check the health of the model repository.

    Args:
        model_repository: Model repository instance

    Returns:
        Health status of the model repository
    """
    try:
        # Check if model repository is available
        # This is a placeholder for actual health check logic
        return {
            "status": "healthy",
            "details": {
                "connection": "active",
                "latency_ms": 10
            }
        }
    except Exception as e:
        logger.error(f"Error checking model repository health: {str(e)}")
        return {
            "status": "unhealthy",
            "details": {
                "error": str(e)
            }
        }


async def check_feature_service_health(feature_service) -> Dict[str, Any]:
    """
    Check the health of the feature service.

    Args:
        feature_service: Feature service instance

    Returns:
        Health status of the feature service
    """
    try:
        # Check if feature service is available
        # This is a placeholder for actual health check logic
        return {
            "status": "healthy",
            "details": {
                "connection": "active",
                "latency_ms": 15
            }
        }
    except Exception as e:
        logger.error(f"Error checking feature service health: {str(e)}")
        return {
            "status": "unhealthy",
            "details": {
                "error": str(e)
            }
        }


async def check_data_validator_health(data_validator) -> Dict[str, Any]:
    """
    Check the health of the data validator.

    Args:
        data_validator: Data validator instance

    Returns:
        Health status of the data validator
    """
    try:
        # Check if data validator is available
        # This is a placeholder for actual health check logic
        return {
            "status": "healthy",
            "details": {
                "status": "active"
            }
        }
    except Exception as e:
        logger.error(f"Error checking data validator health: {str(e)}")
        return {
            "status": "unhealthy",
            "details": {
                "error": str(e)
            }
        }


async def check_reconciliation_service_health(reconciliation_service) -> Dict[str, Any]:
    """
    Check the health of the reconciliation service.

    Args:
        reconciliation_service: Reconciliation service instance

    Returns:
        Health status of the reconciliation service
    """
    try:
        # Check if reconciliation service is available
        # This is a placeholder for actual health check logic

        # Get reconciliation configuration
        reconciliation_config = get_reconciliation_config()

        return {
            "status": "healthy",
            "details": {
                "active_reconciliations": len(reconciliation_service.reconciliation_results),
                "max_concurrent_processes": reconciliation_config.max_concurrent_processes,
                "metrics_enabled": reconciliation_config.enable_metrics
            }
        }
    except Exception as e:
        logger.error(f"Error checking reconciliation service health: {str(e)}")
        return {
            "status": "unhealthy",
            "details": {
                "error": str(e)
            }
        }


@router.get("/reconciliation", response_model=ComponentHealth)
async def get_reconciliation_health() -> Dict[str, Any]:
    """
    Get the health status of the reconciliation service.

    This endpoint checks the health of the reconciliation service specifically.
    """
    try:
        # Check reconciliation service health
        reconciliation_service = get_reconciliation_service()
        reconciliation_service_health = await check_reconciliation_service_health(reconciliation_service)

        return {
            "name": "reconciliation_service",
            "status": reconciliation_service_health["status"],
            "details": reconciliation_service_health["details"]
        }
    except Exception as e:
        logger.exception(f"Error checking reconciliation service health: {str(e)}")
        # Don't expose the actual exception details to the client
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while checking reconciliation service health"
        )
