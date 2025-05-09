import logging
"""
Health Check API for Analysis Engine Service.

This module provides API endpoints for health checks, including:
- Liveness probe
- Readiness probe
- Detailed health check
"""

from fastapi import APIRouter, Response, Depends, status
from typing import Dict, Any, Optional

from analysis_engine.monitoring.health_checks import HealthCheck, HealthStatus, ServiceHealth
from analysis_engine.core.container import ServiceContainer
from analysis_engine.monitoring.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

# Create router
router = APIRouter(tags=["Health"])

def get_health_check(service_container: ServiceContainer) -> HealthCheck:
    """
    Get the health check instance.
    
    Args:
        service_container: Service container
        
    Returns:
        HealthCheck instance
    """
    return service_container.health_check

@router.get("/health", response_model=ServiceHealth)
async def health_check(
    health_check: HealthCheck = Depends(get_health_check)
) -> ServiceHealth:
    """
    Get detailed health status of the service.
    
    Returns:
        ServiceHealth object
    """
    return await health_check.check_health()

@router.get("/health/live")
async def liveness_probe() -> Dict[str, str]:
    """
    Liveness probe endpoint.
    
    This endpoint is used by Kubernetes to determine if the service is alive.
    It should return a 200 OK response if the service is running.
    
    Returns:
        Simple status response
    """
    return {"status": "alive"}

@router.get("/health/ready")
async def readiness_probe(
    response: Response,
    health_check: HealthCheck = Depends(get_health_check)
) -> Dict[str, Any]:
    """
    Readiness probe endpoint.
    
    This endpoint is used by Kubernetes to determine if the service is ready
    to receive traffic. It should return a 200 OK response if the service is
    ready, or a 503 Service Unavailable response if the service is not ready.
    
    Args:
        response: FastAPI response
        health_check: Health check instance
        
    Returns:
        Status response
    """
    # Check health
    health_status = await health_check.check_health()
    
    # If service is unhealthy, return 503
    if health_status.status == HealthStatus.UNHEALTHY:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "not ready",
            "reason": "Service is unhealthy",
            "details": {
                "components": [
                    {
                        "name": component.name,
                        "status": component.status,
                        "message": component.message
                    }
                    for component in health_status.components
                    if component.status == HealthStatus.UNHEALTHY
                ],
                "dependencies": [
                    {
                        "name": dependency.name,
                        "status": dependency.status,
                        "message": dependency.message
                    }
                    for dependency in health_status.dependencies
                    if dependency.status == HealthStatus.UNHEALTHY
                ]
            }
        }
    
    # Service is ready
    return {"status": "ready"}

def setup_health_routes(app, service_container: ServiceContainer) -> None:
    """
    Set up health check routes.
    
    Args:
        app: FastAPI application
        service_container: Service container
    """
    # Include router
    app.include_router(router, prefix="/api")
    
    # Add dependency override
    app.dependency_overrides[get_health_check] = lambda: service_container.health_check


# Standardized endpoints
@router.get("/v1/analysis-engine/healths", response_model=ServiceHealth)
async 

@router.get("/v1/analysis-engine/healths/live")
async 

@router.get("/v1/analysis-engine/healths/ready")
async 

# Backward compatibility

@router.get("/health")
async def health_check_legacy(*args, **kwargs):
    """Legacy endpoint for backward compatibility. Use /v1/analysis-engine/healths instead."""
    logger.info(f"Legacy endpoint /health called - consider migrating to /v1/analysis-engine/healths")
    return await health_check(*args, **kwargs)



@router.get("/health/live")
async def liveness_probe_legacy(*args, **kwargs):
    """Legacy endpoint for backward compatibility. Use /v1/analysis-engine/healths/live instead."""
    logger.info(f"Legacy endpoint /health/live called - consider migrating to /v1/analysis-engine/healths/live")
    return await liveness_probe(*args, **kwargs)



@router.get("/health/ready")
async def readiness_probe_legacy(*args, **kwargs):
    """Legacy endpoint for backward compatibility. Use /v1/analysis-engine/healths/ready instead."""
    logger.info(f"Legacy endpoint /health/ready called - consider migrating to /v1/analysis-engine/healths/ready")
    return await readiness_probe(*args, **kwargs)
