"""
Health Check API for Analysis Engine Service.

This module provides API endpoints for health checks, including:
- Liveness probe
- Readiness probe
- Detailed health check

All endpoints follow the platform's standardized API design patterns.
"""

from fastapi import APIRouter, Response, Depends, status, HTTPException
from typing import Dict, Any, Optional

from analysis_engine.monitoring.health_checks import HealthCheck, HealthStatus, ServiceHealth
from analysis_engine.core.container import ServiceContainer
from analysis_engine.monitoring.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

# Create router with standardized prefix
router = APIRouter(
    prefix="/v1/analysis/health-checks",
    tags=["Health"]
)

def get_health_check(service_container: ServiceContainer) -> HealthCheck:
    """
    Get the health check instance.
    
    Args:
        service_container: Service container
        
    Returns:
        HealthCheck instance
    """
    return service_container.health_check

@router.get(
    "",
    response_model=ServiceHealth,
    summary="Get detailed health status",
    description="Get detailed health status of the service including all components and dependencies."
)
async def health_check(
    health_check: HealthCheck = Depends(get_health_check)
) -> ServiceHealth:
    """
    Get detailed health status of the service.
    
    Returns:
        ServiceHealth object containing detailed health information
    
    Raises:
        HTTPException: If there's an error checking health
    """
    try:
        return await health_check.check_health()
    except Exception as e:
        logger.error(f"Error checking health: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/liveness",
    summary="Liveness probe",
    description="Kubernetes liveness probe endpoint to determine if the service is alive."
)
async def liveness_probe() -> Dict[str, str]:
    """
    Liveness probe endpoint.
    
    This endpoint is used by Kubernetes to determine if the service is alive.
    It should return a 200 OK response if the service is running.
    
    Returns:
        Simple status response
    
    Raises:
        HTTPException: If there's an error checking liveness
    """
    try:
        return {"status": "alive"}
    except Exception as e:
        logger.error(f"Error checking liveness: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/readiness",
    summary="Readiness probe",
    description="Kubernetes readiness probe endpoint to determine if the service is ready to receive traffic."
)
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
    
    Raises:
        HTTPException: If there's an error checking readiness
    """
    try:
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
    except Exception as e:
        logger.error(f"Error checking readiness: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Legacy endpoints for backward compatibility
legacy_router = APIRouter(tags=["Health"])

@legacy_router.get("/health", response_model=ServiceHealth)
async def legacy_health_check(
    health_check: HealthCheck = Depends(get_health_check)
) -> ServiceHealth:
    """
    Legacy health check endpoint for backward compatibility.
    
    Returns:
        ServiceHealth object
    """
    logger.info("Legacy health check endpoint called - consider migrating to /v1/analysis/health-checks")
    return await health_check.check_health()

@legacy_router.get("/health/live")
async def legacy_liveness_probe() -> Dict[str, str]:
    """
    Legacy liveness probe endpoint for backward compatibility.
    
    Returns:
        Simple status response
    """
    logger.info("Legacy liveness probe endpoint called - consider migrating to /v1/analysis/health-checks/liveness")
    return {"status": "alive"}

@legacy_router.get("/health/ready")
async def legacy_readiness_probe(
    response: Response,
    health_check: HealthCheck = Depends(get_health_check)
) -> Dict[str, Any]:
    """
    Legacy readiness probe endpoint for backward compatibility.
    
    Args:
        response: FastAPI response
        health_check: Health check instance
        
    Returns:
        Status response
    """
    logger.info("Legacy readiness probe endpoint called - consider migrating to /v1/analysis/health-checks/readiness")
    
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
    # Include standardized router
    app.include_router(router, prefix="/api")
    
    # Include legacy router for backward compatibility
    app.include_router(legacy_router, prefix="/api")
    
    # Add dependency override
    app.dependency_overrides[get_health_check] = lambda: service_container.health_check