"""
Standardized Health Check API for Analysis Engine Service.

This module provides standardized API endpoints for health checks, including:
- Basic health check
- Liveness probe
- Readiness probe
- Detailed health check

All endpoints follow the platform's standardized API design patterns.
"""

from fastapi import APIRouter, Response, Depends, status, HTTPException, Request
from typing import Dict, Any, Optional, List
import logging
import time
from datetime import datetime, timedelta

from analysis_engine.monitoring.health_checks import HealthCheck, HealthStatus, ServiceHealth
from analysis_engine.core.container import ServiceContainer
from analysis_engine.monitoring.structured_logging import get_structured_logger
from analysis_engine.core.config import get_settings
from analysis_engine.core.dependencies import get_service_container

# Get structured logger
logger = get_structured_logger(__name__)

# Create router with standardized prefix
router = APIRouter(
    prefix="/api/v1/analysis/health-checks",
    tags=["Health"]
)

# Create legacy router for backward compatibility
legacy_router = APIRouter(tags=["Legacy Health"])

# Start time for uptime calculation
start_time = time.time()

# Get settings
settings = get_settings()

def get_health_check(service_container: ServiceContainer = Depends(get_service_container)) -> HealthCheck:
    """
    Get health check dependency.
    
    Args:
        service_container: Service container with dependencies
        
    Returns:
        HealthCheck instance
    """
    return HealthCheck(service_container)

@router.get(
    "",
    response_model=ServiceHealth,
    summary="Get detailed health status",
    description="Get detailed health status of the Analysis Engine Service and its dependencies."
)
async def health_check(
    health_check: HealthCheck = Depends(get_health_check)
) -> ServiceHealth:
    """
    Get detailed health status of the Analysis Engine Service.
    
    Args:
        health_check: Health check instance
        
    Returns:
        ServiceHealth object
    """
    logger.info("Health check requested")
    return await health_check.check_health()

@router.get(
    "/liveness",
    summary="Liveness probe",
    description="Kubernetes liveness probe endpoint. Returns 200 OK if the service is alive."
)
async def liveness_probe() -> Dict[str, str]:
    """
    Kubernetes liveness probe endpoint.
    
    This endpoint is used by Kubernetes to determine if the service is alive.
    It should return a 200 OK response if the service is running.
    
    Returns:
        Simple status response
    """
    logger.debug("Liveness probe requested")
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get(
    "/readiness",
    summary="Readiness probe",
    description="Kubernetes readiness probe endpoint. Returns 200 OK if the service is ready to receive traffic."
)
async def readiness_probe(
    response: Response,
    health_check: HealthCheck = Depends(get_health_check)
) -> Dict[str, Any]:
    """
    Kubernetes readiness probe endpoint.
    
    This endpoint is used by Kubernetes to determine if the service is ready
    to receive traffic. It should return a 200 OK response if the service is
    ready, or a 503 Service Unavailable response if the service is not ready.
    
    Args:
        response: FastAPI response
        health_check: Health check instance
        
    Returns:
        Status response
    """
    logger.debug("Readiness probe requested")
    
    try:
        # Check health
        health_status = await health_check.check_health()
        
        # If service is unhealthy, return 503
        if health_status.status == HealthStatus.UNHEALTHY:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {
                "status": "not ready",
                "reason": "Service is unhealthy",
                "details": health_status.dict()
            }
        
        # If service is degraded, check if it's still able to handle requests
        if health_status.status == HealthStatus.DEGRADED:
            # Check if critical dependencies are available
            critical_services = ["database", "feature_store", "market_data_service"]
            critical_failure = False
            
            for service_name in critical_services:
                if service_name in health_status.services and health_status.services[service_name].status == HealthStatus.UNHEALTHY:
                    critical_failure = True
                    break
            
            if critical_failure:
                response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
                return {
                    "status": "not ready",
                    "reason": "Critical dependency unavailable",
                    "details": health_status.dict()
                }
        
        # Service is ready
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error checking readiness: {str(e)}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {
            "status": "not ready",
            "reason": str(e)
        }

# Legacy endpoints for backward compatibility
@legacy_router.get("/health", response_model=ServiceHealth)
async def legacy_health_check(
    request: Request,
    health_check: HealthCheck = Depends(get_health_check)
) -> ServiceHealth:
    """
    Legacy health check endpoint for backward compatibility.
    
    Returns:
        ServiceHealth object
    """
    logger.info("Legacy health check endpoint called - consider migrating to /api/v1/analysis/health-checks")
    
    # Add deprecation notice to response headers
    request.state.deprecation_notice = "This endpoint is deprecated. Please use /api/v1/analysis/health-checks instead."
    
    return await health_check.check_health()

@legacy_router.get("/health/live")
async def legacy_liveness_probe(request: Request) -> Dict[str, str]:
    """
    Legacy liveness probe endpoint for backward compatibility.
    
    Returns:
        Simple status response
    """
    logger.info("Legacy liveness probe endpoint called - consider migrating to /api/v1/analysis/health-checks/liveness")
    
    # Add deprecation notice to response headers
    request.state.deprecation_notice = "This endpoint is deprecated. Please use /api/v1/analysis/health-checks/liveness instead."
    
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }

@legacy_router.get("/health/ready")
async def legacy_readiness_probe(
    request: Request,
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
    logger.info("Legacy readiness probe endpoint called - consider migrating to /api/v1/analysis/health-checks/readiness")
    
    # Add deprecation notice to response headers
    request.state.deprecation_notice = "This endpoint is deprecated. Please use /api/v1/analysis/health-checks/readiness instead."
    
    return await readiness_probe(response, health_check)

def setup_health_routes(app):
    """
    Set up health check routes for the application.
    
    Args:
        app: FastAPI application
    """
    # Include standardized router
    app.include_router(router)
    
    # Include legacy router for backward compatibility
    app.include_router(legacy_router)
    
    logger.info("Health check routes configured")
