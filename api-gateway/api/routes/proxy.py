"""
Proxy Routes

This module provides routes for proxying requests to backend services.
"""

import logging
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import JSONResponse

from ...services.proxy import ProxyService
from ...services.registry import ServiceRegistry
from ...core.response.standard_response import create_error_response


# Create logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Create proxy service
proxy_service = ProxyService()

# Create service registry
service_registry = ServiceRegistry()


@router.on_event("startup")
async def startup():
    """
    Startup event handler.
    """
    # Start service registry
    await service_registry.start()


@router.on_event("shutdown")
async def shutdown():
    """
    Shutdown event handler.
    """
    # Stop service registry
    await service_registry.stop()

    # Close proxy service
    await proxy_service.close()


@router.api_route("/{service_name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def proxy_request(request: Request, service_name: str, path: str):
    """
    Proxy a request to a backend service.

    Args:
        request: Request
        service_name: Service name
        path: Request path

    Returns:
        Response from the backend service
    """
    # Get correlation ID and request ID
    correlation_id = request.headers.get("X-Correlation-ID", "")
    request_id = request.headers.get("X-Request-ID", "")

    # Check if service exists
    try:
        service = service_registry.get_service(service_name)
        if not service:
            return JSONResponse(
                status_code=404,
                content=create_error_response(
                    code="SERVICE_NOT_FOUND",
                    message=f"Service {service_name} not found",
                    correlation_id=correlation_id,
                    request_id=request_id
                ).dict()
            )
    except Exception as e:
        logger.error(f"Error getting service: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=create_error_response(
                code="INTERNAL_SERVER_ERROR",
                message="Internal server error",
                correlation_id=correlation_id,
                request_id=request_id
            ).dict()
        )

    # Check if service is healthy
    if service["status"] == "unhealthy":
        return JSONResponse(
            status_code=503,
            content=create_error_response(
                code="SERVICE_UNAVAILABLE",
                message=f"Service {service_name} is unavailable",
                correlation_id=correlation_id,
                request_id=request_id
            ).dict()
        )

    # Proxy request
    return await proxy_service.proxy_request(
        request=request,
        service_name=service_name,
        path=f"/{path}",
        correlation_id=correlation_id,
        request_id=request_id
    )