"""
Auth module.

This module provides functionality for...
"""

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import APIKeyHeader
from typing import Optional

from ..config.settings import Settings
from ..models.schemas import ServiceAuth

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
service_name_header = APIKeyHeader(name="X-Service-Name", auto_error=False)


async def verify_service_auth(
    request: Request,
    api_key: Optional[str] = Depends(api_key_header),
    service_name: Optional[str] = Depends(service_name_header),
    settings: Settings = Depends(lambda: Settings())
) -> ServiceAuth:
    """
    Verify service-to-service authentication.
    
    Args:
        request: The request object
        api_key: API key from X-API-Key header
        service_name: Service name from X-Service-Name header
        settings: Application settings
        
    Returns:
        ServiceAuth object with authenticated service information
        
    Raises:
        HTTPException: If authentication fails
    """
    if not api_key or not service_name:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials"
        )
    
    # Validate service name and API key
    # In production, this would use a more sophisticated approach
    # like checking a database of valid service credentials
    valid_services = {
        "feature-store-service": settings.FEATURE_STORE_API_KEY,
        "strategy-execution-engine": settings.STRATEGY_ENGINE_API_KEY,
        "portfolio-management-service": settings.PORTFOLIO_API_KEY,
        "ml_workbench-service": settings.ML_WORKBENCH_API_KEY,
    }
    
    if service_name not in valid_services:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Unknown service: {service_name}"
        )
    
    if api_key != valid_services[service_name]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key for service"
        )
    
    return ServiceAuth(service_name=service_name, api_key=api_key)