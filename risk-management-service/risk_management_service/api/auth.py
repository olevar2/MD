"""
Authentication utilities for the Risk Management Service API.
"""
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import APIKeyHeader
from typing import Optional, Dict

from common_lib.security import validate_api_key

# API key header definition
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(
    request: Request,
    api_key: Optional[str] = Depends(api_key_header),
) -> str:
    """
    Verify API key authentication for protected endpoints.
    
    Args:
        request: The request object
        api_key: API key from X-API-Key header
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: If authentication fails
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    # In a real-world scenario, you'd validate against keys stored in a secure location
    # like AWS Secret Manager, Azure Key Vault, or a database
    # For simplicity, we're using a hardcoded dictionary of valid keys
    valid_keys: Dict[str, str] = {
        "trading-gateway-service": "tg-service-key-12345",
        "portfolio-management-service": "pm-service-key-67890",
        "feature-store-service": "fs-service-key-abcde",
        "analysis-engine-service": "ae-service-key-fghij",
        "ml-integration-service": "ml-service-key-klmno",
    }
    
    # Check against all valid keys
    if not any(api_key == valid_key for valid_key in valid_keys.values()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    
    return api_key
