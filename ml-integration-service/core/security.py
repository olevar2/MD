"""
Security middleware for ML Integration Service.

Implements API key authentication using common-lib security features.
"""
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse

from common_lib.security import validate_api_key
from config.settings import settings

async def api_key_middleware(request: Request, call_next):
    """
    Middleware to validate API key for requests.
    Uses the common_lib.security validate_api_key function.
    
    Args:
        request: FastAPI request
        call_next: Next middleware handler
        
    Returns:
        Response from next handler if key is valid
    """
    # Skip validation for certain paths
    if request.url.path in ["/docs", "/redoc", "/openapi.json", "/health", "/"]:
        return await call_next(request)
    
    # Get API key from header
    api_key = request.headers.get(settings.API_KEY_NAME)
    
    # Valid keys config
    valid_keys = {
        "ml-integration-service": settings.API_KEY
    }
    
    if not api_key:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": f"{settings.API_KEY_NAME} header required"}
        )
    
    # Use common_lib to validate the API key
    is_valid = validate_api_key(api_key, valid_keys)
    
    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"detail": "Invalid API key"}
        )
    
    # Key is valid, continue to handler
    return await call_next(request)
