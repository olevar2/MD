"""
Security utilities for the ML Workbench Service.

Provides API key and JWT token authentication using common-lib security components.
"""
from fastapi import Request, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from typing import Dict, Optional, List

from common_lib.security import (
    validate_api_key,
    validate_token,
    jwt_auth,
    HTTPBearer,
    HTTPAuthorizationCredentials,
)

# Configuration - in a real implementation, these should be loaded from settings
API_KEY_HEADER = "X-API-Key"
JWT_SECRET_KEY = "your-secret-key-here"  # Should be loaded from environment or secret management
ALGORITHMS = ["HS256"]

# Valid API keys - in production, these should be stored securely
VALID_API_KEYS = {
    "ml-integration-service": "integration-service-key",
    "strategy-execution-engine": "strategy-engine-key",
    "ml-workbench-api": "workbench-internal-key",
}

# Security scheme for bearer token authentication
security = HTTPBearer()

async def api_key_middleware(request: Request, call_next):
    """
    Middleware to authenticate requests using API keys.
    Uses the common_lib.security validate_api_key function.
    
    Args:
        request: FastAPI request
        call_next: Next middleware handler
        
    Returns:
        Response from next handler if key is valid
    """
    # Skip validation for non-API routes
    if not request.url.path.startswith("/api"):
        return await call_next(request)
        
    # Skip validation for docs and OpenAPI schema
    if any(path in request.url.path for path in ["/docs", "/redoc", "/openapi.json", "/health"]):
        return await call_next(request)
    
    # Get API key from header
    api_key = request.headers.get(API_KEY_HEADER)
    
    if not api_key:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": f"{API_KEY_HEADER} header required"}
        )
    
    # Use common_lib to validate API key
    is_valid = validate_api_key(api_key, VALID_API_KEYS)
    
    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN, 
            content={"detail": "Invalid API key"}
        )
    
    # Key is valid, continue
    return await call_next(request)

async def get_token_payload(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    required_scopes: Optional[List[str]] = None
) -> Dict:
    """
    FastAPI dependency for JWT token validation.
    Uses common_lib.security to validate JWT tokens.
    
    Args:
        credentials: HTTP Authorization credentials
        required_scopes: Optional list of required permission scopes
        
    Returns:
        Token payload if valid
    """
    try:
        return validate_token(
            token=credentials.credentials,
            secret_key=JWT_SECRET_KEY,
            algorithms=ALGORITHMS,
            required_scopes=required_scopes
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
