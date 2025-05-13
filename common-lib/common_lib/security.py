"""
Core Security Utilities for Forex Trading Platform

This module provides centralized security mechanisms used across all services.
It includes functionality for API key validation, JWT token handling, role-based
access control, and security header utilities.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union
import json
import logging

try:
    from jose import jwt, JWTError
    from fastapi import Request, HTTPException, status, Depends
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
    from pydantic import BaseModel
except ImportError:
    # Create mock classes for when FastAPI is not installed
    # This allows the module to be imported even if FastAPI is not a dependency
    class HTTPException(Exception):
    """
    HTTPException class that inherits from Exception.
    
    Attributes:
        Add attributes here
    """

        def __init__(self, status_code=None, detail=None, headers=None):
    """
      init  .
    
    Args:
        status_code: Description of status_code
        detail: Description of detail
        headers: Description of headers
    
    """

            # This is a mock class for when FastAPI is not installed.
            # It allows the module to be imported without FastAPI as a hard dependency.
            # No specific initialization logic is needed for the mock.
            self.status_code = status_code
            self.detail = detail
            self.headers = headers

    class Request:
    """
    Request class.
    
    Attributes:
        Add attributes here
    """

        pass
        
    class Depends:
    """
    Depends class.
    
    Attributes:
        Add attributes here
    """

        def __init__(self, *args, **kwargs):
    """
      init  .
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

            # This is a mock implementation that intentionally does nothing
            # It exists solely to allow imports without FastAPI dependency
            pass

    status = type('obj', (object,), {
        'HTTP_401_UNAUTHORIZED': 401,
        'HTTP_403_FORBIDDEN': 403
    })

    BaseModel = type('BaseModel', (object,), {})
    HTTPBearer = object
    HTTPAuthorizationCredentials = object
    APIKeyHeader = object
    JWTError = Exception
    jwt = None


logger = logging.getLogger(__name__)


class SecurityEvent(BaseModel):
    """Model for security-related events for auditing and monitoring"""
    timestamp: datetime
    event_type: str
    user_id: str
    resource: str
    action: str
    status: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class TokenPayload(BaseModel):
    """Standard JWT token payload structure"""
    sub: str
    exp: int
    scopes: List[str] = []
    additional_claims: Optional[Dict[str, Any]] = None


# Security scheme for bearer token authentication
security = HTTPBearer()


def validate_api_key(
    api_key: str, 
    valid_keys: Dict[str, str],
    service_name: Optional[str] = None
) -> bool:
    """
    Validate an API key against valid keys.
    
    Args:
        api_key: The API key to validate
        valid_keys: Dictionary mapping service names to valid API keys
        service_name: Optional service name to validate against
        
    Returns:
        True if valid, False otherwise
    """
    if service_name:
        # Validate against specific service
        return service_name in valid_keys and api_key == valid_keys[service_name]
    else:
        # Validate against any service
        return api_key in valid_keys.values()


def create_jwt_token(
    subject: str,
    secret_key: str,
    scopes: List[str] = None,
    expires_delta: Optional[timedelta] = None,
    algorithm: str = "HS256",
    additional_claims: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a JWT token with standard claims.
    
    Args:
        subject: The subject (typically user_id)
        secret_key: Secret key for signing
        scopes: Optional list of permission scopes
        expires_delta: Optional expiration time delta
        algorithm: JWT algorithm to use
        additional_claims: Additional claims to include in the token
        
    Returns:
        Encoded JWT token string
    """
    if scopes is None:
        scopes = []
    
    if expires_delta is None:
        expires_delta = timedelta(minutes=15)  # Default 15 minutes
        
    expires = datetime.now(timezone.utc) + expires_delta
    
    to_encode = {
        "sub": str(subject),
        "exp": expires.timestamp(),
        "scopes": scopes
    }
    
    if additional_claims:
        to_encode.update(additional_claims)
        
    return jwt.encode(to_encode, secret_key, algorithm=algorithm)


def validate_token(
    token: str,
    secret_key: str,
    algorithms: List[str] = None,
    required_scopes: List[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Validate a JWT token and verify required scopes.
    
    Args:
        token: JWT token to validate
        secret_key: Secret key for verification
        algorithms: List of allowed algorithms
        required_scopes: List of required scopes
        
    Returns:
        Token payload if valid, None if invalid
        
    Raises:
        JWTError: If token is invalid
        ValueError: If required scopes are not present
    """
    if algorithms is None:
        algorithms = ["HS256"]
        
    payload = jwt.decode(token, secret_key, algorithms=algorithms)
    
    if required_scopes:
        token_scopes = set(payload.get("scopes", []))
        required_scope_set = set(required_scopes)
        if not required_scope_set.issubset(token_scopes):
            raise ValueError(f"Token missing required scopes: {required_scopes}")
    
    return payload


def get_security_headers() -> Dict[str, str]:
    """
    Get recommended security headers for HTTP responses.
    
    Returns:
        Dictionary of security headers
    """
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Content-Security-Policy": "default-src 'self'",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache"
    }


async def api_key_auth(
    request: Request,
    api_key_header: str = "X-API-Key",
    service_name_header: str = "X-Service-Name",
    valid_keys: Dict[str, str] = None
):
    """
    FastAPI dependency for API key authentication.
    
    Args:
        request: FastAPI request object
        api_key_header: Header name for API key
        service_name_header: Header name for service name
        valid_keys: Dictionary mapping service names to valid API keys
        
    Returns:
        Dict with service_name and is_valid
        
    Raises:
        HTTPException: If authentication fails
    """
    if valid_keys is None:
        valid_keys = {}
        
    api_key = request.headers.get(api_key_header)
    service_name = request.headers.get(service_name_header)
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    if service_name and service_name not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Unknown service: {service_name}"
        )
    
    is_valid = validate_api_key(api_key, valid_keys, service_name)
    
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    
    return {"service_name": service_name, "is_valid": True}


async def jwt_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    secret_key: str = None,
    algorithms: List[str] = None,
    required_scopes: List[str] = None
) -> Dict[str, Any]:
    """
    FastAPI dependency for JWT authentication.
    
    Args:
        credentials: HTTP Authorization credentials
        secret_key: Secret key for JWT validation
        algorithms: List of allowed algorithms
        required_scopes: List of required scopes
        
    Returns:
        Token payload if valid
        
    Raises:
        HTTPException: If authentication fails
    """
    if algorithms is None:
        algorithms = ["HS256"]
        
    if not secret_key:
        raise ValueError("JWT secret key is required")
        
    try:
        token = credentials.credentials
        payload = validate_token(
            token, 
            secret_key, 
            algorithms=algorithms,
            required_scopes=required_scopes
        )
        return payload
    
    except (JWTError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"}
        )


def create_security_event(
    event_type: str,
    user_id: str,
    resource: str,
    action: str,
    status: str,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> SecurityEvent:
    """
    Create a security event using current UTC time.
    
    Args:
        event_type: Type of security event
        user_id: ID of the user involved
        resource: Resource being accessed
        action: Action being performed
        status: Status of the event
        ip_address: Optional client IP address
        user_agent: Optional client user agent
        details: Optional additional details
        
    Returns:
        SecurityEvent object
    """
    return SecurityEvent(
        timestamp=datetime.now(timezone.utc),
        event_type=event_type,
        user_id=user_id,
        resource=resource,
        action=action,
        status=status,
        ip_address=ip_address,
        user_agent=user_agent,
        details=details
    )


def log_security_event(event: SecurityEvent, logger_instance=None):
    """
    Log a security event for audit purposes.
    
    Args:
        event: SecurityEvent to log
        logger_instance: Optional logger instance to use
    """
    if logger_instance is None:
        logger_instance = logger
    
    try:
        logger_instance.info(
            json.dumps(event.dict(), default=str),
            extra={"event_type": event.event_type}
        )
    except Exception as e:
        logger.error(f"Failed to log security event: {e}")