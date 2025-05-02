"""
Authentication Middleware for Feedback API

This module implements authentication middleware for securing the feedback API endpoints.
"""

from typing import Optional, Callable
from fastapi import Request, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from core_foundations.utils.logger import get_logger
from core_foundations.api.auth.token_validator import validate_token
from core_foundations.config.configuration import ConfigurationManager

logger = get_logger(__name__)

# Security scheme for OAuth2 bearer token
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    config_manager: ConfigurationManager = None
) -> dict:
    """
    Validate the access token and return the current user.
    
    Args:
        credentials: HTTP authorization credentials
        config_manager: Configuration manager
        
    Returns:
        dict: User information
        
    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    token = credentials.credentials
    
    try:
        # Use the core foundations token validator
        user_info = await validate_token(token, config_manager)
        
        if not user_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        return user_info
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_role(required_roles: list[str]) -> Callable:
    """
    Dependency for requiring specific roles.
    
    Args:
        required_roles: List of roles required for access
        
    Returns:
        Callable: Dependency function
    """
    async def role_checker(user: dict = Depends(get_current_user)) -> dict:
        """Check if the user has the required roles."""
        user_roles = user.get("roles", [])
        
        # Super admin always has access
        if "admin" in user_roles:
            return user
            
        # Check if the user has any of the required roles
        if not any(role in user_roles for role in required_roles):
            logger.warning(f"Access denied for user {user.get('username')}. Required roles: {required_roles}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
            )
            
        return user
        
    return role_checker


# Predefined permission levels
FEEDBACK_ADMIN = require_role(["feedback_admin", "admin"])
FEEDBACK_READER = require_role(["feedback_reader", "feedback_admin", "admin"])
FEEDBACK_WRITER = require_role(["feedback_writer", "feedback_admin", "admin"])
