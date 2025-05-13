"""
CSRF Protection Middleware

This module provides middleware for CSRF protection.
"""

import logging
import secrets
import time
from typing import Dict, Any, Optional, List, Callable, Awaitable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from common_lib.config.config_manager import ConfigManager


class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    """
    Middleware for CSRF protection.
    
    This middleware implements CSRF protection using the Double Submit Cookie pattern.
    """
    
    def __init__(self, app):
        """
        Initialize the middleware.
        
        Args:
            app: FastAPI application
        """
        super().__init__(app)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config_manager = ConfigManager()
        
        # Get configuration
        try:
            service_specific = self.config_manager.get_service_specific_config()
            if hasattr(service_specific, "csrf_protection"):
                self.csrf_config = getattr(service_specific, "csrf_protection")
            else:
                self.csrf_config = None
        except Exception as e:
            self.logger.warning(f"Error getting CSRF protection configuration: {str(e)}")
            self.csrf_config = None
        
        # Set default values
        self.enabled = getattr(self.csrf_config, "enabled", True) if self.csrf_config else True
        self.exempt_paths = getattr(self.csrf_config, "exempt_paths", []) if self.csrf_config else []
        self.cookie_name = getattr(self.csrf_config, "cookie_name", "csrf_token") if self.csrf_config else "csrf_token"
        self.header_name = getattr(self.csrf_config, "header_name", "X-CSRF-Token") if self.csrf_config else "X-CSRF-Token"
        self.cookie_max_age = getattr(self.csrf_config, "cookie_max_age", 86400) if self.csrf_config else 86400  # 24 hours
        self.secure = getattr(self.csrf_config, "secure", True) if self.csrf_config else True
        self.same_site = getattr(self.csrf_config, "same_site", "lax") if self.csrf_config else "lax"
        
        # Always add health check to exempt paths
        self.exempt_paths.append("/health")
        
        # Add login and authentication endpoints to exempt paths
        self.exempt_paths.extend([
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/api/v1/auth/refresh",
            "/docs*",
            "/redoc*",
            "/openapi.json"
        ])
    
    async def dispatch(self, request: Request, call_next):
        """
        Process the request.
        
        Args:
            request: FastAPI request
            call_next: Next middleware or route handler
            
        Returns:
            Response
        """
        # Skip CSRF protection if disabled or path is exempt
        if not self.enabled or self._is_exempt_path(request.url.path):
            return await call_next(request)
        
        # Skip CSRF protection for safe methods (GET, HEAD, OPTIONS, TRACE)
        if request.method.upper() in ["GET", "HEAD", "OPTIONS", "TRACE"]:
            response = await call_next(request)
            
            # Set CSRF token cookie if it doesn't exist
            if self.cookie_name not in request.cookies:
                csrf_token = self._generate_csrf_token()
                response.set_cookie(
                    key=self.cookie_name,
                    value=csrf_token,
                    max_age=self.cookie_max_age,
                    httponly=True,
                    secure=self.secure,
                    samesite=self.same_site
                )
            
            return response
        
        # For unsafe methods (POST, PUT, DELETE, PATCH), validate CSRF token
        csrf_cookie = request.cookies.get(self.cookie_name)
        csrf_header = request.headers.get(self.header_name)
        
        if not csrf_cookie or not csrf_header or csrf_cookie != csrf_header:
            self.logger.warning(f"CSRF validation failed for {request.url.path}")
            return JSONResponse(
                status_code=403,
                content={"detail": "CSRF token validation failed"}
            )
        
        # Continue processing
        return await call_next(request)
    
    def _is_exempt_path(self, path: str) -> bool:
        """
        Check if a path is exempt from CSRF protection.
        
        Args:
            path: Path to check
            
        Returns:
            True if the path is exempt, False otherwise
        """
        # Check exact matches
        if path in self.exempt_paths:
            return True
        
        # Check prefix matches
        for exempt_path in self.exempt_paths:
            if exempt_path.endswith("*") and path.startswith(exempt_path[:-1]):
                return True
        
        return False
    
    def _generate_csrf_token(self) -> str:
        """
        Generate a secure CSRF token.
        
        Returns:
            CSRF token
        """
        return secrets.token_hex(32)