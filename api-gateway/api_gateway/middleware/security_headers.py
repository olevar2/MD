"""
Security Headers Middleware

This module provides middleware for adding security headers to responses.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from common_lib.config.config_manager import ConfigManager


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware for adding security headers to responses.
    
    This middleware adds various security headers to responses, including:
    - Content-Security-Policy (CSP)
    - X-Content-Type-Options
    - X-Frame-Options
    - X-XSS-Protection
    - Referrer-Policy
    - Strict-Transport-Security (HSTS)
    - Permissions-Policy
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
            if hasattr(service_specific, "security_headers"):
                self.security_headers_config = getattr(service_specific, "security_headers")
            else:
                self.security_headers_config = None
        except Exception as e:
            self.logger.warning(f"Error getting security headers configuration: {str(e)}")
            self.security_headers_config = None
        
        # Set default values
        self.enabled = getattr(self.security_headers_config, "enabled", True) if self.security_headers_config else True
        self.exempt_paths = getattr(self.security_headers_config, "exempt_paths", []) if self.security_headers_config else []
        
        # Always add health check to exempt paths
        self.exempt_paths.append("/health")
        
        # Set default CSP directives
        self.csp_directives = getattr(self.security_headers_config, "csp_directives", {}) if self.security_headers_config else {}
        if not self.csp_directives:
            self.csp_directives = {
                "default-src": ["'self'"],
                "script-src": ["'self'"],
                "style-src": ["'self'"],
                "img-src": ["'self'", "data:"],
                "font-src": ["'self'"],
                "connect-src": ["'self'"],
                "frame-src": ["'none'"],
                "object-src": ["'none'"],
                "base-uri": ["'self'"],
                "form-action": ["'self'"],
                "frame-ancestors": ["'none'"],
                "upgrade-insecure-requests": []
            }
        
        # Set other security headers
        self.x_content_type_options = getattr(self.security_headers_config, "x_content_type_options", "nosniff") if self.security_headers_config else "nosniff"
        self.x_frame_options = getattr(self.security_headers_config, "x_frame_options", "DENY") if self.security_headers_config else "DENY"
        self.x_xss_protection = getattr(self.security_headers_config, "x_xss_protection", "1; mode=block") if self.security_headers_config else "1; mode=block"
        self.referrer_policy = getattr(self.security_headers_config, "referrer_policy", "strict-origin-when-cross-origin") if self.security_headers_config else "strict-origin-when-cross-origin"
        self.hsts_max_age = getattr(self.security_headers_config, "hsts_max_age", 31536000) if self.security_headers_config else 31536000  # 1 year
        self.hsts_include_subdomains = getattr(self.security_headers_config, "hsts_include_subdomains", True) if self.security_headers_config else True
        self.hsts_preload = getattr(self.security_headers_config, "hsts_preload", True) if self.security_headers_config else True
        
        # Set permissions policy
        self.permissions_policy = getattr(self.security_headers_config, "permissions_policy", {}) if self.security_headers_config else {}
        if not self.permissions_policy:
            self.permissions_policy = {
                "accelerometer": ["()"],
                "ambient-light-sensor": ["()"],
                "autoplay": ["()"],
                "camera": ["()"],
                "encrypted-media": ["()"],
                "fullscreen": ["()"],
                "geolocation": ["()"],
                "gyroscope": ["()"],
                "magnetometer": ["()"],
                "microphone": ["()"],
                "midi": ["()"],
                "payment": ["()"],
                "picture-in-picture": ["()"],
                "speaker": ["()"],
                "usb": ["()"],
                "vr": ["()"]
            }
    
    async def dispatch(self, request: Request, call_next):
        """
        Process the request.
        
        Args:
            request: FastAPI request
            call_next: Next middleware or route handler
            
        Returns:
            Response
        """
        # Continue processing
        response = await call_next(request)
        
        # Skip security headers if disabled or path is exempt
        if not self.enabled or self._is_exempt_path(request.url.path):
            return response
        
        # Add security headers
        self._add_csp_header(response)
        self._add_other_security_headers(response)
        
        return response
    
    def _is_exempt_path(self, path: str) -> bool:
        """
        Check if a path is exempt from security headers.
        
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
    
    def _add_csp_header(self, response: Response):
        """
        Add Content-Security-Policy header to response.
        
        Args:
            response: Response to add header to
        """
        # Build CSP header value
        csp_parts = []
        for directive, sources in self.csp_directives.items():
            if sources:
                csp_parts.append(f"{directive} {' '.join(sources)}")
            else:
                csp_parts.append(directive)
        
        csp_value = "; ".join(csp_parts)
        
        # Add CSP header
        response.headers["Content-Security-Policy"] = csp_value
    
    def _add_other_security_headers(self, response: Response):
        """
        Add other security headers to response.
        
        Args:
            response: Response to add headers to
        """
        # Add X-Content-Type-Options header
        response.headers["X-Content-Type-Options"] = self.x_content_type_options
        
        # Add X-Frame-Options header
        response.headers["X-Frame-Options"] = self.x_frame_options
        
        # Add X-XSS-Protection header
        response.headers["X-XSS-Protection"] = self.x_xss_protection
        
        # Add Referrer-Policy header
        response.headers["Referrer-Policy"] = self.referrer_policy
        
        # Add Strict-Transport-Security header
        hsts_value = f"max-age={self.hsts_max_age}"
        if self.hsts_include_subdomains:
            hsts_value += "; includeSubDomains"
        if self.hsts_preload:
            hsts_value += "; preload"
        response.headers["Strict-Transport-Security"] = hsts_value
        
        # Add Permissions-Policy header
        permissions_parts = []
        for feature, allowlist in self.permissions_policy.items():
            permissions_parts.append(f"{feature}={', '.join(allowlist)}")
        
        permissions_value = ", ".join(permissions_parts)
        response.headers["Permissions-Policy"] = permissions_value