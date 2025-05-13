"""
XSS Protection Middleware

This module provides middleware for XSS protection.
"""

import logging
import re
import html
from typing import Dict, Any, Optional, List, Callable, Awaitable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import UploadFile

from common_lib.config.config_manager import ConfigManager


class XSSProtectionMiddleware(BaseHTTPMiddleware):
    """
    Middleware for XSS protection.
    
    This middleware sanitizes request data to prevent XSS attacks.
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
            if hasattr(service_specific, "xss_protection"):
                self.xss_config = getattr(service_specific, "xss_protection")
            else:
                self.xss_config = None
        except Exception as e:
            self.logger.warning(f"Error getting XSS protection configuration: {str(e)}")
            self.xss_config = None
        
        # Set default values
        self.enabled = getattr(self.xss_config, "enabled", True) if self.xss_config else True
        self.exempt_paths = getattr(self.xss_config, "exempt_paths", []) if self.xss_config else []
        
        # Always add health check to exempt paths
        self.exempt_paths.append("/health")
    
    async def dispatch(self, request: Request, call_next):
        """
        Process the request.
        
        Args:
            request: FastAPI request
            call_next: Next middleware or route handler
            
        Returns:
            Response
        """
        # Skip XSS protection if disabled or path is exempt
        if not self.enabled or self._is_exempt_path(request.url.path):
            return await call_next(request)
        
        # Clone the request and sanitize the body
        try:
            # Get the original request body
            body = await request.body()
            
            # If the body is empty, just continue
            if not body:
                return await call_next(request)
            
            # Parse the body as JSON
            try:
                import json
                json_data = json.loads(body)
                
                # Sanitize the JSON data
                sanitized_data = self._sanitize_data(json_data)
                
                # Create a new request with the sanitized body
                sanitized_body = json.dumps(sanitized_data).encode()
                
                # Override the request body
                async def receive():
    """
    Receive.
    
    """

                    return {"type": "http.request", "body": sanitized_body}
                
                # Create a new request with the sanitized body
                request._receive = receive
            except json.JSONDecodeError:
                # If the body is not JSON, just continue
                pass
        except Exception as e:
            self.logger.warning(f"Error sanitizing request body: {str(e)}")
        
        # Continue processing
        return await call_next(request)
    
    def _is_exempt_path(self, path: str) -> bool:
        """
        Check if a path is exempt from XSS protection.
        
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
    
    def _sanitize_data(self, data: Any) -> Any:
        """
        Sanitize data to prevent XSS attacks.
        
        Args:
            data: Data to sanitize
            
        Returns:
            Sanitized data
        """
        # Handle different data types
        if isinstance(data, dict):
            # Sanitize dictionary
            return {k: self._sanitize_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            # Sanitize list
            return [self._sanitize_data(item) for item in data]
        elif isinstance(data, str):
            # Sanitize string
            return self._sanitize_string(data)
        elif isinstance(data, (int, float, bool, type(None))):
            # No need to sanitize primitive types
            return data
        else:
            # For other types, convert to string and sanitize
            return self._sanitize_string(str(data))
    
    def _sanitize_string(self, value: str) -> str:
        """
        Sanitize a string to prevent XSS attacks.
        
        Args:
            value: String to sanitize
            
        Returns:
            Sanitized string
        """
        # Escape HTML entities
        sanitized = html.escape(value)
        
        # Remove potentially dangerous patterns
        sanitized = re.sub(r"javascript:", "", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"data:", "", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"vbscript:", "", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"on\w+\s*=", "", sanitized, flags=re.IGNORECASE)
        
        return sanitized