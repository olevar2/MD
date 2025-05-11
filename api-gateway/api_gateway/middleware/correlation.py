"""
Correlation Middleware

This module provides middleware for correlation ID tracking.
"""

import logging
import uuid
from typing import Dict, Any, Optional, List, Callable, Awaitable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class CorrelationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for correlation ID tracking.
    
    This middleware ensures that all requests have a correlation ID for tracing.
    """
    
    def __init__(self, app):
        """
        Initialize the middleware.
        
        Args:
            app: FastAPI application
        """
        super().__init__(app)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def dispatch(self, request: Request, call_next):
        """
        Process the request.
        
        Args:
            request: FastAPI request
            call_next: Next middleware or route handler
            
        Returns:
            Response
        """
        # Get correlation ID from header, or generate a new one
        correlation_id = request.headers.get("X-Correlation-ID")
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
            
            # Add correlation ID to request headers
            request.headers.__dict__["_list"].append(
                (b"x-correlation-id", correlation_id.encode())
            )
        
        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())
            
            # Add request ID to request headers
            request.headers.__dict__["_list"].append(
                (b"x-request-id", request_id.encode())
            )
        
        # Process request
        response = await call_next(request)
        
        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response