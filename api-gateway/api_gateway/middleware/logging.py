"""
Logging Middleware

This module provides middleware for request logging.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Callable, Awaitable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request logging.
    
    This middleware logs information about incoming requests and outgoing responses.
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
        # Get request information
        request_id = request.headers.get("X-Request-ID", "")
        correlation_id = request.headers.get("X-Correlation-ID", "")
        user_id = request.headers.get("X-User-ID", "")
        
        # Log request
        self.logger.info(
            f"Request: {request.method} {request.url.path} "
            f"(request_id={request_id}, correlation_id={correlation_id}, user_id={user_id})"
        )
        
        # Record start time
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate request duration
            duration = time.time() - start_time
            
            # Log response
            self.logger.info(
                f"Response: {request.method} {request.url.path} {response.status_code} "
                f"({duration:.3f}s) "
                f"(request_id={request_id}, correlation_id={correlation_id}, user_id={user_id})"
            )
            
            return response
        except Exception as e:
            # Calculate request duration
            duration = time.time() - start_time
            
            # Log error
            self.logger.error(
                f"Error: {request.method} {request.url.path} "
                f"({duration:.3f}s) "
                f"(request_id={request_id}, correlation_id={correlation_id}, user_id={user_id}): "
                f"{str(e)}"
            )
            
            # Re-raise exception
            raise