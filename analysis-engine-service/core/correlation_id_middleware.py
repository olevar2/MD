"""
Correlation ID Middleware

This middleware ensures that all requests have a correlation ID for error tracking.
If a request doesn't have a correlation ID, one is generated and added to the request.
The correlation ID is also added to the response headers.
"""

import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from analysis_engine.core.exceptions_bridge import generate_correlation_id
from analysis_engine.core.logging import get_logger

logger = get_logger(__name__)

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware to ensure all requests have a correlation ID.
    
    This middleware:
    1. Checks if the request has a correlation ID header
    2. If not, generates a new correlation ID
    3. Adds the correlation ID to the request state
    4. Adds the correlation ID to the response headers
    5. Adds the correlation ID to the logging context
    """
    
    def __init__(self, app: ASGIApp, header_name: str = "X-Correlation-ID"):
        """
        Initialize the middleware.
        
        Args:
            app: The ASGI application
            header_name: The name of the correlation ID header
        """
        super().__init__(app)
        self.header_name = header_name
    
    async def dispatch(self, request: Request, call_next):
        """
        Process the request and add correlation ID.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            The response from the next middleware or route handler
        """
        # Check if the request has a correlation ID header
        correlation_id = request.headers.get(self.header_name)
        
        # If not, generate a new correlation ID
        if not correlation_id:
            correlation_id = generate_correlation_id()
        
        # Add the correlation ID to the request state
        request.state.correlation_id = correlation_id
        
        # Add the correlation ID to the logging context
        logger_adapter = logging.LoggerAdapter(
            logger, {"correlation_id": correlation_id}
        )
        logger_adapter.debug(f"Request {request.method} {request.url.path}")
        
        # Process the request
        response = await call_next(request)
        
        # Add the correlation ID to the response headers
        response.headers[self.header_name] = correlation_id
        
        return response
