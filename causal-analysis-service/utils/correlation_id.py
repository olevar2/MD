"""
Correlation ID Middleware

This module provides middleware for adding correlation IDs to requests.
"""
import logging
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware for adding correlation IDs to requests.
    """
    async def dispatch(self, request: Request, call_next):
        """
        Add correlation ID to request and response headers.
        
        Args:
            request: The request
            call_next: The next middleware or route handler
            
        Returns:
            Response: The response
        """
        # Get correlation ID from header or generate a new one
        correlation_id = request.headers.get("X-Correlation-ID")
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        
        # Add correlation ID to request state
        request.state.correlation_id = correlation_id
        
        # Log request with correlation ID
        logger.info(f"Request {request.method} {request.url.path} - Correlation ID: {correlation_id}")
        
        # Call next middleware or route handler
        response = await call_next(request)
        
        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id
        
        return response