"""
Middleware for correlation ID handling.
"""

import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware for handling correlation IDs.
    
    This middleware ensures that every request has a correlation ID for tracing.
    If the request already has a correlation ID in the headers, it will be used.
    Otherwise, a new correlation ID will be generated.
    """
    
    async def dispatch(self, request: Request, call_next):
        """
        Process the request and add correlation ID.
        
        Args:
            request: FastAPI request
            call_next: Next middleware or endpoint
            
        Returns:
            Response
        """
        # Get or generate correlation ID
        correlation_id = request.headers.get('X-Correlation-ID')
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        
        # Store in request state
        request.state.correlation_id = correlation_id
        
        # Process the request
        response = await call_next(request)
        
        # Add correlation ID to response headers
        response.headers['X-Correlation-ID'] = correlation_id
        
        return response
