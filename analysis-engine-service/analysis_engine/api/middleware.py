"""
API Middleware for Analysis Engine Service.

This module provides middleware for the FastAPI application, including:
- Request ID middleware
- Correlation ID middleware
- Metrics middleware
- Logging middleware
"""

import time
import uuid
from typing import Callable, Dict, Any, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from analysis_engine.monitoring.metrics import MetricsRecorder, API_REQUEST_DURATION_SECONDS
from analysis_engine.monitoring.structured_logging import (
    get_structured_logger,
    set_correlation_id,
    set_request_id,
    get_correlation_id,
    get_request_id
)

logger = get_structured_logger(__name__)

class RequestIdMiddleware(BaseHTTPMiddleware):
    """Middleware that adds a request ID to each request."""
    
    def __init__(
        self,
        app: ASGIApp,
        header_name: str = "X-Request-ID"
    ):
        """
        Initialize the middleware.
        
        Args:
            app: ASGI application
            header_name: Name of the header to use for the request ID
        """
        super().__init__(app)
        self.header_name = header_name
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Process the request.
        
        Args:
            request: FastAPI request
            call_next: Function to call the next middleware
            
        Returns:
            FastAPI response
        """
        # Get request ID from header or generate a new one
        request_id = request.headers.get(self.header_name)
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Set request ID in context
        set_request_id(request_id)
        
        # Process the request
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers[self.header_name] = request_id
        
        return response

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware that adds a correlation ID to each request."""
    
    def __init__(
        self,
        app: ASGIApp,
        header_name: str = "X-Correlation-ID"
    ):
        """
        Initialize the middleware.
        
        Args:
            app: ASGI application
            header_name: Name of the header to use for the correlation ID
        """
        super().__init__(app)
        self.header_name = header_name
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Process the request.
        
        Args:
            request: FastAPI request
            call_next: Function to call the next middleware
            
        Returns:
            FastAPI response
        """
        # Get correlation ID from header or generate a new one
        correlation_id = request.headers.get(self.header_name)
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        
        # Set correlation ID in context
        set_correlation_id(correlation_id)
        
        # Process the request
        response = await call_next(request)
        
        # Add correlation ID to response headers
        response.headers[self.header_name] = correlation_id
        
        return response

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware that collects metrics for each request."""
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Process the request and collect metrics.
        
        Args:
            request: FastAPI request
            call_next: Function to call the next middleware
            
        Returns:
            FastAPI response
        """
        # Get request path and method
        path = request.url.path
        method = request.method
        
        # Start timer
        start_time = time.time()
        
        # Process the request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Record metrics
        MetricsRecorder.record_api_request(
            endpoint=path,
            method=method,
            status_code=response.status_code
        )
        
        # Record duration with the correct status code
        API_REQUEST_DURATION_SECONDS.labels(
            endpoint=path,
            method=method,
            status_code=str(response.status_code)
        ).observe(duration)
        
        return response

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs each request and response."""
    
    def __init__(
        self,
        app: ASGIApp,
        exclude_paths: Optional[list] = None
    ):
        """
        Initialize the middleware.
        
        Args:
            app: ASGI application
            exclude_paths: List of paths to exclude from logging
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/health", "/metrics"]
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Process the request and log details.
        
        Args:
            request: FastAPI request
            call_next: Function to call the next middleware
            
        Returns:
            FastAPI response
        """
        # Get request details
        path = request.url.path
        method = request.method
        
        # Skip logging for excluded paths
        if any(path.startswith(excluded) for excluded in self.exclude_paths):
            return await call_next(request)
        
        # Get IDs
        request_id = get_request_id()
        correlation_id = get_correlation_id()
        
        # Log request
        logger.info(
            f"Request: {method} {path}",
            {
                "request_id": request_id,
                "correlation_id": correlation_id,
                "method": method,
                "path": path,
                "query_params": dict(request.query_params),
                "client_host": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent")
            }
        )
        
        # Start timer
        start_time = time.time()
        
        # Process the request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log response
        logger.info(
            f"Response: {method} {path} - {response.status_code}",
            {
                "request_id": request_id,
                "correlation_id": correlation_id,
                "method": method,
                "path": path,
                "status_code": response.status_code,
                "duration": duration
            }
        )
        
        return response

def setup_middleware(app):
    """
    Set up middleware for the FastAPI application.
    
    Args:
        app: FastAPI application
    """
    # Add middleware in reverse order (last added is executed first)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(CorrelationIdMiddleware)
    app.add_middleware(RequestIdMiddleware)
