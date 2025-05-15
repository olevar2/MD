"""
Middleware components for the chat service
"""
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from typing import Callable, Awaitable
import uuid
import time
import logging
from ..config.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Middleware for handling correlation IDs."""
    
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Add correlation ID to request and response.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/endpoint
            
        Returns:
            Response with correlation ID header
        """
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        
        return response

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses."""
    
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Log request and response details.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/endpoint
            
        Returns:
            Response after logging
        """
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "correlation_id": getattr(request.state, "correlation_id", None),
                "path": request.url.path,
                "method": request.method,
                "client_ip": request.client.host if request.client else None
            }
        )
        
        response = await call_next(request)
        
        # Calculate request duration
        duration = time.time() - start_time
        
        # Log response
        logger.info(
            f"Request completed: {request.method} {request.url.path} - {response.status_code}",
            extra={
                "correlation_id": getattr(request.state, "correlation_id", None),
                "path": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "duration": duration
            }
        )
        
        return response

class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication."""
    
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Validate API key in request.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/endpoint
            
        Returns:
            Response after API key validation
        """
        if request.url.path.startswith("/api/"):
            api_key = request.headers.get(settings.API_KEY_NAME)
            if not api_key:
                return Response(
                    content="API key missing",
                    status_code=401,
                    headers={"WWW-Authenticate": settings.API_KEY_NAME}
                )
            
            # Here you would typically validate the API key against your database or cache
            # For now, we'll use a simple check against the configured secret key
            if api_key != settings.SECRET_KEY:
                return Response(
                    content="Invalid API key",
                    status_code=401,
                    headers={"WWW-Authenticate": settings.API_KEY_NAME}
                )
        
        return await call_next(request)

def setup_middleware(app) -> None:
    """Setup middleware for the application.
    
    Args:
        app: FastAPI application instance
    """
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    
    # Add correlation ID middleware
    app.add_middleware(CorrelationIDMiddleware)
    
    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware)
    
    # Add API key middleware
    app.add_middleware(APIKeyMiddleware)