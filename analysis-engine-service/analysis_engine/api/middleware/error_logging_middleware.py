"""
Error Logging Middleware

This middleware ensures that all errors are properly logged with context information.
It captures exceptions, logs them with structured data, and re-raises them for the
exception handlers to process.
"""

import time
import traceback
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from analysis_engine.core.logging import get_logger
from analysis_engine.core.exceptions_bridge import ForexTradingPlatformError

logger = get_logger(__name__)

class ErrorLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to ensure all errors are properly logged with context information.
    
    This middleware:
    1. Captures exceptions during request processing
    2. Logs them with structured data including correlation ID, request details, etc.
    3. Re-raises the exceptions for the exception handlers to process
    """
    
    def __init__(self, app: ASGIApp):
        """
        Initialize the middleware.
        
        Args:
            app: The ASGI application
        """
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next):
        """
        Process the request and log any errors.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            The response from the next middleware or route handler
        """
        start_time = time.time()
        
        # Get correlation ID from request state (set by CorrelationIdMiddleware)
        correlation_id = getattr(request.state, "correlation_id", "unknown")
        
        # Prepare context for logging
        context = {
            "correlation_id": correlation_id,
            "method": request.method,
            "path": request.url.path,
            "client_host": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("User-Agent", "unknown")
        }
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Log request completion time
            process_time = time.time() - start_time
            logger.debug(
                f"Request completed in {process_time:.3f}s",
                extra={
                    **context,
                    "status_code": response.status_code,
                    "process_time": process_time
                }
            )
            
            return response
            
        except ForexTradingPlatformError as exc:
            # Log platform-specific errors with structured data
            process_time = time.time() - start_time
            
            # Add correlation ID to exception if it doesn't have one
            if hasattr(exc, "correlation_id") and not exc.correlation_id:
                exc.correlation_id = correlation_id
            
            # Get error details
            error_details = getattr(exc, "details", {}) or {}
            
            logger.error(
                f"Platform error during request processing: {exc.message}",
                extra={
                    **context,
                    "error_type": exc.__class__.__name__,
                    "error_code": getattr(exc, "error_code", "UNKNOWN"),
                    "error_details": error_details,
                    "process_time": process_time
                }
            )
            
            # Re-raise the exception for the exception handlers
            raise
            
        except Exception as exc:
            # Log unexpected errors with traceback
            process_time = time.time() - start_time
            
            logger.error(
                f"Unexpected error during request processing: {str(exc)}",
                extra={
                    **context,
                    "error_type": exc.__class__.__name__,
                    "traceback": traceback.format_exc(),
                    "process_time": process_time
                }
            )
            
            # Re-raise the exception for the exception handlers
            raise
