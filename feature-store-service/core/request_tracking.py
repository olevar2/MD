"""
Request tracking middleware for the Feature Store Service.

This middleware adds request tracking capabilities to the FastAPI application,
including request ID generation, performance monitoring, and logging.
"""
import time
import uuid
from typing import Callable, Dict, Any
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
logger = logging.getLogger(__name__)


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for tracking and logging HTTP requests.
    
    This middleware:
    1. Generates a unique request ID for each request
    2. Logs request details (method, path, client)
    3. Measures and logs request processing time
    4. Logs response status and size
    """

    def __init__(self, app: ASGIApp, exclude_paths: list=None,
        request_id_header: str='X-Request-ID'):
        """
        Initialize the middleware.
        
        Args:
            app: The ASGI application
            exclude_paths: List of paths to exclude from tracking
            request_id_header: Header name for the request ID
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or ['/health', '/metrics']
        self.request_id_header = request_id_header

    @async_with_exception_handling
    async def dispatch(self, request: Request, call_next: Callable) ->Response:
        """
        Process the request through the middleware.
        
        Args:
            request: The incoming request
            call_next: Function to call the next middleware/route handler
            
        Returns:
            The response from the route handler
        """
        if any(request.url.path.startswith(path) for path in self.exclude_paths
            ):
            return await call_next(request)
        request_id = request.headers.get(self.request_id_header)
        if not request_id:
            request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        log_context = {'request_id': request_id, 'method': request.method,
            'path': request.url.path, 'client': request.client.host if
            request.client else 'unknown', 'user_agent': request.headers.
            get('user-agent', 'unknown')}
        logger.info(f'Request started: {request.method} {request.url.path}',
            extra=log_context)
        start_time = time.time()
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers[self.request_id_header] = request_id
            response.headers['X-Process-Time'] = str(process_time)
            log_context.update({'status_code': response.status_code,
                'process_time': process_time})
            logger.info(
                f'Request completed: {request.method} {request.url.path} - {response.status_code} in {process_time:.4f}s'
                , extra=log_context)
            return response
        except Exception as e:
            process_time = time.time() - start_time
            log_context.update({'process_time': process_time, 'exception':
                str(e), 'exception_type': e.__class__.__name__})
            logger.error(
                f'Request failed: {request.method} {request.url.path} - {str(e)} in {process_time:.4f}s'
                , extra=log_context, exc_info=True)
            raise
