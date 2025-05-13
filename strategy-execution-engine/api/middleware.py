"""
Middleware for Strategy Execution Engine

This module provides middleware for the Strategy Execution Engine.
"""
import time
import logging
import uuid
from typing import Callable, Dict, Any
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from adapters.metrics_integration import setup_metrics
logger = logging.getLogger(__name__)


from core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging request information.
    """

    @async_with_exception_handling
    async def dispatch(self, request: Request, call_next: Callable) ->Response:
        """
        Process the request, log information, and pass to the next middleware.

        Args:
            request: The incoming request
            call_next: The next middleware to call

        Returns:
            Response: The response from the next middleware
        """
        request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
        request.state.request_id = request_id
        logger.info(
            f"Request started: {request.method} {request.url.path} (ID: {request_id}, Client: {request.client.host if request.client else 'unknown'})"
            )
        start_time = time.time()
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers['X-Request-ID'] = request_id
            logger.info(
                f'Request completed: {request.method} {request.url.path} (ID: {request_id}, Status: {response.status_code}, Time: {process_time:.4f}s)'
                )
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f'Request failed: {request.method} {request.url.path} (ID: {request_id}, Error: {str(e)}, Time: {process_time:.4f}s)'
                , exc_info=True)
            raise


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting request metrics.
    """

    async def dispatch(self, request: Request, call_next: Callable) ->Response:
        """
        Process the request, collect metrics, and pass to the next middleware.

        Args:
            request: The incoming request
            call_next: The next middleware to call

        Returns:
            Response: The response from the next middleware
        """
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        path_template = request.scope.get('path', request.url.path)
        return response


def setup_middleware(app: FastAPI) ->None:
    """
    Set up middleware for the application.

    Args:
        app: FastAPI application instance
    """
    app.add_middleware(RequestLoggingMiddleware)
    setup_metrics(app, service_name='strategy-execution-engine')
    logger.info('Middleware configured')
