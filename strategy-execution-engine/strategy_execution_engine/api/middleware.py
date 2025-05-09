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

from strategy_execution_engine.api.metrics_integration import setup_metrics

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging request information.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request, log information, and pass to the next middleware.

        Args:
            request: The incoming request
            call_next: The next middleware to call

        Returns:
            Response: The response from the next middleware
        """
        # Generate request ID if not present
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Add request ID to request state
        request.state.request_id = request_id

        # Log request
        logger.info(
            f"Request started: {request.method} {request.url.path} "
            f"(ID: {request_id}, Client: {request.client.host if request.client else 'unknown'})"
        )

        # Record start time
        start_time = time.time()

        # Process request
        try:
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            # Log response
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"(ID: {request_id}, Status: {response.status_code}, Time: {process_time:.4f}s)"
            )

            return response
        except Exception as e:
            # Calculate processing time
            process_time = time.time() - start_time

            # Log error
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"(ID: {request_id}, Error: {str(e)}, Time: {process_time:.4f}s)",
                exc_info=True
            )

            # Re-raise exception
            raise

class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting request metrics.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request, collect metrics, and pass to the next middleware.

        Args:
            request: The incoming request
            call_next: The next middleware to call

        Returns:
            Response: The response from the next middleware
        """
        # Record start time
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time

        # Extract path template for grouping similar routes
        path_template = request.scope.get("path", request.url.path)

        # Record metrics (implement actual metrics collection here)
        # Example: prometheus_request_duration.observe(process_time, {"path": path_template, "method": request.method, "status": response.status_code})

        return response

def setup_middleware(app: FastAPI) -> None:
    """
    Set up middleware for the application.

    Args:
        app: FastAPI application instance
    """
    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware)

    # Set up standardized metrics
    setup_metrics(app, service_name="strategy-execution-engine")

    logger.info("Middleware configured")
