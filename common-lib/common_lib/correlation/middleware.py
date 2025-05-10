"""
Correlation ID Middleware

This module provides middleware implementations for adding correlation ID
support to different web frameworks.
"""

import logging
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from common_lib.correlation.correlation_id import (
    generate_correlation_id,
    set_correlation_id,
    clear_correlation_id,
    CORRELATION_ID_HEADER
)

logger = logging.getLogger(__name__)


class FastAPICorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for correlation ID propagation.

    This middleware:
    1. Extracts correlation ID from request headers if present
    2. Generates a new correlation ID if not present
    3. Sets the correlation ID in request state
    4. Sets the correlation ID in thread-local and async context
    5. Adds the correlation ID to response headers
    6. Clears the correlation ID after the request is processed
    """

    def __init__(
        self,
        app: ASGIApp,
        header_name: str = CORRELATION_ID_HEADER,
        always_generate_if_missing: bool = True
    ):
        """
        Initialize the middleware.

        Args:
            app: The ASGI application
            header_name: The name of the correlation ID header
            always_generate_if_missing: Whether to generate a new correlation ID if not present
        """
        super().__init__(app)
        self.header_name = header_name
        self.always_generate_if_missing = always_generate_if_missing

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and add correlation ID.

        Args:
            request: The incoming request
            call_next: The next middleware or route handler

        Returns:
            The response
        """
        # Extract correlation ID from headers
        correlation_id = request.headers.get(self.header_name)

        # Generate a new correlation ID if not present and configured to do so
        if not correlation_id and self.always_generate_if_missing:
            correlation_id = generate_correlation_id()
            logger.debug(f"Generated new correlation ID: {correlation_id}")
        elif correlation_id:
            logger.debug(f"Using existing correlation ID: {correlation_id}")

        # Store correlation ID in request state and context
        if correlation_id:
            request.state.correlation_id = correlation_id
            set_correlation_id(correlation_id)

        try:
            # Process the request
            response = await call_next(request)

            # Add correlation ID to response headers if present
            if correlation_id:
                response.headers[self.header_name] = correlation_id

            return response
        finally:
            # Clear correlation ID from context
            clear_correlation_id()


# Factory function for easy middleware creation
def create_correlation_id_middleware(
    framework: str = "fastapi",
    app: Optional[ASGIApp] = None,
    **kwargs
) -> BaseHTTPMiddleware:
    """
    Create a correlation ID middleware for the specified framework.

    Args:
        framework: The web framework to create middleware for
        app: The ASGI application (required for FastAPI)
        **kwargs: Additional arguments for the middleware

    Returns:
        Correlation ID middleware instance

    Raises:
        ValueError: If the framework is not supported
    """
    if framework.lower() == "fastapi":
        if app is None:
            from fastapi import FastAPI
            app = FastAPI()
        return FastAPICorrelationIdMiddleware(app, **kwargs)
    else:
        raise ValueError(f"Unsupported framework: {framework}")
