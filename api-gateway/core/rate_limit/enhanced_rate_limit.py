"""
Enhanced Rate Limiting Middleware

This module provides enhanced middleware for rate limiting with support for
different rate limits for different user roles and API keys.
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from common_lib.config.config_manager import ConfigManager

from ..response.standard_response import create_error_response


class TokenBucket:
    """
    Token bucket rate limiter.

    This class implements a token bucket algorithm for rate limiting.
    """

    def __init__(self, rate: float, capacity: float):
        """
        Initialize the token bucket.

        Args:
            rate: Token refill rate (tokens per second)
            capacity: Maximum number of tokens
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = asyncio.Lock()

    async def consume(self, tokens: float = 1.0) -> bool:
        """
        Consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False otherwise
        """
        async with self.lock:
            # Refill tokens
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_refill = now

            # Check if enough tokens are available
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            else:
                return False


class EnhancedRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Enhanced middleware for rate limiting.

    This middleware implements rate limiting with support for different rate limits
    for different user roles and API keys.
    """

    def __init__(self, app):
        """
        Initialize the middleware.

        Args:
            app: FastAPI application
        """
        super().__init__(app)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config_manager = ConfigManager()

        # Get configuration
        try:
            service_specific = self.config_manager.get_service_specific_config()
            if hasattr(service_specific, "rate_limit"):
                self.rate_limit_config = getattr(service_specific, "rate_limit")
            else:
                self.rate_limit_config = None
        except Exception as e:
            self.logger.warning(f"Error getting rate limit configuration: {str(e)}")
            self.rate_limit_config = None

        # Set default values
        self.enabled = getattr(self.rate_limit_config, "enabled", True) if self.rate_limit_config else True
        self.default_limit = getattr(self.rate_limit_config, "limit", 100) if self.rate_limit_config else 100
        self.default_window = getattr(self.rate_limit_config, "window", 60) if self.rate_limit_config else 60
        self.exempt_paths = getattr(self.rate_limit_config, "exempt_paths", []) if self.rate_limit_config else []
        self.role_limits = getattr(self.rate_limit_config, "role_limits", {}) if self.rate_limit_config else {}
        self.api_key_limits = getattr(self.rate_limit_config, "api_key_limits", {}) if self.rate_limit_config else {}

        # Calculate default rate and capacity
        self.default_rate = self.default_limit / self.default_window
        self.default_capacity = self.default_limit

        # Create rate limiters
        self.limiters = {}

        # Always add health check to exempt paths
        self.exempt_paths.append("/health")

    def _is_exempt_path(self, path: str) -> bool:
        """
        Check if a path is exempt from rate limiting.

        Args:
            path: Request path

        Returns:
            True if the path is exempt, False otherwise
        """
        # Check exact matches
        if path in self.exempt_paths:
            return True

        # Check wildcard matches
        for exempt_path in self.exempt_paths:
            if exempt_path.endswith("*") and path.startswith(exempt_path[:-1]):
                return True

        return False

    def _get_client_id(self, request: Request) -> str:
        """
        Get a unique identifier for the client.

        Args:
            request: FastAPI request

        Returns:
            Client identifier
        """
        # Check if user is authenticated
        user = getattr(request.state, "user", None)
        if user:
            # Use user ID as client identifier
            return f"user:{user.get('sub', '')}"

        # Check if API key is used
        api_key_info = getattr(request.state, "api_key_info", None)
        if api_key_info:
            # Use service ID as client identifier
            return f"service:{api_key_info.get('service_id', '')}"

        # Use IP address as client identifier
        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"

    def _get_rate_and_capacity(self, request: Request) -> tuple:
        """
        Get rate and capacity for a request.

        Args:
            request: FastAPI request

        Returns:
            Tuple of (rate, capacity)
        """
        # Check if user is authenticated
        user = getattr(request.state, "user", None)
        if user:
            # Get user roles
            roles = user.get("roles", [])

            # Check each role for rate limits
            for role in roles:
                if role in self.role_limits:
                    # Get role rate limit
                    role_limit = self.role_limits[role]
                    limit = role_limit.get("limit", self.default_limit)
                    window = role_limit.get("window", self.default_window)
                    return limit / window, limit

        # Check if API key is used
        api_key_info = getattr(request.state, "api_key_info", None)
        if api_key_info:
            # Get service ID
            service_id = api_key_info.get("service_id", "")

            # Check if service has rate limits
            if service_id in self.api_key_limits:
                # Get service rate limit
                service_limit = self.api_key_limits[service_id]
                limit = service_limit.get("limit", self.default_limit)
                window = service_limit.get("window", self.default_window)
                return limit / window, limit

        # Use default rate and capacity
        return self.default_rate, self.default_capacity

    async def dispatch(self, request: Request, call_next):
        """
        Process the request.

        Args:
            request: FastAPI request
            call_next: Next middleware or route handler

        Returns:
            Response
        """
        # Get correlation ID and request ID
        correlation_id = request.headers.get("X-Correlation-ID", "")
        request_id = request.headers.get("X-Request-ID", "")

        # Skip rate limiting if disabled or exempt path
        if not self.enabled or self._is_exempt_path(request.url.path):
            return await call_next(request)

        # Get client identifier
        client_id = self._get_client_id(request)

        # Get rate and capacity
        rate, capacity = self._get_rate_and_capacity(request)

        # Get or create rate limiter
        if client_id not in self.limiters:
            self.limiters[client_id] = TokenBucket(rate, capacity)

        # Try to consume a token
        if not await self.limiters[client_id].consume():
            # Rate limit exceeded
            self.logger.warning(f"Rate limit exceeded for client {client_id}")
            return JSONResponse(
                status_code=429,
                content=create_error_response(
                    code="RATE_LIMIT_EXCEEDED",
                    message="Rate limit exceeded",
                    correlation_id=correlation_id,
                    request_id=request_id,
                    details={
                        "limit": capacity,
                        "window": capacity / rate,
                        "retry_after": 1.0 / rate
                    }
                ).dict(),
                headers={
                    "Retry-After": str(int(1.0 / rate))
                }
            )

        # Continue processing
        return await call_next(request)