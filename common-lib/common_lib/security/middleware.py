"""
Security Middleware for FastAPI Applications

This module provides security middleware for FastAPI applications, including:
1. Security logging middleware
2. Rate limiting middleware
3. IP filtering middleware
4. Security headers middleware
"""

import time
import logging
import uuid
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timezone
import ipaddress

from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import JSONResponse

from common_lib.security.monitoring import (
    SecurityMonitor,
    EnhancedSecurityEvent,
    SecurityEventCategory,
    SecurityEventSeverity
)

# Configure logger
logger = logging.getLogger(__name__)


class SecurityLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging security events.

    This middleware logs security events for all requests, including:
    - Authentication events
    - Authorization events
    - API access events
    """

    def __init__(
        self,
        app,
        security_monitor: SecurityMonitor,
        exclude_paths: Optional[List[str]] = None,
        user_id_header: str = "X-User-ID",
        correlation_id_header: str = "X-Correlation-ID",
        session_id_header: str = "X-Session-ID"
    ):
        """
        Initialize the middleware.

        Args:
            app: FastAPI application
            security_monitor: Security monitor
            exclude_paths: List of paths to exclude from logging
            user_id_header: Header for user ID
            correlation_id_header: Header for correlation ID
            session_id_header: Header for session ID
        """
        super().__init__(app)
        self.security_monitor = security_monitor
        self.exclude_paths = exclude_paths or []
        self.user_id_header = user_id_header
        self.correlation_id_header = correlation_id_header
        self.session_id_header = session_id_header

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process the request and log security events.

        Args:
            request: FastAPI request
            call_next: Next middleware or route handler

        Returns:
            Response
        """
        # Skip excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        # Get request information
        path = request.url.path
        method = request.method
        client_host = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("User-Agent", "unknown")

        # Get user ID, correlation ID, and session ID from headers
        user_id = request.headers.get(self.user_id_header, "anonymous")
        correlation_id = request.headers.get(
            self.correlation_id_header,
            str(uuid.uuid4())
        )
        session_id = request.headers.get(self.session_id_header)

        # Add correlation ID to request state
        request.state.correlation_id = correlation_id

        # Start timer
        start_time = time.time()

        # Log request event
        self.security_monitor.log_security_event(
            event_type="api_request",
            category=SecurityEventCategory.API,
            severity=SecurityEventSeverity.INFO,
            user_id=user_id,
            resource=path,
            action=method,
            status="received",
            ip_address=client_host,
            user_agent=user_agent,
            session_id=session_id,
            correlation_id=correlation_id,
            details={
                "path": path,
                "method": method,
                "query_params": dict(request.query_params),
                "headers": {k: v for k, v in request.headers.items() if k.lower() not in ["authorization", "cookie"]}
            }
        )

        try:
            # Check if IP is blacklisted
            if self.security_monitor.is_ip_blacklisted(client_host):
                # Log blocked request
                self.security_monitor.log_security_event(
                    event_type="api_request_blocked",
                    category=SecurityEventCategory.NETWORK,
                    severity=SecurityEventSeverity.WARNING,
                    user_id=user_id,
                    resource=path,
                    action=method,
                    status="blocked",
                    ip_address=client_host,
                    user_agent=user_agent,
                    session_id=session_id,
                    correlation_id=correlation_id,
                    details={
                        "reason": "IP address blacklisted"
                    },
                    tags=["blocked", "blacklist", "ip"]
                )

                return JSONResponse(
                    status_code=403,
                    content={"detail": "Access denied"}
                )

            # Check if user is blacklisted
            if user_id != "anonymous" and self.security_monitor.is_user_blacklisted(user_id):
                # Log blocked request
                self.security_monitor.log_security_event(
                    event_type="api_request_blocked",
                    category=SecurityEventCategory.USER_MANAGEMENT,
                    severity=SecurityEventSeverity.WARNING,
                    user_id=user_id,
                    resource=path,
                    action=method,
                    status="blocked",
                    ip_address=client_host,
                    user_agent=user_agent,
                    session_id=session_id,
                    correlation_id=correlation_id,
                    details={
                        "reason": "User blacklisted"
                    },
                    tags=["blocked", "blacklist", "user"]
                )

                return JSONResponse(
                    status_code=403,
                    content={"detail": "Access denied"}
                )

            # Process request
            response = await call_next(request)

            # Calculate request duration
            duration = time.time() - start_time

            # Get response status
            status_code = response.status_code

            # Determine status
            if status_code < 400:
                status = "success"
                severity = SecurityEventSeverity.INFO
            elif status_code < 500:
                status = "client_error"
                severity = SecurityEventSeverity.WARNING
            else:
                status = "server_error"
                severity = SecurityEventSeverity.ERROR

            # Log response event
            self.security_monitor.log_security_event(
                event_type="api_response",
                category=SecurityEventCategory.API,
                severity=severity,
                user_id=user_id,
                resource=path,
                action=method,
                status=status,
                ip_address=client_host,
                user_agent=user_agent,
                session_id=session_id,
                correlation_id=correlation_id,
                details={
                    "status_code": status_code,
                    "duration": duration
                }
            )

            return response
        except Exception as e:
            # Calculate request duration
            duration = time.time() - start_time

            # Log error event
            self.security_monitor.log_security_event(
                event_type="api_error",
                category=SecurityEventCategory.API,
                severity=SecurityEventSeverity.ERROR,
                user_id=user_id,
                resource=path,
                action=method,
                status="error",
                ip_address=client_host,
                user_agent=user_agent,
                session_id=session_id,
                correlation_id=correlation_id,
                details={
                    "error": str(e),
                    "duration": duration
                }
            )

            # Re-raise exception
            raise


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting requests.

    This middleware implements rate limiting for API requests using a token bucket algorithm.
    It supports different rate limits for different user roles and IP-based rate limiting.
    """

    def __init__(
        self,
        app,
        security_monitor: SecurityMonitor,
        default_rate: int = 60,  # requests per minute
        default_burst: int = 10,
        user_rates: Optional[Dict[str, int]] = None,
        role_rates: Optional[Dict[str, int]] = None,
        exclude_paths: Optional[List[str]] = None,
        user_id_header: str = "X-User-ID",
        role_header: str = "X-User-Role",
        correlation_id_header: str = "X-Correlation-ID",
        session_id_header: str = "X-Session-ID"
    ):
        """
        Initialize the middleware.

        Args:
            app: FastAPI application
            security_monitor: Security monitor
            default_rate: Default rate limit (requests per minute)
            default_burst: Default burst size
            user_rates: Rate limits for specific users
            role_rates: Rate limits for specific roles
            exclude_paths: List of paths to exclude from rate limiting
            user_id_header: Header for user ID
            role_header: Header for user role
            correlation_id_header: Header for correlation ID
            session_id_header: Header for session ID
        """
        super().__init__(app)
        self.security_monitor = security_monitor
        self.default_rate = default_rate
        self.default_burst = default_burst
        self.user_rates = user_rates or {}
        self.role_rates = role_rates or {}
        self.exclude_paths = exclude_paths or []
        self.user_id_header = user_id_header
        self.role_header = role_header
        self.correlation_id_header = correlation_id_header
        self.session_id_header = session_id_header

        # Token buckets for rate limiting
        self.token_buckets: Dict[str, Dict[str, Any]] = {}

        # Progressive rate limiting
        self.failed_attempts: Dict[str, List[float]] = {}

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process the request and apply rate limiting.

        Args:
            request: FastAPI request
            call_next: Next middleware or route handler

        Returns:
            Response
        """
        # Skip excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        # Get request information
        path = request.url.path
        method = request.method
        client_host = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("User-Agent", "unknown")

        # Get user ID, role, correlation ID, and session ID from headers
        user_id = request.headers.get(self.user_id_header, "anonymous")
        user_role = request.headers.get(self.role_header, "anonymous")
        correlation_id = request.headers.get(
            self.correlation_id_header,
            getattr(request.state, "correlation_id", str(uuid.uuid4()))
        )
        session_id = request.headers.get(self.session_id_header)

        # Determine rate limit
        rate_limit = self._get_rate_limit(user_id, user_role)

        # Apply progressive rate limiting for failed attempts
        if client_host in self.failed_attempts:
            # Get recent failed attempts
            now = time.time()
            recent_failures = [t for t in self.failed_attempts[client_host] if t >= now - 3600]  # Last hour

            # Apply progressive rate limiting
            if len(recent_failures) >= 10:
                # Severe rate limiting
                rate_limit = max(1, rate_limit // 10)
            elif len(recent_failures) >= 5:
                # Moderate rate limiting
                rate_limit = max(1, rate_limit // 5)
            elif len(recent_failures) >= 3:
                # Mild rate limiting
                rate_limit = max(1, rate_limit // 2)

        # Check rate limit
        bucket_key = f"ip:{client_host}"
        if user_id != "anonymous":
            bucket_key = f"user:{user_id}"

        if not self._check_rate_limit(bucket_key, rate_limit):
            # Log rate limit exceeded
            self.security_monitor.log_security_event(
                event_type="rate_limit_exceeded",
                category=SecurityEventCategory.API,
                severity=SecurityEventSeverity.WARNING,
                user_id=user_id,
                resource=path,
                action=method,
                status="blocked",
                ip_address=client_host,
                user_agent=user_agent,
                session_id=session_id,
                correlation_id=correlation_id,
                details={
                    "rate_limit": rate_limit,
                    "bucket_key": bucket_key
                },
                tags=["rate_limit", "blocked"]
            )

            # Return rate limit exceeded response
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": "60"}
            )

        # Process request
        response = await call_next(request)

        # Update failed attempts for client errors
        if 400 <= response.status_code < 500:
            if client_host not in self.failed_attempts:
                self.failed_attempts[client_host] = []

            self.failed_attempts[client_host].append(time.time())

            # Limit the size of the failed attempts list
            self.failed_attempts[client_host] = self.failed_attempts[client_host][-100:]

        return response

    def _get_rate_limit(self, user_id: str, user_role: str) -> int:
        """
        Get rate limit for user.

        Args:
            user_id: User ID
            user_role: User role

        Returns:
            Rate limit (requests per minute)
        """
        # Check user-specific rate limit
        if user_id in self.user_rates:
            return self.user_rates[user_id]

        # Check role-specific rate limit
        if user_role in self.role_rates:
            return self.role_rates[user_role]

        # Use default rate limit
        return self.default_rate

    def _check_rate_limit(self, key: str, rate: int) -> bool:
        """
        Check rate limit for key.

        Args:
            key: Rate limit key
            rate: Rate limit (requests per minute)

        Returns:
            True if request is allowed, False otherwise
        """
        now = time.time()

        # Initialize token bucket if it doesn't exist
        if key not in self.token_buckets:
            self.token_buckets[key] = {
                "tokens": self.default_burst,
                "last_refill": now,
                "rate": rate / 60.0  # Convert to tokens per second
            }

        bucket = self.token_buckets[key]

        # Update rate if it has changed
        bucket["rate"] = rate / 60.0

        # Refill tokens
        time_since_refill = now - bucket["last_refill"]
        new_tokens = time_since_refill * bucket["rate"]
        bucket["tokens"] = min(bucket["tokens"] + new_tokens, self.default_burst)
        bucket["last_refill"] = now

        # Check if there are tokens available
        if bucket["tokens"] < 1.0:
            return False

        # Consume a token
        bucket["tokens"] -= 1.0

        return True


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware for adding security headers to responses.

    This middleware adds security headers to all responses, including:
    - Content-Security-Policy
    - X-Content-Type-Options
    - X-Frame-Options
    - X-XSS-Protection
    - Strict-Transport-Security
    - Referrer-Policy
    """

    def __init__(
        self,
        app,
        csp: Optional[str] = None,
        hsts_max_age: int = 31536000,  # 1 year
        exclude_paths: Optional[List[str]] = None
    ):
        """
        Initialize the middleware.

        Args:
            app: FastAPI application
            csp: Content Security Policy
            hsts_max_age: HSTS max age in seconds
            exclude_paths: List of paths to exclude from security headers
        """
        super().__init__(app)
        self.csp = csp or "default-src 'self'; script-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
        self.hsts_max_age = hsts_max_age
        self.exclude_paths = exclude_paths or []

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process the request and add security headers to the response.

        Args:
            request: FastAPI request
            call_next: Next middleware or route handler

        Returns:
            Response
        """
        # Process request
        response = await call_next(request)

        # Skip excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return response

        # Add security headers
        response.headers["Content-Security-Policy"] = self.csp
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = f"max-age={self.hsts_max_age}; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        return response
