"""
Rate limiter for the ML Integration Service API.

This module provides a rate limiter for the API endpoints to prevent abuse
and ensure fair usage of the service.
"""
from typing import Dict, Any, Optional, Callable
import time
import logging
import asyncio
from fastapi import Request, HTTPException, status, Depends
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
logger = logging.getLogger(__name__)


from ml_integration_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class RateLimiter:
    """
    Rate limiter for API endpoints.
    
    This class implements a token bucket algorithm for rate limiting.
    """

    def __init__(self, requests_per_minute: int=60, burst_size: int=10,
        key_func: Optional[Callable[[Request], str]]=None):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_minute: Maximum number of requests allowed per minute
            burst_size: Maximum number of requests allowed in a burst
            key_func: Function to extract a key from the request for rate limiting
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.key_func = key_func or self._default_key_func
        self.tokens: Dict[str, float] = {}
        self.last_refill: Dict[str, float] = {}
        self.refill_rate = requests_per_minute / 60.0

    def _default_key_func(self, request: Request) ->str:
        """
        Default function to extract a key from the request.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Key for rate limiting
        """
        forwarded = request.headers.get('X-Forwarded-For')
        if forwarded:
            return forwarded.split(',')[0].strip()
        return request.client.host or 'unknown'

    def _refill_tokens(self, key: str) ->None:
        """
        Refill tokens for a key.
        
        Args:
            key: Key for rate limiting
        """
        now = time.time()
        if key not in self.tokens:
            self.tokens[key] = self.burst_size
            self.last_refill[key] = now
            return
        time_since_refill = now - self.last_refill[key]
        new_tokens = time_since_refill * self.refill_rate
        self.tokens[key] = min(self.tokens[key] + new_tokens, self.burst_size)
        self.last_refill[key] = now

    def _consume_token(self, key: str) ->bool:
        """
        Consume a token for a key.
        
        Args:
            key: Key for rate limiting
            
        Returns:
            True if a token was consumed, False otherwise
        """
        self._refill_tokens(key)
        if self.tokens[key] < 1.0:
            return False
        self.tokens[key] -= 1.0
        return True

    async def __call__(self, request: Request) ->None:
        """
        Check if the request is allowed.
        
        Args:
            request: FastAPI request object
            
        Raises:
            HTTPException: If the request is rate limited
        """
        key = self.key_func(request)
        if not self._consume_token(key):
            logger.warning(f'Rate limit exceeded for {key}')
            raise HTTPException(status_code=status.
                HTTP_429_TOO_MANY_REQUESTS, detail=
                'Rate limit exceeded. Please try again later.')


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting.
    
    This middleware applies rate limiting to all requests.
    """

    def __init__(self, app: ASGIApp, requests_per_minute: int=60,
        burst_size: int=10, key_func: Optional[Callable[[Request], str]]=None):
        """
        Initialize the middleware.
        
        Args:
            app: ASGI application
            requests_per_minute: Maximum number of requests allowed per minute
            burst_size: Maximum number of requests allowed in a burst
            key_func: Function to extract a key from the request for rate limiting
        """
        super().__init__(app)
        self.rate_limiter = RateLimiter(requests_per_minute=
            requests_per_minute, burst_size=burst_size, key_func=key_func)

    @async_with_exception_handling
    async def dispatch(self, request: Request, call_next):
        """
        Dispatch the request.
        
        Args:
            request: FastAPI request object
            call_next: Function to call the next middleware
            
        Returns:
            Response from the next middleware
        """
        try:
            await self.rate_limiter(request)
            return await call_next(request)
        except HTTPException as e:
            if e.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
                headers = {'Retry-After': '60'}
                return HTTPException(status_code=e.status_code, detail=e.
                    detail, headers=headers)
            raise


reconciliation_rate_limiter = RateLimiter(requests_per_minute=30, burst_size=5)
health_rate_limiter = RateLimiter(requests_per_minute=120, burst_size=20)


def get_reconciliation_rate_limiter():
    """
    Get the rate limiter for reconciliation endpoints.
    
    Returns:
        Rate limiter for reconciliation endpoints
    """
    return reconciliation_rate_limiter


def get_health_rate_limiter():
    """
    Get the rate limiter for health endpoints.
    
    Returns:
        Rate limiter for health endpoints
    """
    return health_rate_limiter
