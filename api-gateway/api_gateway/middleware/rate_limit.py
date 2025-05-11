"""
Rate Limiting Middleware

This module provides middleware for rate limiting.
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Callable, Awaitable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from common_lib.config.config_manager import ConfigManager


class RateLimiter:
    """
    Rate limiter.
    
    This class implements a token bucket algorithm for rate limiting.
    """
    
    def __init__(self, rate: float, capacity: float):
        """
        Initialize the rate limiter.
        
        Args:
            rate: Rate at which tokens are added to the bucket (tokens per second)
            capacity: Maximum capacity of the bucket
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_time = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: float = 1.0) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens were acquired, False otherwise
        """
        async with self.lock:
            # Update tokens
            current_time = time.time()
            elapsed = current_time - self.last_time
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_time = current_time
            
            # Check if enough tokens
            if tokens <= self.tokens:
                self.tokens -= tokens
                return True
            else:
                return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting.
    
    This middleware limits the rate of requests from clients.
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
        self.rate = getattr(self.rate_limit_config, "rate", 10.0) if self.rate_limit_config else 10.0
        self.capacity = getattr(self.rate_limit_config, "capacity", 20.0) if self.rate_limit_config else 20.0
        self.exempt_paths = getattr(self.rate_limit_config, "exempt_paths", []) if self.rate_limit_config else []
        
        # Always add health check to exempt paths
        self.exempt_paths.append("/health")
        
        # Create rate limiters
        self.limiters = {}
    
    async def dispatch(self, request: Request, call_next):
        """
        Process the request.
        
        Args:
            request: FastAPI request
            call_next: Next middleware or route handler
            
        Returns:
            Response
        """
        # Skip rate limiting if disabled or exempt path
        if not self.enabled or self._is_exempt_path(request.url.path):
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Get or create rate limiter
        if client_id not in self.limiters:
            self.limiters[client_id] = RateLimiter(self.rate, self.capacity)
        
        # Try to acquire token
        if await self.limiters[client_id].acquire():
            # Continue processing
            return await call_next(request)
        else:
            # Rate limit exceeded
            self.logger.warning(f"Rate limit exceeded for client: {client_id}")
            
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"}
            )
    
    def _get_client_id(self, request: Request) -> str:
        """
        Get client identifier.
        
        Args:
            request: FastAPI request
            
        Returns:
            Client identifier
        """
        # Try to get user ID from request state
        user_id = getattr(request.state, "user", {}).get("sub", "")
        
        if user_id:
            return f"user:{user_id}"
        
        # Use client IP address
        client_host = request.client.host if request.client else "unknown"
        
        return f"ip:{client_host}"
    
    def _is_exempt_path(self, path: str) -> bool:
        """
        Check if a path is exempt from rate limiting.
        
        Args:
            path: Path to check
            
        Returns:
            True if the path is exempt, False otherwise
        """
        # Check exact matches
        if path in self.exempt_paths:
            return True
        
        # Check prefix matches
        for exempt_path in self.exempt_paths:
            if exempt_path.endswith("*") and path.startswith(exempt_path[:-1]):
                return True
        
        return False