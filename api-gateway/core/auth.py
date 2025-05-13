"""
Authentication Middleware

This module provides middleware for authentication.
"""

import logging
import time
import jwt
from typing import Dict, Any, Optional, List, Callable, Awaitable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from common_lib.config.config_manager import ConfigManager


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware for authentication.

    This middleware validates JWT tokens for authenticated routes.
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
            if hasattr(service_specific, "auth"):
                self.auth_config = getattr(service_specific, "auth")
            else:
                self.auth_config = None
        except Exception as e:
            self.logger.warning(f"Error getting auth configuration: {str(e)}")
            self.auth_config = None

        # Get secret key from environment variable
        import os
        from dotenv import load_dotenv

        # Load environment variables from .env file
        load_dotenv()

        # Set default values
        default_secret = os.getenv("JWT_SECRET_KEY")
        if not default_secret:
            self.logger.warning("JWT_SECRET_KEY environment variable not set. Using a random secret key.")
            import secrets
            default_secret = secrets.token_hex(32)

        self.secret_key = getattr(self.auth_config, "secret_key", default_secret) if self.auth_config else default_secret
        self.algorithm = getattr(self.auth_config, "algorithm", "HS256") if self.auth_config else "HS256"
        self.public_paths = getattr(self.auth_config, "public_paths", []) if self.auth_config else []

        # Always add health check to public paths
        self.public_paths.append("/health")

    async def dispatch(self, request: Request, call_next):
        """
        Process the request.

        Args:
            request: FastAPI request
            call_next: Next middleware or route handler

        Returns:
            Response
        """
        # Skip authentication for public paths
        if self._is_public_path(request.url.path):
            return await call_next(request)

        # Get token from header
        token = request.headers.get("Authorization")

        if not token:
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required"}
            )

        # Remove "Bearer " prefix
        if token.startswith("Bearer "):
            token = token[7:]

        try:
            # Validate token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Add user information to request state
            request.state.user = payload

            # Add user ID to headers
            request.headers.__dict__["_list"].append(
                (b"x-user-id", str(payload.get("sub", "")).encode())
            )

            # Continue processing
            return await call_next(request)
        except jwt.ExpiredSignatureError:
            return JSONResponse(
                status_code=401,
                content={"detail": "Token has expired"}
            )
        except jwt.InvalidTokenError:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid token"}
            )

    def _is_public_path(self, path: str) -> bool:
        """
        Check if a path is public.

        Args:
            path: Path to check

        Returns:
            True if the path is public, False otherwise
        """
        # Check exact matches
        if path in self.public_paths:
            return True

        # Check prefix matches
        for public_path in self.public_paths:
            if public_path.endswith("*") and path.startswith(public_path[:-1]):
                return True

        return False