"""
Enhanced Authentication Middleware

This module provides enhanced middleware for authentication with support for
multiple authentication methods and role-based access control.
"""

import logging
import time
import jwt
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from common_lib.config.config_manager import ConfigManager
from common_lib.exceptions import AuthenticationError, AuthorizationError

from ..response.standard_response import create_error_response


class EnhancedAuthMiddleware(BaseHTTPMiddleware):
    """
    Enhanced middleware for authentication.

    This middleware validates various authentication methods for authenticated routes
    and implements role-based access control.
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

        # Set default values
        self.secret_key = getattr(self.auth_config, "secret_key", "default_secret_key") if self.auth_config else "default_secret_key"
        self.algorithm = getattr(self.auth_config, "algorithm", "HS256") if self.auth_config else "HS256"
        self.public_paths = getattr(self.auth_config, "public_paths", []) if self.auth_config else []
        self.api_key_paths = getattr(self.auth_config, "api_key_paths", []) if self.auth_config else []
        self.api_keys = getattr(self.auth_config, "api_keys", {}) if self.auth_config else {}
        self.role_permissions = getattr(self.auth_config, "role_permissions", {}) if self.auth_config else {}

        # Always add health check to public paths
        self.public_paths.append("/health")

    def _is_public_path(self, path: str) -> bool:
        """
        Check if a path is public.

        Args:
            path: Request path

        Returns:
            True if the path is public, False otherwise
        """
        # Check exact matches
        if path in self.public_paths:
            return True

        # Check wildcard matches
        for public_path in self.public_paths:
            if public_path.endswith("*") and path.startswith(public_path[:-1]):
                return True

        return False

    def _is_api_key_path(self, path: str) -> bool:
        """
        Check if a path requires API key authentication.

        Args:
            path: Request path

        Returns:
            True if the path requires API key authentication, False otherwise
        """
        # Check exact matches
        if path in self.api_key_paths:
            return True

        # Check wildcard matches
        for api_key_path in self.api_key_paths:
            if api_key_path.endswith("*") and path.startswith(api_key_path[:-1]):
                return True

        return False

    def _validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Validate an API key.

        Args:
            api_key: API key to validate

        Returns:
            API key information if valid, None otherwise
        """
        # Check if API key exists
        if api_key in self.api_keys:
            return self.api_keys[api_key]

        return None

    def _validate_jwt(self, token: str) -> Dict[str, Any]:
        """
        Validate a JWT token.

        Args:
            token: JWT token to validate

        Returns:
            Token payload if valid

        Raises:
            AuthenticationError: If the token is invalid
        """
        try:
            # Validate token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")

    def _check_permissions(self, user_roles: List[str], path: str, method: str) -> bool:
        """
        Check if a user has permission to access a resource.

        Args:
            user_roles: User roles
            path: Request path
            method: Request method

        Returns:
            True if the user has permission, False otherwise
        """
        # Check if role permissions are defined
        if not self.role_permissions:
            # If no role permissions are defined, allow access
            return True

        # Check each role
        for role in user_roles:
            # Check if role exists
            if role not in self.role_permissions:
                continue

            # Get role permissions
            permissions = self.role_permissions[role]

            # Check each permission
            for permission in permissions:
                # Get permission details
                permission_path = permission.get("path", "")
                permission_methods = permission.get("methods", [])

                # Check if path matches
                if permission_path.endswith("*") and path.startswith(permission_path[:-1]):
                    # Wildcard match
                    if not permission_methods or method in permission_methods:
                        return True
                elif permission_path == path:
                    # Exact match
                    if not permission_methods or method in permission_methods:
                        return True

        return False

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

        # Skip authentication for public paths
        if self._is_public_path(request.url.path):
            return await call_next(request)

        # Check if path requires API key authentication
        if self._is_api_key_path(request.url.path):
            # Get API key from header
            api_key = request.headers.get("X-API-Key")

            if not api_key:
                return JSONResponse(
                    status_code=401,
                    content=create_error_response(
                        code="AUTHENTICATION_ERROR",
                        message="API key required",
                        correlation_id=correlation_id,
                        request_id=request_id
                    ).dict()
                )

            # Validate API key
            api_key_info = self._validate_api_key(api_key)

            if not api_key_info:
                return JSONResponse(
                    status_code=401,
                    content=create_error_response(
                        code="AUTHENTICATION_ERROR",
                        message="Invalid API key",
                        correlation_id=correlation_id,
                        request_id=request_id
                    ).dict()
                )

            # Add API key information to request state
            request.state.api_key_info = api_key_info

            # Add service ID to headers
            request.headers.__dict__["_list"].append(
                (b"x-service-id", str(api_key_info.get("service_id", "")).encode())
            )

            # Continue processing
            return await call_next(request)

        # JWT authentication
        # Get token from header
        token = request.headers.get("Authorization")

        if not token:
            return JSONResponse(
                status_code=401,
                content=create_error_response(
                    code="AUTHENTICATION_ERROR",
                    message="Authentication required",
                    correlation_id=correlation_id,
                    request_id=request_id
                ).dict()
            )

        # Remove "Bearer " prefix
        if token.startswith("Bearer "):
            token = token[7:]

        try:
            # Validate token
            payload = self._validate_jwt(token)

            # Check permissions
            user_roles = payload.get("roles", [])
            if not self._check_permissions(user_roles, request.url.path, request.method):
                return JSONResponse(
                    status_code=403,
                    content=create_error_response(
                        code="AUTHORIZATION_ERROR",
                        message="Insufficient permissions",
                        correlation_id=correlation_id,
                        request_id=request_id
                    ).dict()
                )

            # Add user information to request state
            request.state.user = payload

            # Add user ID to headers
            request.headers.__dict__["_list"].append(
                (b"x-user-id", str(payload.get("sub", "")).encode())
            )

            # Add user roles to headers
            request.headers.__dict__["_list"].append(
                (b"x-user-roles", ",".join(user_roles).encode())
            )

            # Continue processing
            return await call_next(request)
        except AuthenticationError as e:
            return JSONResponse(
                status_code=401,
                content=create_error_response(
                    code="AUTHENTICATION_ERROR",
                    message=str(e),
                    correlation_id=correlation_id,
                    request_id=request_id
                ).dict()
            )
        except AuthorizationError as e:
            return JSONResponse(
                status_code=403,
                content=create_error_response(
                    code="AUTHORIZATION_ERROR",
                    message=str(e),
                    correlation_id=correlation_id,
                    request_id=request_id
                ).dict()
            )
        except Exception as e:
            self.logger.error(f"Error during authentication: {str(e)}")
            return JSONResponse(
                status_code=500,
                content=create_error_response(
                    code="INTERNAL_SERVER_ERROR",
                    message="Internal server error",
                    correlation_id=correlation_id,
                    request_id=request_id
                ).dict()
            )