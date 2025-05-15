"""
Authentication module for API Gateway.
"""

from .auth_middleware import EnhancedAuthMiddleware

__all__ = ["EnhancedAuthMiddleware"]