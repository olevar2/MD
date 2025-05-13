"""
API Package for Strategy Execution Engine

This package contains API-related functionality for the Strategy Execution Engine,
including routes, middleware, and health checks.
"""

from .routes import setup_routes
from .health import setup_health_routes
from .middleware import setup_middleware

__all__ = [
    "setup_routes",
    "setup_health_routes",
    "setup_middleware"
]
