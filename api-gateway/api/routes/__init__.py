"""
Routes module for API Gateway.
"""

from .proxy import router as proxy_router

__all__ = ["proxy_router"]