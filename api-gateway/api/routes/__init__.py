"""
Routes module for API Gateway.
"""

from .proxy import router as proxy_router
from .analysis_engine_routes import router as analysis_engine_router

__all__ = ["proxy_router", "analysis_engine_router"]