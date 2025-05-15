"""
API v1

This package provides the v1 API routes for the causal analysis service.
"""

from causal_analysis_service.api.v1.causal_routes import router as causal_router
from causal_analysis_service.api.v1.health_routes import router as health_router

__all__ = [
    'causal_router',
    'health_router'
]