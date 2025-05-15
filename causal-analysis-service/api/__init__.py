"""
API

This package provides the API routes for the causal analysis service.
"""

from causal_analysis_service.api.v1 import causal_router, health_router

__all__ = [
    'causal_router',
    'health_router'
]