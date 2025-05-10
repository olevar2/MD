"""
API v1 package for Analysis Engine Service.

This package contains all v1 API endpoints for the Analysis Engine Service.
"""

from analysis_engine.api.v1.health import router as health_router

__all__ = ["health_router"]