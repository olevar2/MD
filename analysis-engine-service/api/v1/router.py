"""
API Router for Analysis Engine Service.

This module configures the API router for the Analysis Engine Service,
including all the API endpoints.
"""

from fastapi import APIRouter

from analysis_engine.api.v1 import (
    health,
    indicators,
    patterns,
    analysis,
    integrated_analysis
)


# Create the main API router
api_router = APIRouter(prefix="/api/v1")

# Include all the API routers
api_router.include_router(health.router)
api_router.include_router(indicators.router)
api_router.include_router(patterns.router)
api_router.include_router(analysis.router)
api_router.include_router(integrated_analysis.router)
