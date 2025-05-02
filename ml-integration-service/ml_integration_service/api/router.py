"""
Main API Router for ML Integration Service.

This module provides the main API router and consolidates all routes
from different components of the service.
"""
from fastapi import APIRouter

from ml_integration_service.api.enhanced_routes import router as enhanced_router
from ml_integration_service.services.optimization_service import OptimizationService
from ml_integration_service.services.model_selection_service import ModelSelectionService

# Create main API router
api_router = APIRouter()

# Include enhanced routes
api_router.include_router(enhanced_router)

# Dependencies
def get_optimization_service():
    """Get optimization service instance."""
    return OptimizationService()

def get_model_selection_service():
    """Get model selection service instance."""
    return ModelSelectionService()
