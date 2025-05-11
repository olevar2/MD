"""
Main API Router for ML Integration Service.

This module provides the main API router and consolidates all routes
from different components of the service.
"""
from fastapi import APIRouter

from ml_integration_service.api.enhanced_routes import router as enhanced_router
from ml_integration_service.api.v1.cache_api import router as cache_router
from ml_integration_service.api.v1.dashboard_api import router as dashboard_router
from ml_integration_service.api.v1.reconciliation_api import router as reconciliation_router
from ml_integration_service.api.v1.health_api import router as health_router
from ml_integration_service.services.optimization_service import OptimizationService
from ml_integration_service.services.model_selection_service import ModelSelectionService
from ml_integration_service.services.reconciliation_service import ReconciliationService

# Create main API router
api_router = APIRouter()

# Include enhanced routes
api_router.include_router(enhanced_router)

# Include cache routes
api_router.include_router(cache_router)

# Include dashboard routes
api_router.include_router(dashboard_router)

# Include reconciliation routes
api_router.include_router(reconciliation_router)

# Include health routes
api_router.include_router(health_router)

# Dependencies
def get_optimization_service():
    """Get optimization service instance."""
    return OptimizationService()

def get_model_selection_service():
    """Get model selection service instance."""
    return ModelSelectionService()

def get_reconciliation_service():
    """Get reconciliation service instance."""
    return ReconciliationService()
