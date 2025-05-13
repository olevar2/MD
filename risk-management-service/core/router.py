"""
Main router for the Risk Management Service API.

This module creates the main FastAPI router and includes all API endpoints for the Risk Management Service.
"""
from fastapi import APIRouter
from api.risk_management_api import router as risk_management_router

router = APIRouter()

# Include all v1 API routers
router.include_router(risk_management_router)

# Additional health check endpoints
@router.get("/health", tags=["Health"])
async def health_check():
    """Perform a health check of the Risk Management Service."""
    return {"status": "ok", "service": "risk-management-service"}
