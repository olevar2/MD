"""
Health check routes.
"""
from fastapi import APIRouter

# Create router
router = APIRouter(tags=["Health"])

@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}