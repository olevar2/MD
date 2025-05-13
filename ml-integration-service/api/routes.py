"""
API route definitions.
"""

from fastapi import APIRouter, Depends, FastAPI

def setup_routes(app: FastAPI):
    """
    Set up API routes.
    
    Args:
        app: FastAPI application
    """
    # Create routers
    main_router = APIRouter()
    
    # Add routes to main router
    
    # Include routers in app
    app.include_router(main_router, prefix="/api")
