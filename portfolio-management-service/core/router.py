"""
API Router Module.

Main router for the Portfolio Management Service API.
"""
from fastapi import APIRouter

from portfolio_management_service.api.v1 import positions, accounts, portfolio_api

# Create main API router
api_router = APIRouter()

# Include routers for different resources
api_router.include_router(
    positions.router,
    prefix="/positions",
    tags=["Positions"]
)

api_router.include_router(
    accounts.router,
    prefix="/accounts",
    tags=["Accounts"]
)

api_router.include_router(
    portfolio_api.router,
    tags=["Portfolio"]
)