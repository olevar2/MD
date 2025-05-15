"""
API Routes

This module aggregates all API routes for the backtesting service.
"""
from fastapi import APIRouter

from app.api.v1.backtest_routes import router as backtest_router
from app.api.v1.health_routes import router as health_router

router = APIRouter()

# Include all routers
router.include_router(backtest_router)
router.include_router(health_router)
