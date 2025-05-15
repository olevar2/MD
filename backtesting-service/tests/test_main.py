"""
Test main application for API route testing.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.backtest_routes import router as backtest_router

# Create a test app
app = FastAPI(
    title="Backtesting Service Test",
    description="Test app for backtesting service API routes",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the backtest router
app.include_router(backtest_router)
