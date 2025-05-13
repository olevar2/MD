"""
Historical Data Management Service.

This is the main entry point for the Historical Data Management service.
"""

import logging
import os
from typing import Dict

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.api_1 import router as historical_router
from api.api_2 import router as reconciliation_router
from api.dashboard_api import router as reconciliation_dashboard_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Historical Data Management Service",
    description="Service for managing historical data for forex trading",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, this should be restricted
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add routers
app.include_router(historical_router)
app.include_router(reconciliation_router)
app.include_router(reconciliation_dashboard_router)

# Mount static files for dashboard
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
app.mount("/dashboard", StaticFiles(directory=static_dir, html=True), name="dashboard")


# Add exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> Response:
    """Global exception handler."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


# Add health check endpoint
@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


# Add readiness check endpoint
@app.get("/ready")
async def readiness_check() -> Dict[str, str]:
    """Readiness check endpoint."""
    return {"status": "ready"}


# Add startup event handler
@app.on_event("startup")
async def startup_event() -> None:
    """Startup event handler."""
    logger.info("Starting Historical Data Management Service")


# Add shutdown event handler
@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Shutdown event handler."""
    logger.info("Shutting down Historical Data Management Service")


if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))

    # Run the application
    uvicorn.run(
        "data_management_service.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )
