"""
Causal Analysis Service

This module provides the main entry point for the causal analysis service.
"""
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from causal_analysis_service.api.v1.causal_router import router as causal_router
from causal_analysis_service.api.v1.health_router import router as health_router
from causal_analysis_service.utils.correlation_id import CorrelationIdMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Causal Analysis Service",
    description="Service for causal analysis of market data",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add correlation ID middleware
app.add_middleware(CorrelationIdMiddleware)

# Include routers
app.include_router(health_router)
app.include_router(causal_router)

@app.on_event("startup")
async def startup_event():
    """
    Startup event handler.
    """
    logger.info("Starting causal analysis service")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event handler.
    """
    logger.info("Shutting down causal analysis service")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)