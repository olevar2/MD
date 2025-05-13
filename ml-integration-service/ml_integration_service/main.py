"""
Main module for ML Integration Service.

This module initializes and starts the ML Integration Service.
"""

import os
import logging
from fastapi import FastAPI, Depends, HTTPException, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI(
    title="ML Integration Service",
    description="ML Integration Service for Forex Trading Platform",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add health check endpoint
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}

# Add root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "ML Integration Service",
        "version": "1.0.0",
        "status": "running",
    }

# Add API prefix
api_prefix = "/api/v1"

# Add API routers
# from ml_integration_service.api.v1 import predict_router, feature_importance_router, model_router

# app.include_router(predict_router, prefix=f"{api_prefix}", tags=["Predict"])
# app.include_router(feature_importance_router, prefix=f"{api_prefix}", tags=["Feature Importance"])
# app.include_router(model_router, prefix=f"{api_prefix}/models", tags=["Models"])

# Startup event
@app.on_event("startup")
async def startup_event():
    """Startup event."""
    logging.info("Starting ML Integration Service")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event."""
    logging.info("Shutting down ML Integration Service")