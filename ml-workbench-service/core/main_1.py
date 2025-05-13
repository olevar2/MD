"""
Main module for ML Workbench Service.

This module initializes and starts the ML Workbench Service.
"""

import os
import logging
from fastapi import FastAPI, Depends, HTTPException, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI(
    title="ML Workbench Service",
    description="ML Workbench Service for Forex Trading Platform",
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
        "service": "ML Workbench Service",
        "version": "1.0.0",
        "status": "running",
    }

# Add API prefix
api_prefix = "/api/v1"

# Add API routers
from ml_workbench_service.api.v1 import model_registry_router, model_training_router, model_serving_router, model_monitoring_router, transfer_learning_router

app.include_router(model_registry_router, prefix=f"{api_prefix}/model-registry", tags=["Model Registry"])
app.include_router(model_training_router, prefix=f"{api_prefix}/model-training", tags=["Model Training"])
app.include_router(model_serving_router, prefix=f"{api_prefix}/model-serving", tags=["Model Serving"])
app.include_router(model_monitoring_router, prefix=f"{api_prefix}/model-monitoring", tags=["Model Monitoring"])
app.include_router(transfer_learning_router, prefix=f"{api_prefix}/transfer-learning", tags=["Transfer Learning"])

# Startup event
@app.on_event("startup")
async def startup_event():
    """Startup event."""
    logging.info("Starting ML Workbench Service")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event."""
    logging.info("Shutting down ML Workbench Service")