"""
Main module for Data Pipeline Service.

This module initializes and starts the Data Pipeline Service.
"""

import os
import logging
from fastapi import FastAPI, Depends, HTTPException, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI(
    title="Data Pipeline Service",
    description="Data Pipeline Service for Forex Trading Platform",
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
        "service": "Data Pipeline Service",
        "version": "1.0.0",
        "status": "running",
    }

# Add API prefix
api_prefix = "/api/v1"

# Add API routers
# from data_pipeline_service.api.v1 import pipelines_router, executions_router, sources_router, destinations_router

# app.include_router(pipelines_router, prefix=f"{api_prefix}/pipelines", tags=["Pipelines"])
# app.include_router(executions_router, prefix=f"{api_prefix}/executions", tags=["Executions"])
# app.include_router(sources_router, prefix=f"{api_prefix}/sources", tags=["Sources"])
# app.include_router(destinations_router, prefix=f"{api_prefix}/destinations", tags=["Destinations"])

# Startup event
@app.on_event("startup")
async def startup_event():
    """Startup event."""
    logging.info("Starting Data Pipeline Service")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event."""
    logging.info("Shutting down Data Pipeline Service")