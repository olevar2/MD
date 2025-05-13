"""
Main module for Monitoring Alerting Service.

This module initializes and starts the Monitoring Alerting Service.
"""

import os
import logging
from fastapi import FastAPI, Depends, HTTPException, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI(
    title="Monitoring Alerting Service",
    description="Monitoring Alerting Service for Forex Trading Platform",
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
        "service": "Monitoring Alerting Service",
        "version": "1.0.0",
        "status": "running",
    }

# Add API prefix
api_prefix = "/api/v1"

# Add API routers
from monitoring_alerting_service.api.v1 import alerts_router, dashboards_router, prometheus_router, alertmanager_router, grafana_router, notifications_router

app.include_router(alerts_router, prefix=f"{api_prefix}/alerts", tags=["Alerts"])
app.include_router(dashboards_router, prefix=f"{api_prefix}/dashboards", tags=["Dashboards"])
app.include_router(prometheus_router, prefix=f"{api_prefix}/prometheus", tags=["Prometheus"])
app.include_router(alertmanager_router, prefix=f"{api_prefix}/alertmanager", tags=["Alertmanager"])
app.include_router(grafana_router, prefix=f"{api_prefix}/grafana", tags=["Grafana"])
app.include_router(notifications_router, prefix=f"{api_prefix}/notifications", tags=["Notifications"])

# Startup event
@app.on_event("startup")
async def startup_event():
    """Startup event."""
    logging.info("Starting Monitoring Alerting Service")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event."""
    logging.info("Shutting down Monitoring Alerting Service")