"""
Monitoring & Alerting Service - Main Application

This is the entry point for the Monitoring & Alerting Service, which provides monitoring,
alerting, and visualization capabilities for the Forex Trading Platform.
"""

import os
import logging
import traceback
from typing import Union

import uvicorn
from common_lib.correlation import FastAPICorrelationIdMiddleware
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from prometheus_client import make_asgi_app

# Import error handling package
from monitoring_alerting_service.error import register_exception_handlers
from monitoring_alerting_service.api import CorrelationIdMiddleware
from monitoring_alerting_service.metrics import initialize_error_metrics
from monitoring_alerting_service.notifications import initialize_notification_service

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("monitoring_alerting_service")

# Initialize FastAPI app
app = FastAPI(
    title="Monitoring & Alerting Service",
    description="Monitoring, alerting, and visualization for the Forex Trading Platform",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add correlation ID middleware
app.add_middleware(FastAPICorrelationIdMiddleware)

# Add correlation ID middleware
app.add_middleware(CorrelationIdMiddleware)

# Mount Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Register standardized exception handlers
register_exception_handlers(app)

# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to Monitoring & Alerting Service", "docs_url": "/docs"}

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize service components on startup"""
    logger.info("Starting Monitoring & Alerting Service")

    # Initialize error metrics
    try:
        error_metrics = initialize_error_metrics("monitoring-alerting-service")
        logger.info("Error metrics initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize error metrics: {str(e)}", exc_info=True)

    # Initialize notification service
    try:
        notification_service = initialize_notification_service("monitoring-alerting-service")
        logger.info("Notification service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize notification service: {str(e)}", exc_info=True)

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down Monitoring & Alerting Service")

# Main entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8009))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting Monitoring & Alerting Service on {host}:{port}")
    uvicorn.run(
        "monitoring_alerting_service.main:app",
        host=host,
        port=port,
        reload=os.environ.get("DEBUG", "false").lower() == "true",
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )
