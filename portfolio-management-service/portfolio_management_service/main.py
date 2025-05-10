"""
Portfolio Management Service Main Application.

This service handles portfolio state tracking, balance management, and position tracking.
"""
import os
import logging
import traceback
from typing import Dict, Optional, Union
import asyncio  # Add asyncio import if needed for health check

import uvicorn
from common_lib.correlation import FastAPICorrelationIdMiddleware
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from prometheus_client import make_asgi_app

from core_foundations.utils.logger import get_logger
from core_foundations.api.health_check import add_health_check_to_app
from core_foundations.events.kafka_event_bus import KafkaEventBus
from core_foundations.events.event_schema import Event, EventType
from core_foundations.models.schemas import HealthStatus, HealthCheckResult

# Import error handling package
from portfolio_management_service.error import register_exception_handlers

# Service-specific imports - Updated
from portfolio_management_service.api.router import api_router
# Import new async db functions and session getter
from portfolio_management_service.db.connection import (
    initialize_database,
    dispose_database,
    get_db_session,
    get_engine  # Keep if needed directly, e.g., for health check raw connection
)
# Remove old sync imports: get_db_connection, check_db_connection, init_db

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
APP_NAME = "Portfolio Management Service"
APP_VERSION = "0.1.0"

# Initialize logger
logger = get_logger("portfolio-management-service")

# Initialize application
app = FastAPI(
    title=APP_NAME,
    description="Service for tracking portfolio state, positions, and balances",
    version=APP_VERSION,
    # Add lifespan context manager for startup/shutdown events
    # lifespan=lifespan # Alternative to on_event decorators if preferred
)

# --- Startup and Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    """Application startup logic."""
    logger.info("Starting Portfolio Management Service...")
    try:
        await initialize_database()
        logger.info("Database initialized successfully.")
        # Initialize Kafka Event Bus (if applicable)
        # global event_bus # Assuming event_bus is global or managed elsewhere
        # event_bus = KafkaEventBus(...)
        # await event_bus.start(...)
        # health_checks[1]["check_func"] = event_bus.check_connection # Update Kafka health check
        logger.info("Kafka connection check function updated.")  # Placeholder
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        # Decide if the app should exit or continue in a degraded state
        # For critical components like DB, it might be better to exit
        # raise SystemExit(f"Failed to initialize critical components: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown logic."""
    logger.info("Shutting down Portfolio Management Service...")
    await dispose_database()
    # Shutdown Kafka Event Bus (if applicable)
    # if event_bus:
    #     await event_bus.stop()
    logger.info("Service shutdown complete.")

# --- Health Check ---
async def check_db_connection_async() -> HealthCheckResult:
    """Async check for database connection status."""
    try:
        # Option 1: Try getting a session (simpler)
        async with get_db_session():
             # Option 2: Execute a simple query using the engine directly
             # engine = get_engine()
             # async with engine.connect() as connection:
             #     await connection.execute(text("SELECT 1"))
             #     await connection.commit() # Not strictly needed for SELECT 1
            return HealthCheckResult(status="OK", details="Database connection successful.")
    except Exception as e:
        logger.error(f"Database health check failed: {e}", exc_info=True)
        # Provide more specific error if possible
        error_detail = f"Database connection failed: {type(e).__name__}"
        return HealthCheckResult(status="ERROR", details=error_detail)

# Import middleware
from portfolio_management_service.api.middleware import CorrelationIdMiddleware

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add correlation ID middleware
app.add_middleware(FastAPICorrelationIdMiddleware)

# Add correlation ID middleware
app.add_middleware(CorrelationIdMiddleware)

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Setup health check endpoints - Updated
health_checks = [
    {
        "name": "database_connection",
        "check_func": check_db_connection_async,  # Use the new async check
        "critical": True,
    },
    {
        "name": "kafka_connection",  # Placeholder - update if Kafka is used
        "check_func": lambda: HealthCheckResult(status="OK", details="Kafka connection placeholder OK."),  # Placeholder sync lambda
        # "check_func": lambda: await event_bus.check_connection(), # Example if event_bus has async check
        "critical": True,  # Set based on actual Kafka usage importance
    }
]

# Setup dependencies (if any specific needed for routes)
dependencies: Dict[str, callable] = {}

# Add health check endpoints - Uses the updated health_checks list
add_health_check_to_app(
    app=app,
    service_name=APP_NAME,
    version=APP_VERSION,
    checks=health_checks,
    dependencies=dependencies,  # Pass dependencies if needed by health checks
)

# Register standardized exception handlers
register_exception_handlers(app)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# --- Main execution ---
if __name__ == "__main__":
    # Note: init_db() call removed, handled by initialize_database if configured
    logger.info(f"Starting {APP_NAME} v{APP_VERSION}")
    uvicorn.run(
        "portfolio_management_service.main:app",
        host="0.0.0.0",
        port=8002,  # Example port, adjust as needed
        reload=True,  # Enable reload for development
        log_level="info"
    )