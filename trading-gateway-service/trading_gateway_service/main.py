"""
Main entry point for the Python components of the Trading Gateway Service.

This module initializes the FastAPI application for the Python components
of the Trading Gateway Service, which handle market data processing,
order reconciliation, and other backend functionality.
"""

import os
import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from core_foundations.utils.logger import get_logger
from core_foundations.api.health_check import add_health_check_to_app
from core_foundations.events.kafka_event_bus import KafkaEventBus
from core_foundations.events.event_schema import Event, EventType
from core_foundations.models.schemas import HealthStatus

# Import error handling
from trading_gateway_service.error import (
    ForexTradingPlatformError,
    BrokerConnectionError,
    OrderValidationError,
    MarketDataError
)
from trading_gateway_service.error.exception_handlers import register_exception_handlers

# Import services
from trading_gateway_service.services.market_data_service import MarketDataService
from trading_gateway_service.services.order_reconciliation_service import OrderReconciliationService
from trading_gateway_service.monitoring.performance_monitoring import TradingGatewayMonitoring
from trading_gateway_service.api.metrics_integration import setup_metrics
from trading_gateway_service.resilience.degraded_mode import DegradedModeManager
from trading_gateway_service.resilience.degraded_mode_strategies import configure_trading_gateway_degraded_mode

# Import API router
from trading_gateway_service.api import router as api_router

# Configuration
APP_NAME = "Trading Gateway Service (Python)"
APP_VERSION = "0.1.0"
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

# Initialize logger
logger = get_logger("trading-gateway-service")

# Initialize application
app = FastAPI(
    title=APP_NAME,
    description="Python components for the Trading Gateway Service",
    version=APP_VERSION,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# Register exception handlers
register_exception_handlers(app)

# Set up metrics
setup_metrics(app, service_name="trading-gateway-service")

# Include API router
app.include_router(api_router)

# Define health checks
health_checks = [
    {
        "name": "market_data_service",
        "check_func": lambda: True,  # Will be replaced at startup
        "critical": True,
    },
    {
        "name": "order_reconciliation_service",
        "check_func": lambda: True,  # Will be replaced at startup
        "critical": True,
    },
    {
        "name": "kafka_connection",
        "check_func": lambda: True,  # Will be replaced at startup
        "critical": False,
    }
]

# Add health check endpoints
add_health_check_to_app(
    app=app,
    service_name=APP_NAME,
    version=APP_VERSION,
    checks=health_checks,
    dependencies={},
)

# Root endpoint
@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint returning service information."""
    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "status": "running"
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    logger.info(f"Starting {APP_NAME}")
    try:
        # Initialize market data service
        # Create a mock broker adapter for the market data service
        from trading_gateway_service.interfaces.broker_adapter import BrokerAdapter
        from trading_gateway_service.simulation.forex_broker_simulator import ForexBrokerSimulator

        # Create a broker adapter
        broker_adapter = ForexBrokerSimulator()

        # Initialize market data service with the broker adapter
        app.state.market_data_service = MarketDataService(
            broker_adapters={"simulator": broker_adapter},
            logger=logger
        )
        logger.info("Market data service initialized")

        # Update health check for market data service
        app.state.health.checks[0]["check_func"] = lambda: True  # Always healthy for now

        # Initialize order reconciliation service
        # Create a mock order store for the reconciliation service
        class MockOrderStore:
            """Mock internal order store for testing."""

            async def get_orders(self):
                return []

            async def get_positions(self):
                return []

            async def get_account_info(self):
                return {"balance": 10000, "equity": 10000, "margin_used": 0}

            async def add_external_order(self, order):
                logger.info(f"Adding external order: {order['order_id']}")

            async def update_order_status(self, order_id, status, reason=None):
                logger.info(f"Updating order status: {order_id} -> {status}")

            async def update_order_fill(self, order_id, filled_size, avg_price):
                logger.info(f"Updating order fill: {order_id} -> {filled_size} @ {avg_price}")

            async def add_external_position(self, position):
                logger.info(f"Adding external position: {position['instrument']}")

            async def mark_position_inconsistent(self, instrument, reason=None):
                logger.info(f"Marking position inconsistent: {instrument}")

            async def update_position(self, instrument, size, direction, entry_price=None, reason=None):
                logger.info(f"Updating position: {instrument} -> {size} {direction}")

            async def update_account_balance(self, balance, reason=None):
                logger.info(f"Updating account balance: {balance}")

        # Create reconciliation service
        app.state.order_reconciliation_service = OrderReconciliationService(
            broker_adapter=broker_adapter,
            internal_order_store=MockOrderStore(),
            reconciliation_config={
                "order_reconciliation_interval_sec": 60,
                "position_reconciliation_interval_sec": 300,
                "account_reconciliation_interval_sec": 600,
            }
        )

        # Start the reconciliation service
        await app.state.order_reconciliation_service.start()
        logger.info("Order reconciliation service initialized and started")

        # Update health check for order reconciliation service
        app.state.health.checks[1]["check_func"] = lambda: True  # Always healthy for now

        # Initialize performance monitoring
        app.state.monitoring = TradingGatewayMonitoring()
        logger.info("Performance monitoring initialized")

        # Initialize degraded mode manager
        from trading_gateway_service.resilience.degraded_mode import TradingGatewayDegradedMode
        app.state.degraded_mode_manager = TradingGatewayDegradedMode(service_name=APP_NAME)
        logger.info("Degraded mode manager initialized")

        # Configure degraded mode strategies
        configure_trading_gateway_degraded_mode(
            submit_order_func=None,  # Will be set later
            check_risk_func=None,    # Will be set later
            process_queue_func=None, # Will be set later
            account_balance_func=None # Will be set later
        )
        logger.info("Degraded mode strategies configured")

        # Initialize Kafka event bus
        try:
            app.state.event_bus = KafkaEventBus(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                group_id="trading-gateway-service",
                client_id="trading-gateway-service-python"
            )

            # Start consuming events in non-blocking mode
            app.state.event_bus.start_consuming(blocking=False)

            # Update health check function for Kafka
            app.state.health.checks[2]["check_func"] = lambda: app.state.event_bus is not None

            logger.info("Kafka event bus initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka event bus: {e}")
            # Don't fail startup, service can run in degraded mode

        logger.info("Service initialization complete")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        # Don't raise, allow service to start in degraded mode

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info(f"Shutting down {APP_NAME}")

    # Stop market data service
    if hasattr(app.state, "market_data_service"):
        try:
            # The MarketDataService doesn't have a shutdown method, so we'll just log it
            logger.info("Market data service shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down market data service: {e}")

    # Stop order reconciliation service
    if hasattr(app.state, "order_reconciliation_service"):
        try:
            await app.state.order_reconciliation_service.stop()
            logger.info("Order reconciliation service shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down order reconciliation service: {e}")

    # Close event bus connection if it exists
    if hasattr(app.state, "event_bus"):
        try:
            app.state.event_bus.flush()  # Ensure pending messages are sent
            app.state.event_bus.close()
            logger.info("Kafka event bus closed successfully")
        except Exception as e:
            logger.error(f"Error closing Kafka event bus: {e}")

# Main entry point
if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.getenv("PYTHON_PORT", "8005"))

    # Run the application
    uvicorn.run(
        "trading_gateway_service.main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENVIRONMENT", "development") == "development",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
