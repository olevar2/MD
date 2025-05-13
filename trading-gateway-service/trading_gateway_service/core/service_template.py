"""
Service Template Implementation for Trading Gateway Service

This module implements the service template pattern from common-lib for the Trading Gateway Service.
It provides standardized configuration, logging, service clients, database connectivity, and error handling.
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional, List, Type

from fastapi import FastAPI, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from common_lib.templates.service_template import (
    ServiceConfig,
    config_manager,
    get_service_config,
    get_database_config,
    get_logging_config,
    get_service_client_config,
    setup_logging,
    ServiceClients,
    Database,
    handle_error,
    handle_exception,
    handle_async_exception
)

from common_lib.monitoring.middleware import (
    StandardMetricsMiddleware,
    StandardLoggingMiddleware,
    StandardTracingMiddleware
)

from common_lib.security.middleware import (
    SecurityLoggingMiddleware,
    RateLimitingMiddleware,
    SecurityHeadersMiddleware
)

from common_lib.security.monitoring import (
    SecurityMonitor,
    SecurityEventCategory,
    SecurityEventSeverity
)

from trading_gateway_service.config.config import get_trading_gateway_config
from trading_gateway_service.config.standardized_config import config_manager, get_config
from trading_gateway_service.core.logging import get_logger

# Configure logger
logger = logging.getLogger(__name__)


class TradingGatewayServiceTemplate:
    """
    Service template implementation for the Trading Gateway Service.
    
    This class provides standardized configuration, logging, service clients,
    database connectivity, and error handling for the Trading Gateway Service.
    """
    
    def __init__(self, app: FastAPI):
        """
        Initialize the service template.
        
        Args:
            app: FastAPI application
        """
        self.app = app
        self.service_config = get_service_config()
        self.database_config = get_database_config()
        self.logging_config = get_logging_config()
        self.service_client_config = get_service_client_config()
        self.trading_gateway_config = get_trading_gateway_config()
        
        # Initialize logger
        self.logger = setup_logging("trading-gateway-service")
        
        # Initialize security monitor
        self.security_monitor = SecurityMonitor(
            service_name="trading-gateway-service",
            thresholds=[
                {
                    "name": "type:authentication_failure",
                    "description": "Authentication failure threshold",
                    "threshold": 5,
                    "time_window_seconds": 300,
                    "severity": SecurityEventSeverity.WARNING,
                    "actions": ["alert"]
                },
                {
                    "name": "type:authorization_failure",
                    "description": "Authorization failure threshold",
                    "threshold": 5,
                    "time_window_seconds": 300,
                    "severity": SecurityEventSeverity.WARNING,
                    "actions": ["alert"]
                },
                {
                    "name": "type:api_request:status:error",
                    "description": "API error threshold",
                    "threshold": 10,
                    "time_window_seconds": 300,
                    "severity": SecurityEventSeverity.WARNING,
                    "actions": ["alert"]
                },
                {
                    "name": "type:order_submission_failure",
                    "description": "Order submission failure threshold",
                    "threshold": 3,
                    "time_window_seconds": 300,
                    "severity": SecurityEventSeverity.ERROR,
                    "actions": ["alert"]
                }
            ]
        )
        
        # Initialize service clients
        self.service_clients = ServiceClients(self.service_client_config)
        
        # Initialize database
        self.database = Database(self.database_config)
    
    def configure_middleware(self):
        """Configure middleware for the application."""
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.service_config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add metrics middleware
        self.app.add_middleware(
            StandardMetricsMiddleware,
            service_name="trading-gateway-service",
            exclude_paths=["/health", "/metrics"]
        )
        
        # Add logging middleware
        self.app.add_middleware(
            StandardLoggingMiddleware,
            service_name="trading-gateway-service",
            exclude_paths=["/health", "/metrics"]
        )
        
        # Add tracing middleware
        self.app.add_middleware(
            StandardTracingMiddleware,
            service_name="trading-gateway-service",
            exclude_paths=["/health", "/metrics"]
        )
        
        # Add security logging middleware
        self.app.add_middleware(
            SecurityLoggingMiddleware,
            security_monitor=self.security_monitor,
            exclude_paths=["/health", "/metrics"]
        )
        
        # Add rate limiting middleware with trading-specific rate limits
        role_rates = {
            "admin": self.trading_gateway_config_manager.get('rate_max_requests_per_minute', 120),
            "trader": self.trading_gateway_config_manager.get('rate_max_requests_per_minute', 60),
            "readonly": self.trading_gateway_config_manager.get('rate_max_requests_per_minute', 120)
        }
        
        self.app.add_middleware(
            RateLimitingMiddleware,
            security_monitor=self.security_monitor,
            default_rate=self.service_config.max_requests_per_minute,
            role_rates=role_rates,
            exclude_paths=["/health", "/metrics"]
        )
        
        # Add security headers middleware
        self.app.add_middleware(
            SecurityHeadersMiddleware,
            exclude_paths=["/health", "/metrics"]
        )
    
    async def startup(self):
        """Startup event handler."""
        # Connect to database
        await self.database.connect()
        
        # Initialize service clients
        await self.service_clients.initialize()
        
        self.logger.info("Service template initialized successfully")
    
    async def shutdown(self):
        """Shutdown event handler."""
        # Close database connection
        await self.database.close()
        
        # Close service clients
        await self.service_clients.close()
        
        self.logger.info("Service template shut down successfully")
    
    def register_exception_handlers(self):
        """Register exception handlers."""
        # Import trading gateway exceptions
        from trading_gateway_service.error.exceptions import TradingGatewayException
        
        @self.app.exception_handler(TradingGatewayException)
        async def trading_gateway_exception_handler(request: Request, exc: TradingGatewayException):
            """Handle TradingGatewayException."""
            return handle_error(exc, operation=request.url.path)
        
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            """Handle general exceptions."""
            return handle_error(exc, operation=request.url.path)
    
    def initialize(self):
        """Initialize the service template."""
        # Configure middleware
        self.configure_middleware()
        
        # Register exception handlers
        self.register_exception_handlers()
        
        # Register startup and shutdown events
        @self.app.on_event("startup")
        async def startup_event():
    """
    Startup event.
    
    """

            await self.startup()
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
    """
    Shutdown event.
    
    """

            await self.shutdown()
        
        self.logger.info("Service template configured successfully")


# Create a singleton instance
service_template = None


def get_service_template(app: Optional[FastAPI] = None) -> TradingGatewayServiceTemplate:
    """
    Get the service template instance.
    
    Args:
        app: FastAPI application (required on first call)
        
    Returns:
        Service template instance
        
    Raises:
        ValueError: If app is not provided on first call
    """
    global service_template
    
    if service_template is None:
        if app is None:
            raise ValueError("FastAPI app must be provided on first call")
        
        service_template = TradingGatewayServiceTemplate(app)
    
    return service_template
