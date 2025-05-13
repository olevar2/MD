"""
Client Factory for Strategy Execution Engine

This module provides factory functions for creating and managing client instances
for various services used by the Strategy Execution Engine.
"""

import logging
from typing import Dict, Any, Optional

from common_lib.clients import get_client, register_client_config, ClientConfig

from config.config_1 import get_settings
from adapters.trading_gateway_client import TradingGatewayClient

logger = logging.getLogger(__name__)

# Client instances
_trading_gateway_client: Optional[TradingGatewayClient] = None


def initialize_clients() -> None:
    """
    Initialize all service clients.

    This function registers client configurations and creates client instances
    for all services used by the Strategy Execution Engine.
    """
    settings = get_settings()

    # Register client configurations
    register_client_config(
        "trading-gateway-service",
        ClientConfig(
            base_url=settings.trading_gateway_url,
            service_name="trading-gateway-service",
            api_key=settings.trading_gateway_key,
            timeout_seconds=settings.client_timeout_seconds,
            max_retries=settings.client_max_retries,
            retry_base_delay=settings.client_retry_base_delay,
            retry_backoff_factor=settings.client_retry_backoff_factor,
            circuit_breaker_failure_threshold=settings.client_circuit_breaker_threshold,
            circuit_breaker_reset_timeout_seconds=settings.client_circuit_breaker_reset_timeout
        )
    )

    # Add more client configurations as needed

    logger.info("Service clients initialized successfully")


def get_trading_gateway_client() -> TradingGatewayClient:
    """
    Get the Trading Gateway client.

    Returns:
        TradingGatewayClient: Trading Gateway client instance

    Raises:
        RuntimeError: If clients have not been initialized
    """
    global _trading_gateway_client

    if _trading_gateway_client is None:
        # Initialize clients if not already initialized
        initialize_clients()

        # Create client instance
        _trading_gateway_client = get_client(
            client_class=TradingGatewayClient,
            service_name="trading-gateway-service"
        )

    return _trading_gateway_client


def get_trading_gateway_client_with_config(config: Dict[str, Any]) -> TradingGatewayClient:
    """
    Get a Trading Gateway client with custom configuration.

    Args:
        config: Custom client configuration

    Returns:
        TradingGatewayClient: Trading Gateway client instance
    """
    # Create client instance with custom configuration
    return get_client(
        client_class=TradingGatewayClient,
        service_name="trading-gateway-service",
        config_override=config
    )


def reset_clients() -> None:
    """
    Reset all client instances.

    This function clears all client instances, forcing them to be recreated
    the next time they are requested.
    """
    global _trading_gateway_client

    _trading_gateway_client = None

    logger.info("Service clients reset successfully")
