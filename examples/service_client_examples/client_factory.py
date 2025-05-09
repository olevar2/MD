"""
Client Factory Example

This module demonstrates how to implement a client factory for standardized service clients.
"""

import logging
from typing import Dict, Any, Optional

from common_lib.clients.base_client import ClientConfig
from common_lib.clients.client_factory import register_client_config, get_client

from examples.service_client_examples.market_data_client import MarketDataClient

logger = logging.getLogger(__name__)


def initialize_clients():
    """
    Initialize service clients with proper configuration.
    
    This function should be called during service startup to register
    client configurations and initialize clients.
    """
    logger.info("Initializing service clients...")
    
    # Configure Market Data client
    market_data_config = {
        "base_url": "http://market-data-service:8000/api/v1",
        "service_name": "market-data-service",
        "timeout_seconds": 30.0,
        "retry_base_delay": 0.5,
        "max_retries": 3,
        "circuit_breaker_failure_threshold": 5,
        "circuit_breaker_reset_timeout_seconds": 60,
        "bulkhead_max_concurrent": 20,
    }
    
    # Register client configurations
    register_client_config("market-data-service", ClientConfig(**market_data_config))
    
    # Add configurations for other clients as needed
    
    logger.info("Service clients initialized successfully")


def get_market_data_client(config_override: Optional[Dict[str, Any]] = None) -> MarketDataClient:
    """
    Get a configured Market Data client.
    
    Args:
        config_override: Optional configuration overrides
        
    Returns:
        Configured Market Data client
    """
    return get_client(
        client_class=MarketDataClient,
        service_name="market-data-service",
        config_override=config_override
    )


# Add factory functions for other clients as needed