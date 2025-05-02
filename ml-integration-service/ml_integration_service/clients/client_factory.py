"""
Client Factory for ML Integration Service

This module provides factory functions for creating service clients
used by the ML Integration Service.
"""

import logging
from typing import Dict, Any, Optional

from common_lib.clients import get_client, ClientConfig
from ml_integration_service.config.settings import settings

# Import client implementations
from ml_integration_service.clients.analysis_engine_client import AnalysisEngineClient
from ml_integration_service.clients.ml_workbench_client import MLWorkbenchClient

logger = logging.getLogger(__name__)


def get_analysis_engine_client(config_override: Optional[Dict[str, Any]] = None) -> AnalysisEngineClient:
    """
    Get a configured Analysis Engine client.

    Args:
        config_override: Optional configuration overrides

    Returns:
        Configured Analysis Engine client
    """
    return get_client(
        client_class=AnalysisEngineClient,
        service_name="analysis-engine-service",
        config_override=config_override
    )


def get_ml_workbench_client(config_override: Optional[Dict[str, Any]] = None) -> MLWorkbenchClient:
    """
    Get a configured ML Workbench client.

    Args:
        config_override: Optional configuration overrides

    Returns:
        Configured ML Workbench client
    """
    return get_client(
        client_class=MLWorkbenchClient,
        service_name="ml-workbench-service",
        config_override=config_override
    )

# Add factory functions for other clients as needed
# def get_feature_store_client(...): ...


def initialize_clients() -> None:
    """
    Initialize all clients with proper configuration.
    This function should be called during service startup.
    """
    logger.info("Initializing service clients...")

    # Import registration function
    from common_lib.clients import register_client_config

    # Configure Analysis Engine client
    analysis_engine_config = {
        "base_url": settings.ANALYSIS_ENGINE_API_URL,
        "service_name": "analysis-engine-service",
        "api_key": settings.ANALYSIS_ENGINE_API_KEY.get_secret_value() if settings.ANALYSIS_ENGINE_API_KEY else None,
        "timeout_seconds": 30.0,
        "retry_base_delay": 0.5,
        "max_retries": 3,
        "circuit_breaker_failure_threshold": 5,
        "circuit_breaker_reset_timeout_seconds": 60,
        "bulkhead_max_concurrent": 20,
    }

    # Configure ML Workbench client
    ml_workbench_config = {
        "base_url": settings.ML_WORKBENCH_API_URL,
        "service_name": "ml-workbench-service",
        "api_key": settings.ML_WORKBENCH_API_KEY.get_secret_value() if settings.ML_WORKBENCH_API_KEY else None,
        "timeout_seconds": 30.0,
        "retry_base_delay": 0.5,
        "max_retries": 3,
        "circuit_breaker_failure_threshold": 5,
        "circuit_breaker_reset_timeout_seconds": 60,
        "bulkhead_max_concurrent": 20,
    }

    # Register client configurations
    register_client_config("analysis-engine-service", ClientConfig(**analysis_engine_config))
    register_client_config("ml-workbench-service", ClientConfig(**ml_workbench_config))

    # Add configurations for other clients as needed

    logger.info("Service clients initialized successfully")
