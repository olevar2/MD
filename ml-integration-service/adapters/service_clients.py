"""
Standardized Service Clients Module for ML Integration Service

This module provides service clients for communicating with other services
using the standardized service client system from common-lib.
"""

import logging
from typing import Dict, Any, Optional, List

from common_lib.service_client import (
    ServiceClient,
    ServiceClientConfig,
    RetryConfig,
    CircuitBreakerConfig
)
from common_lib.monitoring.tracing import trace_method

from config.standardized_config_1 import settings
from core.logging_setup_standardized import get_service_logger


class ServiceClients:
    """
    Service clients for communicating with other services.
    
    This class provides access to service clients for communicating with other services.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the service clients.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or get_service_logger(__name__)
        self._clients: Dict[str, ServiceClient] = {}
        
        # Initialize service client configurations
        self._client_configs = {
            "ml_workbench_service": ServiceClientConfig(
                service_name="ml_workbench_service",
                base_url=settings.ML_WORKBENCH_API_URL,
                timeout=30,
                retry=RetryConfig(
                    max_retries=3,
                    initial_backoff=1.0,
                    max_backoff=30.0,
                    backoff_factor=2.0
                ),
                circuit_breaker=CircuitBreakerConfig(
                    failure_threshold=5,
                    recovery_timeout=30.0,
                    half_open_timeout=5.0
                ),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-API-Key": settings.ML_WORKBENCH_API_KEY.get_secret_value() if settings.ML_WORKBENCH_API_KEY else ""
                }
            ),
            "analysis_engine_service": ServiceClientConfig(
                service_name="analysis_engine_service",
                base_url=settings.ANALYSIS_ENGINE_API_URL,
                timeout=30,
                retry=RetryConfig(
                    max_retries=3,
                    initial_backoff=1.0,
                    max_backoff=30.0,
                    backoff_factor=2.0
                ),
                circuit_breaker=CircuitBreakerConfig(
                    failure_threshold=5,
                    recovery_timeout=30.0,
                    half_open_timeout=5.0
                ),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-API-Key": settings.ANALYSIS_ENGINE_API_KEY.get_secret_value() if settings.ANALYSIS_ENGINE_API_KEY else ""
                }
            ),
            "strategy_execution_service": ServiceClientConfig(
                service_name="strategy_execution_service",
                base_url=settings.STRATEGY_EXECUTION_API_URL,
                timeout=30,
                retry=RetryConfig(
                    max_retries=3,
                    initial_backoff=1.0,
                    max_backoff=30.0,
                    backoff_factor=2.0
                ),
                circuit_breaker=CircuitBreakerConfig(
                    failure_threshold=5,
                    recovery_timeout=30.0,
                    half_open_timeout=5.0
                ),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-API-Key": settings.STRATEGY_EXECUTION_API_KEY.get_secret_value() if settings.STRATEGY_EXECUTION_API_KEY else ""
                }
            )
        }
    
    @trace_method(name="get_client")
    def get_client(self, service_name: str) -> ServiceClient:
        """
        Get a service client for a specific service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Service client
            
        Raises:
            ValueError: If the service client configuration is not found
        """
        if service_name not in self._clients:
            # Get service client configuration
            config = self._client_configs.get(service_name)
            if config is None:
                raise ValueError(f"Service client configuration not found for {service_name}")
            
            # Create service client
            self._clients[service_name] = ServiceClient(config, logger=self.logger)
        
        return self._clients[service_name]
    
    @trace_method(name="connect_all")
    async def connect_all(self):
        """Connect all service clients."""
        for client in self._clients.values():
            await client.connect()
    
    @trace_method(name="close_all")
    async def close_all(self):
        """Close all service clients."""
        for client in self._clients.values():
            await client.close()
    
    # Convenience methods for specific services
    
    @trace_method(name="get_ml_workbench_client")
    def get_ml_workbench_client(self) -> ServiceClient:
        """
        Get the ML Workbench Service client.
        
        Returns:
            ML Workbench Service client
        """
        return self.get_client("ml_workbench_service")
    
    @trace_method(name="get_analysis_engine_client")
    def get_analysis_engine_client(self) -> ServiceClient:
        """
        Get the Analysis Engine Service client.
        
        Returns:
            Analysis Engine Service client
        """
        return self.get_client("analysis_engine_service")
    
    @trace_method(name="get_strategy_execution_client")
    def get_strategy_execution_client(self) -> ServiceClient:
        """
        Get the Strategy Execution Service client.
        
        Returns:
            Strategy Execution Service client
        """
        return self.get_client("strategy_execution_service")


# Create a singleton instance
service_clients = ServiceClients()
