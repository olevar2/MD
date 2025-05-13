"""
Standardized Service Clients Module for Data Pipeline Service

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
        self.logger = logger or logging.getLogger(__name__)
        self._clients: Dict[str, ServiceClient] = {}
        
        # Initialize service client configurations
        self._client_configs = {
            "market_data_service": ServiceClientConfig(
                service_name="market_data_service",
                base_url=settings.get("MARKET_DATA_SERVICE_URL", "http://market-data-service:8000"),
                timeout=settings.get("MARKET_DATA_SERVICE_TIMEOUT", 30),
                retry=RetryConfig(
                    max_retries=settings.get("MARKET_DATA_SERVICE_MAX_RETRIES", 3),
                    initial_backoff=settings.get("MARKET_DATA_SERVICE_INITIAL_BACKOFF", 1.0),
                    max_backoff=settings.get("MARKET_DATA_SERVICE_MAX_BACKOFF", 30.0),
                    backoff_factor=settings.get("MARKET_DATA_SERVICE_BACKOFF_FACTOR", 2.0)
                ),
                circuit_breaker=CircuitBreakerConfig(
                    failure_threshold=settings.get("MARKET_DATA_SERVICE_FAILURE_THRESHOLD", 5),
                    recovery_timeout=settings.get("MARKET_DATA_SERVICE_RECOVERY_TIMEOUT", 30.0),
                    half_open_timeout=settings.get("MARKET_DATA_SERVICE_HALF_OPEN_TIMEOUT", 5.0)
                ),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-API-Key": settings.get("MARKET_DATA_SERVICE_API_KEY", "")
                }
            ),
            "feature_store_service": ServiceClientConfig(
                service_name="feature_store_service",
                base_url=settings.get("FEATURE_STORE_URL", "http://feature-store-service:8000"),
                timeout=settings.get("FEATURE_STORE_TIMEOUT", 30),
                retry=RetryConfig(
                    max_retries=settings.get("FEATURE_STORE_MAX_RETRIES", 3),
                    initial_backoff=settings.get("FEATURE_STORE_INITIAL_BACKOFF", 1.0),
                    max_backoff=settings.get("FEATURE_STORE_MAX_BACKOFF", 30.0),
                    backoff_factor=settings.get("FEATURE_STORE_BACKOFF_FACTOR", 2.0)
                ),
                circuit_breaker=CircuitBreakerConfig(
                    failure_threshold=settings.get("FEATURE_STORE_FAILURE_THRESHOLD", 5),
                    recovery_timeout=settings.get("FEATURE_STORE_RECOVERY_TIMEOUT", 30.0),
                    half_open_timeout=settings.get("FEATURE_STORE_HALF_OPEN_TIMEOUT", 5.0)
                ),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-API-Key": settings.get("FEATURE_STORE_API_KEY", "")
                }
            ),
            "analysis_engine_service": ServiceClientConfig(
                service_name="analysis_engine_service",
                base_url=settings.get("ANALYSIS_ENGINE_SERVICE_URL", "http://analysis-engine-service:8000"),
                timeout=settings.get("ANALYSIS_ENGINE_SERVICE_TIMEOUT", 30),
                retry=RetryConfig(
                    max_retries=settings.get("ANALYSIS_ENGINE_SERVICE_MAX_RETRIES", 3),
                    initial_backoff=settings.get("ANALYSIS_ENGINE_SERVICE_INITIAL_BACKOFF", 1.0),
                    max_backoff=settings.get("ANALYSIS_ENGINE_SERVICE_MAX_BACKOFF", 30.0),
                    backoff_factor=settings.get("ANALYSIS_ENGINE_SERVICE_BACKOFF_FACTOR", 2.0)
                ),
                circuit_breaker=CircuitBreakerConfig(
                    failure_threshold=settings.get("ANALYSIS_ENGINE_SERVICE_FAILURE_THRESHOLD", 5),
                    recovery_timeout=settings.get("ANALYSIS_ENGINE_SERVICE_RECOVERY_TIMEOUT", 30.0),
                    half_open_timeout=settings.get("ANALYSIS_ENGINE_SERVICE_HALF_OPEN_TIMEOUT", 5.0)
                ),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-API-Key": settings.get("ANALYSIS_ENGINE_SERVICE_API_KEY", "")
                }
            ),
            "trading_service": ServiceClientConfig(
                service_name="trading_service",
                base_url=settings.get("TRADING_SERVICE_URL", "http://trading-service:8000"),
                timeout=settings.get("TRADING_SERVICE_TIMEOUT", 30),
                retry=RetryConfig(
                    max_retries=settings.get("TRADING_SERVICE_MAX_RETRIES", 3),
                    initial_backoff=settings.get("TRADING_SERVICE_INITIAL_BACKOFF", 1.0),
                    max_backoff=settings.get("TRADING_SERVICE_MAX_BACKOFF", 30.0),
                    backoff_factor=settings.get("TRADING_SERVICE_BACKOFF_FACTOR", 2.0)
                ),
                circuit_breaker=CircuitBreakerConfig(
                    failure_threshold=settings.get("TRADING_SERVICE_FAILURE_THRESHOLD", 5),
                    recovery_timeout=settings.get("TRADING_SERVICE_RECOVERY_TIMEOUT", 30.0),
                    half_open_timeout=settings.get("TRADING_SERVICE_HALF_OPEN_TIMEOUT", 5.0)
                ),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-API-Key": settings.get("TRADING_SERVICE_API_KEY", "")
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
    
    @trace_method(name="get_market_data_client")
    def get_market_data_client(self) -> ServiceClient:
        """
        Get the Market Data Service client.
        
        Returns:
            Market Data Service client
        """
        return self.get_client("market_data_service")
    
    @trace_method(name="get_feature_store_client")
    def get_feature_store_client(self) -> ServiceClient:
        """
        Get the Feature Store Service client.
        
        Returns:
            Feature Store Service client
        """
        return self.get_client("feature_store_service")
    
    @trace_method(name="get_analysis_engine_client")
    def get_analysis_engine_client(self) -> ServiceClient:
        """
        Get the Analysis Engine Service client.
        
        Returns:
            Analysis Engine Service client
        """
        return self.get_client("analysis_engine_service")
    
    @trace_method(name="get_trading_client")
    def get_trading_client(self) -> ServiceClient:
        """
        Get the Trading Service client.
        
        Returns:
            Trading Service client
        """
        return self.get_client("trading_service")


# Create a singleton instance
service_clients = ServiceClients()
