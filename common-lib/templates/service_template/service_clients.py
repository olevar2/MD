"""
Service Clients Module

This module provides service clients for communicating with other services.
"""

import logging
from typing import Dict, Any, Optional

from common_lib.service_client import (
    ResilientServiceClientConfig,
    ResilientServiceClient
)
from .config import get_service_client_config


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
        self._clients = {}
    
    def get_client(self, service_name: str) -> ResilientServiceClient:
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
            service_client_config = get_service_client_config(service_name)
            if service_client_config is None:
                raise ValueError(f"Service client configuration not found for {service_name}")
            
            # Create resilient service client config
            config = ResilientServiceClientConfig(
                service_name=service_name,
                base_url=service_client_config.base_url,
                timeout=service_client_config.timeout,
                retry_count=service_client_config.retry.max_retries if service_client_config.retry else 3,
                retry_backoff=service_client_config.retry.initial_backoff if service_client_config.retry else 1.0,
                max_concurrent_requests=10,
                circuit_breaker_threshold=service_client_config.circuit_breaker.failure_threshold if service_client_config.circuit_breaker else 5,
                circuit_breaker_recovery_time=service_client_config.circuit_breaker.recovery_timeout if service_client_config.circuit_breaker else 30.0,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
            )
            
            # Create service client
            self._clients[service_name] = ResilientServiceClient(config, logger=self.logger)
        
        return self._clients[service_name]
    
    async def connect_all(self):
        """Connect all service clients."""
        for client in self._clients.values():
            await client.connect()
    
    async def close_all(self):
        """Close all service clients."""
        for client in self._clients.values():
            await client.close()
    
    # Convenience methods for specific services
    
    def get_market_data_client(self) -> ResilientServiceClient:
        """
        Get the Market Data Service client.
        
        Returns:
            Market Data Service client
        """
        return self.get_client("market_data_service")
    
    def get_feature_store_client(self) -> ResilientServiceClient:
        """
        Get the Feature Store Service client.
        
        Returns:
            Feature Store Service client
        """
        return self.get_client("feature_store_service")
    
    def get_analysis_engine_client(self) -> ResilientServiceClient:
        """
        Get the Analysis Engine Service client.
        
        Returns:
            Analysis Engine Service client
        """
        return self.get_client("analysis_engine_service")
    
    def get_trading_client(self) -> ResilientServiceClient:
        """
        Get the Trading Service client.
        
        Returns:
            Trading Service client
        """
        return self.get_client("trading_service")


# Create a singleton instance
service_clients = ServiceClients()
