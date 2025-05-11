"""
Service Client Factory.

This module provides a factory for creating service clients using the adapter pattern.
This helps to break circular dependencies between services.
"""

import logging
from typing import Dict, Any, Optional

from common_lib.adapters import AdapterFactory
from common_lib.service_client.base_client import ServiceClientConfig


class ServiceClientFactory:
    """
    Factory for creating service clients using the adapter pattern.
    
    This factory uses the adapter pattern to create service clients,
    breaking circular dependencies between services.
    """
    
    def __init__(
        self,
        config_provider: Optional[Dict[str, Dict[str, Any]]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the service client factory.
        
        Args:
            config_provider: Dictionary mapping service names to service configurations
            logger: Logger to use (if None, creates a new logger)
        """
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config_provider = config_provider or {}
        
        # Create adapter factory
        self.adapter_factory = AdapterFactory(
            config_provider=self._create_service_client_configs(),
            logger=self.logger
        )
    
    def _create_service_client_configs(self) -> Dict[str, ServiceClientConfig]:
        """
        Create service client configurations from the config provider.
        
        Returns:
            Dictionary mapping service names to service client configurations
        """
        service_client_configs = {}
        
        for service_name, config in self.config_provider.items():
            service_client_configs[service_name] = ServiceClientConfig(
                service_name=service_name,
                base_url=config.get("base_url", f"http://{service_name}:8000"),
                retry_config=config.get("retry_config"),
                circuit_breaker_config=config.get("circuit_breaker_config"),
                timeout_config=config.get("timeout_config"),
                headers=config.get("headers", {})
            )
        
        return service_client_configs
    
    # Market Data Service Clients
    
    def create_market_data_provider(self):
        """
        Create a Market Data Provider client.
        
        Returns:
            Market Data Provider client
        """
        return self.adapter_factory.create_market_data_provider()
    
    def create_market_data_cache(self):
        """
        Create a Market Data Cache client.
        
        Returns:
            Market Data Cache client
        """
        return self.adapter_factory.create_market_data_cache()
    
    # Feature Store Service Clients
    
    def create_feature_provider(self):
        """
        Create a Feature Provider client.
        
        Returns:
            Feature Provider client
        """
        return self.adapter_factory.create_feature_provider()
    
    def create_feature_store(self):
        """
        Create a Feature Store client.
        
        Returns:
            Feature Store client
        """
        return self.adapter_factory.create_feature_store()
    
    def create_feature_generator(self):
        """
        Create a Feature Generator client.
        
        Returns:
            Feature Generator client
        """
        return self.adapter_factory.create_feature_generator()
    
    # Analysis Engine Service Clients
    
    def create_analysis_provider(self):
        """
        Create an Analysis Provider client.
        
        Returns:
            Analysis Provider client
        """
        return self.adapter_factory.create_analysis_provider()
    
    def create_indicator_provider(self):
        """
        Create an Indicator Provider client.
        
        Returns:
            Indicator Provider client
        """
        return self.adapter_factory.create_indicator_provider()
    
    def create_pattern_recognizer(self):
        """
        Create a Pattern Recognizer client.
        
        Returns:
            Pattern Recognizer client
        """
        return self.adapter_factory.create_pattern_recognizer()
    
    # Trading Service Clients
    
    def create_trading_provider(self):
        """
        Create a Trading Provider client.
        
        Returns:
            Trading Provider client
        """
        return self.adapter_factory.create_trading_provider()
    
    def create_order_book_provider(self):
        """
        Create an Order Book Provider client.
        
        Returns:
            Order Book Provider client
        """
        return self.adapter_factory.create_order_book_provider()
    
    def create_risk_manager(self):
        """
        Create a Risk Manager client.
        
        Returns:
            Risk Manager client
        """
        return self.adapter_factory.create_risk_manager()