"""
Client Factory Module

This module provides a factory for creating service clients with proper configuration.
It centralizes client configuration and ensures consistent client creation across the platform.
"""

import logging
import os
from typing import Dict, Any, Optional, Type, TypeVar, cast

from common_lib.clients.base_client import BaseServiceClient, ClientConfig

logger = logging.getLogger(__name__)

# Type variable for client classes
T = TypeVar('T', bound=BaseServiceClient)

# Global client registry
_client_registry: Dict[str, BaseServiceClient] = {}

# Default client configurations
_default_configs: Dict[str, ClientConfig] = {}


def register_client_config(service_name: str, config: ClientConfig) -> None:
    """
    Register a default configuration for a service client.
    
    Args:
        service_name: Name of the service
        config: Client configuration
    """
    _default_configs[service_name] = config
    logger.debug(f"Registered default configuration for {service_name}")


def get_client_config(service_name: str) -> ClientConfig:
    """
    Get the configuration for a service client.
    
    Args:
        service_name: Name of the service
        
    Returns:
        Client configuration
        
    Raises:
        ValueError: If no configuration is found for the service
    """
    if service_name in _default_configs:
        return _default_configs[service_name]
    
    # Try to create a default configuration from environment variables
    base_url_env = f"{service_name.upper().replace('-', '_')}_API_URL"
    api_key_env = f"{service_name.upper().replace('-', '_')}_API_KEY"
    
    base_url = os.environ.get(base_url_env)
    if not base_url:
        # Try standard URL format
        base_url = f"http://{service_name}:8000/api/v1"
        logger.warning(
            f"No configuration found for {service_name} and no {base_url_env} "
            f"environment variable. Using default URL: {base_url}"
        )
    
    config = ClientConfig(
        base_url=base_url,
        service_name=service_name,
        api_key=os.environ.get(api_key_env)
    )
    
    # Register for future use
    _default_configs[service_name] = config
    
    return config


def create_client(
    client_class: Type[T],
    service_name: str,
    config_override: Optional[Dict[str, Any]] = None
) -> T:
    """
    Create a service client with proper configuration.
    
    Args:
        client_class: Client class to instantiate
        service_name: Name of the service
        config_override: Optional configuration overrides
        
    Returns:
        Configured client instance
    """
    # Get base configuration
    config = get_client_config(service_name)
    
    # Apply overrides
    if config_override:
        # Create a copy of the config
        config_dict = config.model_dump()
        # Update with overrides
        config_dict.update(config_override)
        # Create new config
        config = ClientConfig(**config_dict)
    
    # Create client
    client = client_class(config)
    
    logger.info(f"Created {client_class.__name__} for {service_name}")
    
    return client


def get_client(
    client_class: Type[T],
    service_name: str,
    config_override: Optional[Dict[str, Any]] = None,
    singleton: bool = True
) -> T:
    """
    Get a service client, creating it if necessary.
    
    Args:
        client_class: Client class to instantiate
        service_name: Name of the service
        config_override: Optional configuration overrides
        singleton: Whether to use a singleton instance
        
    Returns:
        Configured client instance
    """
    # For non-singleton clients, always create a new instance
    if not singleton:
        return create_client(client_class, service_name, config_override)
    
    # For singleton clients, check the registry
    client_key = f"{service_name}:{client_class.__name__}"
    
    if client_key in _client_registry:
        return cast(T, _client_registry[client_key])
    
    # Create and register the client
    client = create_client(client_class, service_name, config_override)
    _client_registry[client_key] = client
    
    return client
