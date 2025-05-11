"""
Configuration Module

This module provides configuration management for the service.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Type

from pydantic import BaseModel, Field, validator

from common_lib.config import (
    ServiceSpecificConfig,
    ConfigManager
)


# Define service-specific configuration
class ServiceConfig(ServiceSpecificConfig):
    """
    Service-specific configuration.
    
    This class defines the service-specific configuration parameters.
    """
    
    # Add service-specific configuration parameters here
    max_workers: int = Field(4, description="Maximum number of worker threads")
    cache_size: int = Field(1000, description="Maximum number of items in the cache")
    
    @validator("max_workers")
    def validate_max_workers(cls, v):
        """Validate maximum number of workers."""
        if v < 1:
            raise ValueError("Maximum workers must be at least 1")
        return v
    
    @validator("cache_size")
    def validate_cache_size(cls, v):
        """Validate cache size."""
        if v < 0:
            raise ValueError("Cache size must be non-negative")
        return v


# Create a singleton ConfigManager instance
config_manager = ConfigManager(
    config_path=os.environ.get("CONFIG_PATH", "config/config.yaml"),
    service_specific_model=ServiceConfig,
    env_prefix=os.environ.get("CONFIG_ENV_PREFIX", "APP_"),
    default_config_path=os.environ.get("DEFAULT_CONFIG_PATH", "config/default_config.yaml")
)


# Helper functions to access configuration
def get_service_config() -> ServiceConfig:
    """
    Get the service-specific configuration.
    
    Returns:
        Service-specific configuration
    """
    return config_manager.get_service_specific_config()


def get_database_config():
    """
    Get the database configuration.
    
    Returns:
        Database configuration
    """
    return config_manager.get_database_config()


def get_logging_config():
    """
    Get the logging configuration.
    
    Returns:
        Logging configuration
    """
    return config_manager.get_logging_config()


def get_service_client_config(service_name: str):
    """
    Get the configuration for a specific service client.
    
    Args:
        service_name: Name of the service
        
    Returns:
        Service client configuration
    """
    return config_manager.get_service_client_config(service_name)


def is_development() -> bool:
    """
    Check if the application is running in development mode.
    
    Returns:
        True if the application is running in development mode, False otherwise
    """
    return config_manager.is_development()


def is_testing() -> bool:
    """
    Check if the application is running in testing mode.
    
    Returns:
        True if the application is running in testing mode, False otherwise
    """
    return config_manager.is_testing()


def is_production() -> bool:
    """
    Check if the application is running in production mode.
    
    Returns:
        True if the application is running in production mode, False otherwise
    """
    return config_manager.is_production()
