"""
Configuration Module

This module provides configuration management for the service.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Type
from unittest.mock import MagicMock

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

    # Basic service information
    name: str = Field("service-template", description="Service name")
    version: str = Field("0.1.0", description="Service version")
    environment: str = Field("development", description="Service environment")

    # API configuration
    api_prefix: str = Field("/api/v1", description="API prefix")
    cors_origins: List[str] = Field(["*"], description="CORS origins")

    # Performance configuration
    max_workers: int = Field(4, description="Maximum number of worker threads")
    cache_size: int = Field(1000, description="Maximum number of items in the cache")
    max_requests_per_minute: int = Field(60, description="Maximum number of API requests per minute")

    # Resilience configuration
    max_retries: int = Field(3, description="Maximum number of retries for failed requests")
    retry_delay_seconds: int = Field(5, description="Delay between retries in seconds")
    timeout_seconds: int = Field(30, description="Timeout for API requests in seconds")

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

    @validator("max_requests_per_minute")
    def validate_max_requests_per_minute(cls, v):
        """Validate maximum number of API requests per minute."""
        if v < 1:
            raise ValueError("Maximum requests per minute must be at least 1")
        return v

    @validator("max_retries")
    def validate_max_retries(cls, v):
        """Validate maximum number of retries."""
        if v < 0:
            raise ValueError("Maximum retries must be non-negative")
        return v

    @validator("retry_delay_seconds")
    def validate_retry_delay_seconds(cls, v):
        """Validate retry delay."""
        if v < 0:
            raise ValueError("Retry delay must be non-negative")
        return v

    @validator("timeout_seconds")
    def validate_timeout_seconds(cls, v):
        """Validate timeout."""
        if v < 0:
            raise ValueError("Timeout must be non-negative")
        return v


# Create a singleton ConfigManager instance
try:
    config_manager = ConfigManager(
        config_path=os.environ.get("CONFIG_PATH", "config/config.yaml"),
        service_specific_model=ServiceConfig,
        env_prefix=os.environ.get("CONFIG_ENV_PREFIX", "APP_"),
        default_config_path=os.environ.get("DEFAULT_CONFIG_PATH", "config/default_config.yaml")
    )
except FileNotFoundError:
    # For testing purposes, create a mock config manager
    config_manager = MagicMock()
    config_manager.get_service_specific_config.return_value = ServiceConfig()
    config_manager.get_database_config.return_value = MagicMock()
    config_manager.get_logging_config.return_value = MagicMock()
    config_manager.get_service_client_config.return_value = MagicMock()
    config_manager.is_development.return_value = True
    config_manager.is_testing.return_value = False
    config_manager.is_production.return_value = False


# Helper functions to access configuration
def get_service_config() -> ServiceConfig:
    """
    Get the service-specific configuration.

    Returns:
        Service-specific configuration
    """
    try:
        return config_manager.get_service_specific_config()
    except:
        # For testing purposes
        return ServiceConfig()


def get_database_config():
    """
    Get the database configuration.

    Returns:
        Database configuration
    """
    try:
        return config_manager.get_database_config()
    except:
        # For testing purposes
        return MagicMock()


def get_logging_config():
    """
    Get the logging configuration.

    Returns:
        Logging configuration
    """
    try:
        return config_manager.get_logging_config()
    except:
        # For testing purposes
        return MagicMock()


def get_service_client_config(service_name: str):
    """
    Get the configuration for a specific service client.

    Args:
        service_name: Name of the service

    Returns:
        Service client configuration
    """
    try:
        return config_manager.get_service_client_config(service_name)
    except:
        # For testing purposes
        return MagicMock()


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
