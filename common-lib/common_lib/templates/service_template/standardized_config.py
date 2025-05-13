"""
Configuration module for the service.

This module provides configuration management for the service using the standardized
configuration management system from common-lib.
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator

from common_lib.config import BaseAppSettings, get_settings, get_config_manager


class ServiceSettings(BaseAppSettings):
    """
    Service-specific settings.
    
    This class extends the base application settings with service-specific configuration.
    """
    
    # Override service name
    SERVICE_NAME: str = Field("service-template", description="Name of the service")
    
    # Add service-specific configuration
    FEATURE_ENABLED: bool = Field(False, description="Enable feature")
    FEATURE_TIMEOUT: int = Field(30, description="Feature timeout in seconds")
    FEATURE_MAX_RETRIES: int = Field(3, description="Maximum number of retries")
    
    # External service configuration
    EXTERNAL_SERVICE_URL: str = Field(
        "http://external-service:8000",
        description="URL of the external service"
    )
    EXTERNAL_SERVICE_TIMEOUT: int = Field(
        30,
        description="Timeout for external service requests in seconds"
    )
    
    # Add validation
    @field_validator("FEATURE_TIMEOUT", "EXTERNAL_SERVICE_TIMEOUT")
    def validate_timeout(cls, v: int, info) -> int:
        """
        Validate timeout.
        
        Args:
            v: Timeout value
            info: Validation info
            
        Returns:
            Validated timeout value
            
        Raises:
            ValueError: If the timeout is negative
        """
        if v < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return v
    
    @field_validator("FEATURE_MAX_RETRIES")
    def validate_max_retries(cls, v: int) -> int:
        """
        Validate maximum number of retries.
        
        Args:
            v: Maximum number of retries
            
        Returns:
            Validated maximum number of retries
            
        Raises:
            ValueError: If the maximum number of retries is negative
        """
        if v < 0:
            raise ValueError("Maximum number of retries must be non-negative")
        return v


@lru_cache()
def get_service_settings() -> ServiceSettings:
    """
    Get cached service settings.
    
    Returns:
        Service settings
    """
    return get_settings(
        settings_class=ServiceSettings,
        env_file=os.environ.get("ENV_FILE", ".env"),
        config_file=os.environ.get("CONFIG_FILE", "config/config.yaml"),
        env_prefix=os.environ.get("ENV_PREFIX", "SERVICE_")
    )


@lru_cache()
def get_service_config_manager():
    """
    Get cached service configuration manager.
    
    Returns:
        Service configuration manager
    """
    return get_config_manager(
        settings_class=ServiceSettings,
        env_file=os.environ.get("ENV_FILE", ".env"),
        config_file=os.environ.get("CONFIG_FILE", "config/config.yaml"),
        env_prefix=os.environ.get("ENV_PREFIX", "SERVICE_")
    )


# Create a settings instance for easy access
settings = get_service_settings()
