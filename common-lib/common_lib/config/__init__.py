"""
Configuration package for the forex trading platform.

This package provides a centralized configuration system for the platform.
It includes both the legacy configuration system and the new standardized configuration system.
"""

# Legacy configuration system
from common_lib.config.config_schema import (
    Config,
    AppConfig,
    DatabaseConfig,
    LoggingConfig,
    ServiceConfig,
    RetryConfig,
    CircuitBreakerConfig,
    ServiceClientConfig,
    ServiceClientsConfig,
    ServiceSpecificConfig
)
from common_lib.config.config_loader import ConfigLoader
from common_lib.config.config_manager import ConfigManager

# Standardized configuration system
from common_lib.config.standardized_config import (
    BaseAppSettings,
    ConfigManager as StandardizedConfigManager,
    ConfigSource,
    ConfigValue,
    get_config_manager,
    get_settings
)

__all__ = [
    # Legacy configuration system
    'Config',
    'AppConfig',
    'DatabaseConfig',
    'LoggingConfig',
    'ServiceConfig',
    'RetryConfig',
    'CircuitBreakerConfig',
    'ServiceClientConfig',
    'ServiceClientsConfig',
    'ServiceSpecificConfig',
    'ConfigLoader',
    'ConfigManager',

    # Standardized configuration system
    'BaseAppSettings',
    'StandardizedConfigManager',
    'ConfigSource',
    'ConfigValue',
    'get_config_manager',
    'get_settings'
]