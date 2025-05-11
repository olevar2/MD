"""
Configuration package for the forex trading platform.

This package provides a centralized configuration system for the platform.
"""

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

__all__ = [
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
    'ConfigManager'
]