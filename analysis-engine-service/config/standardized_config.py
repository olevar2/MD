"""
Standardized Configuration Management for Analysis Engine Service

This module provides standardized configuration management for the Analysis Engine Service.
It uses the AnalysisEngineSettings class from common-lib to define service-specific settings
and provides a ConfigurationManager class for accessing configuration values.
"""

import os
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

from common_lib.config.standardized_config import get_settings
from common_lib.config.service_settings import AnalysisEngineSettings

# Create logger
logger = logging.getLogger(__name__)


class ConfigurationManager:
    """Configuration manager for the application."""

    def __init__(self):
        """Initialize the configuration manager."""
        self._settings = get_settings(settings_class=AnalysisEngineSettings)
        self._config_cache: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: The configuration key
            default: The default value if key is not found

        Returns:
            The configuration value or default
        """
        if key in self._config_cache:
            return self._config_cache[key]

        value = getattr(self._settings, key, default)
        self._config_cache[key] = value
        return value

    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.

        Returns:
            Dictionary of all configuration values
        """
        return self._settings.model_dump()

    def reload(self) -> None:
        """Reload configuration from all sources."""
        self._settings = get_settings(settings_class=AnalysisEngineSettings)
        self._config_cache.clear()

    @property
    def settings(self) -> AnalysisEngineSettings:
        """Get the settings object."""
        return self._settings


# Create a singleton instance
config_manager = ConfigurationManager()


def get_config() -> ConfigurationManager:
    """
    Get the configuration manager.

    Returns:
        Configuration manager instance
    """
    return config_manager
