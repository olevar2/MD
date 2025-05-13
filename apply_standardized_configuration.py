"""
Script to apply standardized configuration management to all services.

This script applies the standardized configuration management to all services in the codebase.
It creates a standardized_config.py file in each service based on the service-specific settings
in common-lib/common_lib/config/service_settings.py.
"""

import os
import re
import shutil
import datetime
from typing import Dict, List, Set, Tuple, Optional, Any

# Service directories to process
SERVICE_DIRS = [
    "analysis-engine-service",
    "data-pipeline-service",
    "feature-store-service",
    "ml-integration-service",
    "ml-workbench-service",
    "monitoring-alerting-service",
    "portfolio-management-service",
    "strategy-execution-engine",
    "trading-gateway-service",
    "ui-service"
]

# Service name mapping
SERVICE_NAME_MAPPING = {
    "analysis-engine-service": "AnalysisEngineSettings",
    "data-pipeline-service": "DataPipelineSettings",
    "feature-store-service": "FeatureStoreSettings",
    "ml-integration-service": "MLIntegrationSettings",
    "ml-workbench-service": "MLWorkbenchSettings",
    "monitoring-alerting-service": "MonitoringAlertingSettings",
    "portfolio-management-service": "PortfolioManagementSettings",
    "strategy-execution-engine": "StrategyExecutionSettings",
    "trading-gateway-service": "TradingGatewaySettings",
    "ui-service": "UIServiceSettings"
}

# Template for standardized_config.py
STANDARDIZED_CONFIG_TEMPLATE = """\"\"\"
Standardized Configuration Management for {service_display_name}

This module provides standardized configuration management for the {service_display_name}.
It uses the {settings_class} class from common-lib to define service-specific settings
and provides a ConfigurationManager class for accessing configuration values.
\"\"\"

import os
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

from common_lib.config.standardized_config import get_settings
from common_lib.config.service_settings import {settings_class}

# Create logger
logger = logging.getLogger(__name__)


class ConfigurationManager:
    """
    ConfigurationManager class.
    
    Attributes:
        Add attributes here
    """

    \"\"\"Configuration manager for the application.\"\"\"

    def __init__(self):
    """
      init  .
    
    """

        \"\"\"Initialize the configuration manager.\"\"\"
        self._settings = get_settings(settings_class={settings_class})
        self._config_cache: Dict[str, Any] = {{}}

    def get(self, key: str, default: Any = None) -> Any:
    """
    Get.
    
    Args:
        key: Description of key
        default: Description of default
    
    Returns:
        Any: Description of return value
    
    """

        \"\"\"
        Get a configuration value.

        Args:
            key: The configuration key
            default: The default value if key is not found

        Returns:
            The configuration value or default
        \"\"\"
        if key in self._config_cache:
            return self._config_cache[key]

        value = getattr(self._settings, key, default)
        self._config_cache[key] = value
        return value

    def get_all(self) -> Dict[str, Any]:
    """
    Get all.
    
    Returns:
        Dict[str, Any]: Description of return value
    
    """

        \"\"\"
        Get all configuration values.

        Returns:
            Dictionary of all configuration values
        \"\"\"
        return self._settings.model_dump()

    def reload(self) -> None:
    """
    Reload.
    
    """

        \"\"\"Reload configuration from all sources.\"\"\"
        self._settings = get_settings(settings_class={settings_class})
        self._config_cache.clear()

    @property
    def settings(self) -> {settings_class}:
    """
    Settings.
    
    Returns:
        {settings_class}: Description of return value
    
    """

        \"\"\"Get the settings object.\"\"\"
        return self._settings


# Create a singleton instance
config_manager = ConfigurationManager()


def get_config() -> ConfigurationManager:
    \"\"\"
    Get the configuration manager.

    Returns:
        Configuration manager instance
    \"\"\"
    return config_manager
"""

# Template for config/__init__.py
CONFIG_INIT_TEMPLATE = """\"\"\"
Configuration package for {service_display_name}.

This package provides configuration management for the {service_display_name}.
It includes both legacy and standardized configuration systems for backward compatibility.
\"\"\"

# Legacy configuration system
from {service_module}.config.config import (
    config_manager as legacy_config_manager,
    get_service_config,
    get_database_config
)

# Standardized configuration system
from {service_module}.config.standardized_config import (
    config_manager,
    get_config
)

__all__ = [
    # Legacy configuration system
    'legacy_config_manager',
    'get_service_config',
    'get_database_config',
    
    # Standardized configuration system
    'config_manager',
    'get_config'
]
"""


def apply_standardized_configuration():
    """
    Apply standardized configuration management to all services.
    """
    # Process each service
    for service_dir in SERVICE_DIRS:
        service_name = service_dir.replace("-", "_").replace("_service", "")
        service_module = service_name
        
        # Get service-specific settings class
        settings_class = SERVICE_NAME_MAPPING.get(service_dir)
        if not settings_class:
            print(f"No settings class found for {service_dir}, skipping")
            continue
        
        # Create service display name
        service_display_name = service_dir.replace("-", " ").title()
        
        # Create standardized_config.py content
        standardized_config_content = STANDARDIZED_CONFIG_TEMPLATE.format(
            service_display_name=service_display_name,
            settings_class=settings_class,
            service_module=service_module
        )
        
        # Create config/__init__.py content
        config_init_content = CONFIG_INIT_TEMPLATE.format(
            service_display_name=service_display_name,
            service_module=service_module
        )
        
        # Create standardized_config.py
        config_dir = os.path.join(service_dir, "config")
        if not os.path.exists(config_dir):
            config_dir = os.path.join(service_dir, service_module, "config")
        
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        
        standardized_config_path = os.path.join(config_dir, "standardized_config.py")
        
        # Check if file already exists
        if os.path.exists(standardized_config_path):
            # Create backup
            backup_file = f"{standardized_config_path}.bak.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            shutil.copy2(standardized_config_path, backup_file)
            print(f"Created backup of existing standardized_config.py: {backup_file}")
        
        # Write standardized_config.py
        with open(standardized_config_path, "w", encoding="utf-8") as f:
            f.write(standardized_config_content)
        
        print(f"Created standardized_config.py for {service_dir}")
        
        # Create or update config/__init__.py
        config_init_path = os.path.join(config_dir, "__init__.py")
        
        # Check if file already exists
        if os.path.exists(config_init_path):
            # Create backup
            backup_file = f"{config_init_path}.bak.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            shutil.copy2(config_init_path, backup_file)
            print(f"Created backup of existing config/__init__.py: {backup_file}")
        
        # Write config/__init__.py
        with open(config_init_path, "w", encoding="utf-8") as f:
            f.write(config_init_content)
        
        print(f"Created or updated config/__init__.py for {service_dir}")
        
        # Create .env file if it doesn't exist
        env_file_path = os.path.join(service_dir, ".env")
        if not os.path.exists(env_file_path):
            with open(env_file_path, "w", encoding="utf-8") as f:
                f.write(f"# Environment variables for {service_display_name}\n")
                f.write(f"SERVICE_NAME={service_name}\n")
                f.write("LOG_LEVEL=INFO\n")
                f.write("DEBUG_MODE=False\n")
            
            print(f"Created .env file for {service_dir}")


if __name__ == "__main__":
    apply_standardized_configuration()
