"""
Configuration management for the service.
"""

import os
from common_lib.config import ConfigManager
from .config_schema import ServiceConfig

# Create a singleton ConfigManager instance
config_manager = ConfigManager(
    config_path=os.environ.get("CONFIG_PATH", "config/config.yaml"),
    service_specific_model=ServiceConfig,
    env_prefix=os.environ.get("CONFIG_ENV_PREFIX", "FEATURE_STORE_SERVICE_BACKUP_"),
    default_config_path=os.environ.get("DEFAULT_CONFIG_PATH", "config/default/config.yaml")
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
