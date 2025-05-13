"""
Configuration package for Data Pipeline Service.

This package provides configuration management for the Data Pipeline Service.
It includes both legacy and standardized configuration systems for backward compatibility.
"""

# Legacy configuration system
from data_pipeline.config.config import (
from config.standardized_config import config_manager, get_config
    config_manager as legacy_config_manager,
    get_service_config,
    get_database_config
)

# Standardized configuration system
from config.standardized_config import (
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
