"""
Configuration package for Analysis Engine Service.

This package provides configuration management for the Analysis Engine Service.
It includes both legacy and standardized configuration systems for backward compatibility.
"""

# Legacy configuration system
from analysis_engine.config.config import (
from analysis_engine.config.standardized_config import config_manager, get_config
    config_manager as legacy_config_manager,
    get_service_config,
    get_database_config
)

# Standardized configuration system
from analysis_engine.config.standardized_config import (
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
