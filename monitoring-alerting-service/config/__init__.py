"""
Configuration package for Monitoring Alerting Service.

This package provides configuration management for the Monitoring Alerting Service.
It includes both legacy and standardized configuration systems for backward compatibility.
"""

# Legacy configuration system
from monitoring_alerting.config.config import (
from monitoring_alerting.config.standardized_config import config_manager, get_config
    config_manager as legacy_config_manager,
    get_service_config,
    get_database_config
)

# Standardized configuration system
from monitoring_alerting.config.standardized_config import (
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
