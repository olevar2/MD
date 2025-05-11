"""
Configuration package for the Analysis Engine Service.

This package provides a centralized configuration management system
that consolidates settings from various sources.
"""

from analysis_engine.config.config import (
    get_service_config,
    get_database_config,
    get_logging_config,
    get_service_client_config,
    get_analysis_engine_config,
    is_development,
    is_testing,
    is_production
)

# For backward compatibility
from analysis_engine.config.settings import AnalysisEngineSettings, get_settings

__all__ = [
    "get_service_config",
    "get_database_config",
    "get_logging_config",
    "get_service_client_config",
    "get_analysis_engine_config",
    "is_development",
    "is_testing",
    "is_production",
    "AnalysisEngineSettings",
    "get_settings"
]
