"""
Configuration Management Module

This module provides configuration management functionality for the Analysis Engine Service.
This is a backward compatibility layer that imports from the consolidated settings module.

from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

DEPRECATED: This module is deprecated and will be removed after 2023-12-31.
Please import from analysis_engine.config instead.

Migration Guide:
1. Replace imports from analysis_engine.core.config with analysis_engine.config
2. If you're importing Settings, use AnalysisEngineSettings instead
3. All other imports (get_settings, ConfigurationManager, etc.) remain the same

Example:
# Before
from analysis_engine.core.config import Settings, get_settings

# After
from analysis_engine.config import AnalysisEngineSettings as Settings, get_settings
"""
import warnings
import inspect
import os
import datetime
from typing import Any, Dict, Optional, List
from analysis_engine.config.settings import AnalysisEngineSettings as Settings, get_settings, ConfigurationManager, get_db_url, get_redis_url, get_market_data_service_url, get_notification_service_url, get_analysis_settings, get_rate_limits, get_db_settings
from analysis_engine.core.deprecation_monitor import record_usage
REMOVAL_DATE = datetime.date(2023, 12, 31)
days_until_removal = (REMOVAL_DATE - datetime.date.today()).days
days_message = (f'{days_until_removal} days' if days_until_removal > 0 else
    'PAST DUE')


@with_exception_handling
def show_deprecation_warning():
    """
    Show deprecation warning.
    
    """

    frame = inspect.currentframe().f_back.f_back
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    function_name = frame.f_code.co_name
    try:
        rel_path = os.path.relpath(filename)
    except ValueError:
        rel_path = filename
    record_usage('analysis_engine.core.config')
    warnings.warn(
        f"""DEPRECATION WARNING: analysis_engine.core.config will be removed after {REMOVAL_DATE} ({days_message}).
Please import from analysis_engine.config instead.
Called from {rel_path}:{lineno} in function '{function_name}'
Migration guide: https://confluence.example.com/display/DEV/Configuration+Migration+Guide"""
        , DeprecationWarning, stacklevel=2)


show_deprecation_warning()
settings = get_settings()
