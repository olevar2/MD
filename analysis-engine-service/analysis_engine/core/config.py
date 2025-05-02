"""
Configuration Management Module

This module provides configuration management functionality for the Analysis Engine Service.
This is a backward compatibility layer that imports from the consolidated settings module.

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

# Import from the consolidated settings module
from analysis_engine.config.settings import (
    AnalysisEngineSettings as Settings,
    get_settings,
    ConfigurationManager,
    get_db_url,
    get_redis_url,
    get_market_data_service_url,
    get_notification_service_url,
    get_analysis_settings,
    get_rate_limits,
    get_db_settings
)

# Import deprecation monitor
from analysis_engine.core.deprecation_monitor import record_usage

# Calculate days until removal
REMOVAL_DATE = datetime.date(2023, 12, 31)
days_until_removal = (REMOVAL_DATE - datetime.date.today()).days
days_message = f"{days_until_removal} days" if days_until_removal > 0 else "PAST DUE"

# Show deprecation warning with file and line information
def _show_deprecation_warning():
    # Get the caller's frame
    frame = inspect.currentframe().f_back.f_back
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    function_name = frame.f_code.co_name

    # Get relative path for better readability
    try:
        rel_path = os.path.relpath(filename)
    except ValueError:
        rel_path = filename

    # Record usage for monitoring
    record_usage("analysis_engine.core.config")

    warnings.warn(
        f"DEPRECATION WARNING: analysis_engine.core.config will be removed after {REMOVAL_DATE} ({days_message}).\n"
        f"Please import from analysis_engine.config instead.\n"
        f"Called from {rel_path}:{lineno} in function '{function_name}'\n"
        f"Migration guide: https://confluence.example.com/display/DEV/Configuration+Migration+Guide",
        DeprecationWarning,
        stacklevel=2
    )

_show_deprecation_warning()

# Create a settings instance for backward compatibility
settings = get_settings()