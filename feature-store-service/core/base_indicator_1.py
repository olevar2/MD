"""
Base Indicator Module.

This module provides the base class for all technical indicators in the feature store.
It now imports the BaseIndicator class from common_lib to ensure consistency across services.

Note: This file is being maintained for backward compatibility.
New code should import BaseIndicator from common_lib.indicators directly.
"""

from common_lib.indicators.base_indicator import BaseIndicator

# Re-export the BaseIndicator class
__all__ = ['BaseIndicator']
