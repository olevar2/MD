"""
Configuration package for the Analysis Engine Service.

This package provides a centralized configuration management system
that consolidates settings from various sources.
"""

from analysis_engine.config.settings import AnalysisEngineSettings, get_settings

__all__ = ["AnalysisEngineSettings", "get_settings"]
