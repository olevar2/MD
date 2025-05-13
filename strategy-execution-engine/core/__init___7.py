"""
Core Package for Strategy Execution Engine

This package contains core functionality for the Strategy Execution Engine,
including configuration, logging, dependency injection, and monitoring.
"""

from .config import get_settings
from .logging import configure_logging, get_logger
from .container import ServiceContainer
from .monitoring import setup_monitoring

__all__ = [
    "get_settings",
    "configure_logging",
    "get_logger",
    "ServiceContainer",
    "setup_monitoring"
]
