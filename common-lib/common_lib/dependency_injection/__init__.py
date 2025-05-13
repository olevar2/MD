"""
Dependency Injection Module

This module provides a standardized dependency injection framework for the Forex Trading Platform.
It helps manage service dependencies, lifecycle, and configuration in a consistent way across services.
"""

from .container import ServiceContainer, ServiceProvider, ServiceLifetime
from .decorators import inject, async_inject

__all__ = [
    'ServiceContainer',
    'ServiceProvider',
    'ServiceLifetime',
    'inject',
    'async_inject'
]
