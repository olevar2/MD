"""
API Package

This package provides API-related functionality for the Monitoring & Alerting Service.
"""

from .middleware import CorrelationIdMiddleware

__all__ = [
    "CorrelationIdMiddleware"
]