"""
API package for the Python components of the Trading Gateway Service.

This package contains the API endpoints for the Python components of the
Trading Gateway Service, which handle market data processing, order
reconciliation, and other backend functionality.
"""

from .router import router

__all__ = ["router"]
