"""
Standardized API Package

This package contains standardized API endpoints for the Analysis Engine Service.
All endpoints follow the platform's standardized API design patterns.
"""

from analysis_engine.api.v1.standardized.routes import setup_standardized_routes

__all__ = ["setup_standardized_routes"]
