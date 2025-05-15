"""
Core

This package provides core functionality for the causal analysis service.
"""

from causal_analysis_service.core.service_dependencies import (
    get_causal_repository,
    get_causal_service
)

__all__ = [
    'get_causal_repository',
    'get_causal_service'
]