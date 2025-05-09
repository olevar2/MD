"""
Adapters package for ML integration service.

This package contains adapter implementations for interfaces
to break circular dependencies between services.
"""

from ml_integration_service.adapters.analysis_engine_adapter import AnalysisEngineClientAdapter

__all__ = [
    'AnalysisEngineClientAdapter'
]
