"""
Adapters package for ML integration service.

This package contains adapter implementations for interfaces
to break circular dependencies between services.
"""

from adapters.analysis_engine_adapter import AnalysisEngineClientAdapter

__all__ = [
    'AnalysisEngineClientAdapter'
]
