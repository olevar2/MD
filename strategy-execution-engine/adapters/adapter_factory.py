#!/usr/bin/env python3
"""
Adapter factory for strategy service.
"""

from typing import Dict, List, Optional, Any

from .analysisprovideradapter import AnalysisProviderAdapter
from .analysisengineadapter import AnalysisEngineAdapter

class AdapterFactory:
    """
    Factory for creating adapter instances.
    """

    @classmethod
    def get_analysisprovideradapter(cls) -> AnalysisProviderAdapter:
        """Get an instance of AnalysisProviderAdapter."""
        return AnalysisProviderAdapter()
    @classmethod
    def get_analysisengineadapter(cls) -> AnalysisEngineAdapter:
        """Get an instance of AnalysisEngineAdapter."""
        return AnalysisEngineAdapter()
