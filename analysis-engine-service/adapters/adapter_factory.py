#!/usr/bin/env python3
"""
Adapter factory for analysis service.
"""

from typing import Dict, List, Optional, Any

from .analysisprovideradapter import AnalysisProviderAdapter
from .indicatorprovideradapter import IndicatorProviderAdapter
from .patternrecognizeradapter import PatternRecognizerAdapter

class AdapterFactory:
    """
    Factory for creating adapter instances.
    """

    @classmethod
    def get_analysisprovideradapter(cls) -> AnalysisProviderAdapter:
        """Get an instance of AnalysisProviderAdapter."""
        return AnalysisProviderAdapter()
    @classmethod
    def get_indicatorprovideradapter(cls) -> IndicatorProviderAdapter:
        """Get an instance of IndicatorProviderAdapter."""
        return IndicatorProviderAdapter()
    @classmethod
    def get_patternrecognizeradapter(cls) -> PatternRecognizerAdapter:
        """Get an instance of PatternRecognizerAdapter."""
        return PatternRecognizerAdapter()
