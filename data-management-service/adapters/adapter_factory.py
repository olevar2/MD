#!/usr/bin/env python3
"""
Adapter factory for data service.
"""

from typing import Dict, List, Optional, Any

from .basealternativedataadapter import BaseAlternativeDataAdapter
from .basecorrelationanalyzer import BaseCorrelationAnalyzer
from .basefeatureextractor import BaseFeatureExtractor
from .basetradingsignalgenerator import BaseTradingSignalGenerator

class AdapterFactory:
    """
    Factory for creating adapter instances.
    """

    @classmethod
    def get_basealternativedataadapter(cls) -> BaseAlternativeDataAdapter:
        """Get an instance of BaseAlternativeDataAdapter."""
        return BaseAlternativeDataAdapter()
    @classmethod
    def get_basecorrelationanalyzer(cls) -> BaseCorrelationAnalyzer:
        """Get an instance of BaseCorrelationAnalyzer."""
        return BaseCorrelationAnalyzer()
    @classmethod
    def get_basefeatureextractor(cls) -> BaseFeatureExtractor:
        """Get an instance of BaseFeatureExtractor."""
        return BaseFeatureExtractor()
    @classmethod
    def get_basetradingsignalgenerator(cls) -> BaseTradingSignalGenerator:
        """Get an instance of BaseTradingSignalGenerator."""
        return BaseTradingSignalGenerator()
