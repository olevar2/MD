"""
Analysis package for common interfaces and models.
"""

from .interfaces import (
    MarketRegimeType,
    AnalysisTimeframe,
    IMarketRegimeAnalyzer,
    IMultiAssetAnalyzer,
    IPatternRecognizer,
    IAnalysisEngine
)

__all__ = [
    'MarketRegimeType',
    'AnalysisTimeframe',
    'IMarketRegimeAnalyzer',
    'IMultiAssetAnalyzer',
    'IPatternRecognizer',
    'IAnalysisEngine'
]
