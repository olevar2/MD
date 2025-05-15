"""
CQRS handlers for the Market Analysis Service.

This module provides the CQRS handlers for the Market Analysis Service.
"""

from market_analysis_service.cqrs.handlers.command_handlers import (
    AnalyzeMarketCommandHandler,
    RecognizePatternsCommandHandler,
    DetectSupportResistanceCommandHandler,
    DetectMarketRegimeCommandHandler,
    AnalyzeCorrelationCommandHandler,
    AnalyzeVolatilityCommandHandler,
    AnalyzeSentimentCommandHandler
)
from market_analysis_service.cqrs.handlers.query_handlers import (
    GetAnalysisResultQueryHandler,
    ListAnalysisResultsQueryHandler,
    GetPatternRecognitionResultQueryHandler,
    ListPatternRecognitionResultsQueryHandler,
    GetSupportResistanceResultQueryHandler,
    ListSupportResistanceResultsQueryHandler,
    GetMarketRegimeResultQueryHandler,
    ListMarketRegimeResultsQueryHandler,
    GetCorrelationAnalysisResultQueryHandler,
    ListCorrelationAnalysisResultsQueryHandler,
    GetAvailableMethodsQueryHandler
)

__all__ = [
    'AnalyzeMarketCommandHandler',
    'RecognizePatternsCommandHandler',
    'DetectSupportResistanceCommandHandler',
    'DetectMarketRegimeCommandHandler',
    'AnalyzeCorrelationCommandHandler',
    'AnalyzeVolatilityCommandHandler',
    'AnalyzeSentimentCommandHandler',
    'GetAnalysisResultQueryHandler',
    'ListAnalysisResultsQueryHandler',
    'GetPatternRecognitionResultQueryHandler',
    'ListPatternRecognitionResultsQueryHandler',
    'GetSupportResistanceResultQueryHandler',
    'ListSupportResistanceResultsQueryHandler',
    'GetMarketRegimeResultQueryHandler',
    'ListMarketRegimeResultsQueryHandler',
    'GetCorrelationAnalysisResultQueryHandler',
    'ListCorrelationAnalysisResultsQueryHandler',
    'GetAvailableMethodsQueryHandler'
]