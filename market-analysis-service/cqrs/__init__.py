"""
CQRS module for the Market Analysis Service.

This module provides the CQRS implementation for the Market Analysis Service.
"""

from market_analysis_service.cqrs.commands import (
    AnalyzeMarketCommand,
    RecognizePatternsCommand,
    DetectSupportResistanceCommand,
    DetectMarketRegimeCommand,
    AnalyzeCorrelationCommand,
    AnalyzeVolatilityCommand,
    AnalyzeSentimentCommand
)
from market_analysis_service.cqrs.queries import (
    GetAnalysisResultQuery,
    ListAnalysisResultsQuery,
    GetPatternRecognitionResultQuery,
    ListPatternRecognitionResultsQuery,
    GetSupportResistanceResultQuery,
    ListSupportResistanceResultsQuery,
    GetMarketRegimeResultQuery,
    ListMarketRegimeResultsQuery,
    GetCorrelationAnalysisResultQuery,
    ListCorrelationAnalysisResultsQuery,
    GetAvailableMethodsQuery
)
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
    'AnalyzeMarketCommand',
    'RecognizePatternsCommand',
    'DetectSupportResistanceCommand',
    'DetectMarketRegimeCommand',
    'AnalyzeCorrelationCommand',
    'AnalyzeVolatilityCommand',
    'AnalyzeSentimentCommand',
    'GetAnalysisResultQuery',
    'ListAnalysisResultsQuery',
    'GetPatternRecognitionResultQuery',
    'ListPatternRecognitionResultsQuery',
    'GetSupportResistanceResultQuery',
    'ListSupportResistanceResultsQuery',
    'GetMarketRegimeResultQuery',
    'ListMarketRegimeResultsQuery',
    'GetCorrelationAnalysisResultQuery',
    'ListCorrelationAnalysisResultsQuery',
    'GetAvailableMethodsQuery',
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