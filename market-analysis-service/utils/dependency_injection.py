"""
Dependency injection module for the Market Analysis Service.

This module provides the dependency injection for the Market Analysis Service.
"""
import logging
from typing import Optional

from common_lib.cqrs.commands import CommandBus
from common_lib.cqrs.queries import QueryBus
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
from market_analysis_service.repositories.read_repositories import AnalysisReadRepository
from market_analysis_service.repositories.write_repositories import AnalysisWriteRepository
from market_analysis_service.services.market_analysis_service import MarketAnalysisService

logger = logging.getLogger(__name__)

# Singleton instances
_command_bus: Optional[CommandBus] = None
_query_bus: Optional[QueryBus] = None


def get_command_bus() -> CommandBus:
    """
    Get the command bus.
    
    Returns:
        The command bus
    """
    global _command_bus
    
    if _command_bus is None:
        _command_bus = CommandBus()
        
        # Create repositories
        analysis_write_repository = AnalysisWriteRepository()
        
        # Create services
        market_analysis_service = MarketAnalysisService()
        
        # Create command handlers
        analyze_market_handler = AnalyzeMarketCommandHandler(
            market_analysis_service=market_analysis_service,
            repository=analysis_write_repository
        )
        recognize_patterns_handler = RecognizePatternsCommandHandler(
            market_analysis_service=market_analysis_service,
            repository=analysis_write_repository
        )
        detect_support_resistance_handler = DetectSupportResistanceCommandHandler(
            market_analysis_service=market_analysis_service,
            repository=analysis_write_repository
        )
        detect_market_regime_handler = DetectMarketRegimeCommandHandler(
            market_analysis_service=market_analysis_service,
            repository=analysis_write_repository
        )
        analyze_correlation_handler = AnalyzeCorrelationCommandHandler(
            market_analysis_service=market_analysis_service,
            repository=analysis_write_repository
        )
        analyze_volatility_handler = AnalyzeVolatilityCommandHandler(
            market_analysis_service=market_analysis_service,
            repository=analysis_write_repository
        )
        analyze_sentiment_handler = AnalyzeSentimentCommandHandler(
            market_analysis_service=market_analysis_service,
            repository=analysis_write_repository
        )
        
        # Register command handlers
        _command_bus.register_handler(AnalyzeMarketCommand, analyze_market_handler)
        _command_bus.register_handler(RecognizePatternsCommand, recognize_patterns_handler)
        _command_bus.register_handler(DetectSupportResistanceCommand, detect_support_resistance_handler)
        _command_bus.register_handler(DetectMarketRegimeCommand, detect_market_regime_handler)
        _command_bus.register_handler(AnalyzeCorrelationCommand, analyze_correlation_handler)
        _command_bus.register_handler(AnalyzeVolatilityCommand, analyze_volatility_handler)
        _command_bus.register_handler(AnalyzeSentimentCommand, analyze_sentiment_handler)
    
    return _command_bus


def get_query_bus() -> QueryBus:
    """
    Get the query bus.
    
    Returns:
        The query bus
    """
    global _query_bus
    
    if _query_bus is None:
        _query_bus = QueryBus()
        
        # Create repositories
        analysis_read_repository = AnalysisReadRepository()
        
        # Create services
        market_analysis_service = MarketAnalysisService()
        
        # Create query handlers
        get_analysis_result_handler = GetAnalysisResultQueryHandler(
            repository=analysis_read_repository
        )
        list_analysis_results_handler = ListAnalysisResultsQueryHandler(
            repository=analysis_read_repository
        )
        get_pattern_recognition_result_handler = GetPatternRecognitionResultQueryHandler(
            repository=analysis_read_repository
        )
        list_pattern_recognition_results_handler = ListPatternRecognitionResultsQueryHandler(
            repository=analysis_read_repository
        )
        get_support_resistance_result_handler = GetSupportResistanceResultQueryHandler(
            repository=analysis_read_repository
        )
        list_support_resistance_results_handler = ListSupportResistanceResultsQueryHandler(
            repository=analysis_read_repository
        )
        get_market_regime_result_handler = GetMarketRegimeResultQueryHandler(
            repository=analysis_read_repository
        )
        list_market_regime_results_handler = ListMarketRegimeResultsQueryHandler(
            repository=analysis_read_repository
        )
        get_correlation_analysis_result_handler = GetCorrelationAnalysisResultQueryHandler(
            repository=analysis_read_repository
        )
        list_correlation_analysis_results_handler = ListCorrelationAnalysisResultsQueryHandler(
            repository=analysis_read_repository
        )
        get_available_methods_handler = GetAvailableMethodsQueryHandler(
            market_analysis_service=market_analysis_service
        )
        
        # Register query handlers
        _query_bus.register_handler(GetAnalysisResultQuery, get_analysis_result_handler)
        _query_bus.register_handler(ListAnalysisResultsQuery, list_analysis_results_handler)
        _query_bus.register_handler(GetPatternRecognitionResultQuery, get_pattern_recognition_result_handler)
        _query_bus.register_handler(ListPatternRecognitionResultsQuery, list_pattern_recognition_results_handler)
        _query_bus.register_handler(GetSupportResistanceResultQuery, get_support_resistance_result_handler)
        _query_bus.register_handler(ListSupportResistanceResultsQuery, list_support_resistance_results_handler)
        _query_bus.register_handler(GetMarketRegimeResultQuery, get_market_regime_result_handler)
        _query_bus.register_handler(ListMarketRegimeResultsQuery, list_market_regime_results_handler)
        _query_bus.register_handler(GetCorrelationAnalysisResultQuery, get_correlation_analysis_result_handler)
        _query_bus.register_handler(ListCorrelationAnalysisResultsQuery, list_correlation_analysis_results_handler)
        _query_bus.register_handler(GetAvailableMethodsQuery, get_available_methods_handler)
    
    return _query_bus


def get_market_analysis_service() -> MarketAnalysisService:
    """
    Get the Market Analysis Service instance.

    Returns:
        The Market Analysis Service instance
    """
    # Create repositories
    analysis_read_repository = AnalysisReadRepository()
    analysis_write_repository = AnalysisWriteRepository()

    # Create adapters (assuming these dependencies are available or can be initialized here)
    # TODO: Properly initialize adapters with their dependencies
    data_pipeline_adapter = DataPipelineAdapter() # Placeholder
    analysis_coordinator_adapter = AnalysisCoordinatorAdapter() # Placeholder
    feature_store_adapter = FeatureStoreAdapter() # Placeholder

    # Create and return the Market Analysis Service instance
    market_analysis_service = MarketAnalysisService(
        data_pipeline_adapter=data_pipeline_adapter,
        analysis_coordinator_adapter=analysis_coordinator_adapter,
        feature_store_adapter=feature_store_adapter,
        analysis_repository=analysis_write_repository # Assuming write repository is needed for the service
    )

    return market_analysis_service