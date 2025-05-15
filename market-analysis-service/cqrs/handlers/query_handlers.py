"""
Query handlers for the Market Analysis Service.

This module provides the query handlers for the Market Analysis Service.
"""
import logging
from typing import Dict, List, Optional, Any

from common_lib.cqrs.queries import QueryHandler
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
from market_analysis_service.models.market_analysis_models import (
    MarketAnalysisResponse,
    PatternRecognitionResponse,
    SupportResistanceResponse,
    MarketRegimeResponse,
    CorrelationAnalysisResponse,
    AnalysisType,
    PatternType,
    MarketRegimeType,
    SupportResistanceMethod
)
from market_analysis_service.repositories.read_repositories import AnalysisReadRepository
from market_analysis_service.services.market_analysis_service import MarketAnalysisService

logger = logging.getLogger(__name__)


class GetAnalysisResultQueryHandler(QueryHandler):
    """Handler for GetAnalysisResultQuery."""
    
    def __init__(self, repository: AnalysisReadRepository):
        """
        Initialize the handler.
        
        Args:
            repository: Analysis read repository
        """
        self.repository = repository
    
    async def handle(self, query: GetAnalysisResultQuery) -> Optional[MarketAnalysisResponse]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            The analysis result or None if not found
        """
        logger.info(f"Handling GetAnalysisResultQuery: {query}")
        
        result = await self.repository.get_by_id(query.analysis_id)
        
        if result and isinstance(result, MarketAnalysisResponse):
            return result
        
        return None


class ListAnalysisResultsQueryHandler(QueryHandler):
    """Handler for ListAnalysisResultsQuery."""
    
    def __init__(self, repository: AnalysisReadRepository):
        """
        Initialize the handler.
        
        Args:
            repository: Analysis read repository
        """
        self.repository = repository
    
    async def handle(self, query: ListAnalysisResultsQuery) -> List[MarketAnalysisResponse]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            List of analysis results
        """
        logger.info(f"Handling ListAnalysisResultsQuery: {query}")
        
        # Build criteria
        criteria = {}
        if query.symbol:
            criteria["symbol"] = query.symbol
        if query.timeframe:
            criteria["timeframe"] = query.timeframe
        if query.analysis_type:
            criteria["analysis_type"] = query.analysis_type
        if query.start_date:
            criteria["start_date"] = query.start_date
        if query.end_date:
            criteria["end_date"] = query.end_date
        
        # Get results by criteria
        all_results = await self.repository.get_by_criteria(criteria)
        
        # Filter for MarketAnalysisResponse
        results = [r for r in all_results if isinstance(r, MarketAnalysisResponse)]
        
        # Apply pagination
        results = results[query.offset:query.offset + query.limit]
        
        return results


class GetPatternRecognitionResultQueryHandler(QueryHandler):
    """Handler for GetPatternRecognitionResultQuery."""
    
    def __init__(self, repository: AnalysisReadRepository):
        """
        Initialize the handler.
        
        Args:
            repository: Analysis read repository
        """
        self.repository = repository
    
    async def handle(self, query: GetPatternRecognitionResultQuery) -> Optional[PatternRecognitionResponse]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            The pattern recognition result or None if not found
        """
        logger.info(f"Handling GetPatternRecognitionResultQuery: {query}")
        
        result = await self.repository.get_by_id(query.result_id)
        
        if result and isinstance(result, PatternRecognitionResponse):
            return result
        
        return None


class ListPatternRecognitionResultsQueryHandler(QueryHandler):
    """Handler for ListPatternRecognitionResultsQuery."""
    
    def __init__(self, repository: AnalysisReadRepository):
        """
        Initialize the handler.
        
        Args:
            repository: Analysis read repository
        """
        self.repository = repository
    
    async def handle(self, query: ListPatternRecognitionResultsQuery) -> List[PatternRecognitionResponse]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            List of pattern recognition results
        """
        logger.info(f"Handling ListPatternRecognitionResultsQuery: {query}")
        
        # Build criteria
        criteria = {}
        if query.symbol:
            criteria["symbol"] = query.symbol
        if query.timeframe:
            criteria["timeframe"] = query.timeframe
        if query.pattern_type:
            criteria["pattern_type"] = query.pattern_type
        if query.start_date:
            criteria["start_date"] = query.start_date
        if query.end_date:
            criteria["end_date"] = query.end_date
        
        # Get results by criteria
        all_results = await self.repository.get_by_criteria(criteria)
        
        # Filter for PatternRecognitionResponse
        results = [r for r in all_results if isinstance(r, PatternRecognitionResponse)]
        
        # Apply pagination
        results = results[query.offset:query.offset + query.limit]
        
        return results


class GetSupportResistanceResultQueryHandler(QueryHandler):
    """Handler for GetSupportResistanceResultQuery."""
    
    def __init__(self, repository: AnalysisReadRepository):
        """
        Initialize the handler.
        
        Args:
            repository: Analysis read repository
        """
        self.repository = repository
    
    async def handle(self, query: GetSupportResistanceResultQuery) -> Optional[SupportResistanceResponse]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            The support/resistance result or None if not found
        """
        logger.info(f"Handling GetSupportResistanceResultQuery: {query}")
        
        result = await self.repository.get_by_id(query.result_id)
        
        if result and isinstance(result, SupportResistanceResponse):
            return result
        
        return None


class ListSupportResistanceResultsQueryHandler(QueryHandler):
    """Handler for ListSupportResistanceResultsQuery."""
    
    def __init__(self, repository: AnalysisReadRepository):
        """
        Initialize the handler.
        
        Args:
            repository: Analysis read repository
        """
        self.repository = repository
    
    async def handle(self, query: ListSupportResistanceResultsQuery) -> List[SupportResistanceResponse]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            List of support/resistance results
        """
        logger.info(f"Handling ListSupportResistanceResultsQuery: {query}")
        
        # Build criteria
        criteria = {}
        if query.symbol:
            criteria["symbol"] = query.symbol
        if query.timeframe:
            criteria["timeframe"] = query.timeframe
        if query.method:
            criteria["method"] = query.method
        if query.start_date:
            criteria["start_date"] = query.start_date
        if query.end_date:
            criteria["end_date"] = query.end_date
        
        # Get results by criteria
        all_results = await self.repository.get_by_criteria(criteria)
        
        # Filter for SupportResistanceResponse
        results = [r for r in all_results if isinstance(r, SupportResistanceResponse)]
        
        # Apply pagination
        results = results[query.offset:query.offset + query.limit]
        
        return results


class GetMarketRegimeResultQueryHandler(QueryHandler):
    """Handler for GetMarketRegimeResultQuery."""
    
    def __init__(self, repository: AnalysisReadRepository):
        """
        Initialize the handler.
        
        Args:
            repository: Analysis read repository
        """
        self.repository = repository
    
    async def handle(self, query: GetMarketRegimeResultQuery) -> Optional[MarketRegimeResponse]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            The market regime result or None if not found
        """
        logger.info(f"Handling GetMarketRegimeResultQuery: {query}")
        
        result = await self.repository.get_by_id(query.result_id)
        
        if result and isinstance(result, MarketRegimeResponse):
            return result
        
        return None


class ListMarketRegimeResultsQueryHandler(QueryHandler):
    """Handler for ListMarketRegimeResultsQuery."""
    
    def __init__(self, repository: AnalysisReadRepository):
        """
        Initialize the handler.
        
        Args:
            repository: Analysis read repository
        """
        self.repository = repository
    
    async def handle(self, query: ListMarketRegimeResultsQuery) -> List[MarketRegimeResponse]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            List of market regime results
        """
        logger.info(f"Handling ListMarketRegimeResultsQuery: {query}")
        
        # Build criteria
        criteria = {}
        if query.symbol:
            criteria["symbol"] = query.symbol
        if query.timeframe:
            criteria["timeframe"] = query.timeframe
        if query.regime_type:
            criteria["regime_type"] = query.regime_type
        if query.start_date:
            criteria["start_date"] = query.start_date
        if query.end_date:
            criteria["end_date"] = query.end_date
        
        # Get results by criteria
        all_results = await self.repository.get_by_criteria(criteria)
        
        # Filter for MarketRegimeResponse
        results = [r for r in all_results if isinstance(r, MarketRegimeResponse)]
        
        # Apply pagination
        results = results[query.offset:query.offset + query.limit]
        
        return results


class GetCorrelationAnalysisResultQueryHandler(QueryHandler):
    """Handler for GetCorrelationAnalysisResultQuery."""
    
    def __init__(self, repository: AnalysisReadRepository):
        """
        Initialize the handler.
        
        Args:
            repository: Analysis read repository
        """
        self.repository = repository
    
    async def handle(self, query: GetCorrelationAnalysisResultQuery) -> Optional[CorrelationAnalysisResponse]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            The correlation analysis result or None if not found
        """
        logger.info(f"Handling GetCorrelationAnalysisResultQuery: {query}")
        
        result = await self.repository.get_by_id(query.result_id)
        
        if result and isinstance(result, CorrelationAnalysisResponse):
            return result
        
        return None


class ListCorrelationAnalysisResultsQueryHandler(QueryHandler):
    """Handler for ListCorrelationAnalysisResultsQuery."""
    
    def __init__(self, repository: AnalysisReadRepository):
        """
        Initialize the handler.
        
        Args:
            repository: Analysis read repository
        """
        self.repository = repository
    
    async def handle(self, query: ListCorrelationAnalysisResultsQuery) -> List[CorrelationAnalysisResponse]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            List of correlation analysis results
        """
        logger.info(f"Handling ListCorrelationAnalysisResultsQuery: {query}")
        
        # Build criteria
        criteria = {}
        if query.symbols:
            criteria["symbols"] = query.symbols
        if query.timeframe:
            criteria["timeframe"] = query.timeframe
        if query.method:
            criteria["method"] = query.method
        if query.start_date:
            criteria["start_date"] = query.start_date
        if query.end_date:
            criteria["end_date"] = query.end_date
        
        # Get results by criteria
        all_results = await self.repository.get_by_criteria(criteria)
        
        # Filter for CorrelationAnalysisResponse
        results = [r for r in all_results if isinstance(r, CorrelationAnalysisResponse)]
        
        # Apply pagination
        results = results[query.offset:query.offset + query.limit]
        
        return results


class GetAvailableMethodsQueryHandler(QueryHandler):
    """Handler for GetAvailableMethodsQuery."""
    
    def __init__(self, market_analysis_service: MarketAnalysisService):
        """
        Initialize the handler.
        
        Args:
            market_analysis_service: Market analysis service
        """
        self.market_analysis_service = market_analysis_service
    
    async def handle(self, query: GetAvailableMethodsQuery) -> Dict[str, List[str]]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            Dictionary of available methods
        """
        logger.info(f"Handling GetAvailableMethodsQuery: {query}")
        
        return await self.market_analysis_service.get_available_methods()