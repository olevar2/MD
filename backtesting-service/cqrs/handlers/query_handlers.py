"""
Query handlers for the Backtesting Service.

This module provides the query handlers for the Backtesting Service.
"""
import logging
from typing import Dict, List, Optional, Any

from common_lib.cqrs.queries import QueryHandler
from backtesting_service.cqrs.queries import (
    GetBacktestQuery,
    ListBacktestsQuery,
    GetOptimizationQuery,
    ListOptimizationsQuery,
    GetWalkForwardTestQuery,
    ListWalkForwardTestsQuery,
    ListStrategiesQuery
)
from backtesting_service.models.backtest_models import (
    BacktestResult,
    OptimizationResult,
    WalkForwardTestResult,
    StrategyMetadata,
    StrategyListResponse,
    BacktestListResponse
)
from backtesting_service.repositories.read_repositories import (
    BacktestReadRepository,
    OptimizationReadRepository,
    WalkForwardReadRepository
)
from backtesting_service.services.strategy_service import StrategyService

logger = logging.getLogger(__name__)


class GetBacktestQueryHandler(QueryHandler):
    """Handler for GetBacktestQuery."""
    
    def __init__(self, repository: BacktestReadRepository):
        """
        Initialize the handler.
        
        Args:
            repository: Backtest read repository
        """
        self.repository = repository
    
    async def handle(self, query: GetBacktestQuery) -> Optional[BacktestResult]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            The backtest result or None if not found
        """
        logger.info(f"Handling GetBacktestQuery: {query}")
        
        return await self.repository.get_by_id(query.backtest_id)


class ListBacktestsQueryHandler(QueryHandler):
    """Handler for ListBacktestsQuery."""
    
    def __init__(self, repository: BacktestReadRepository):
        """
        Initialize the handler.
        
        Args:
            repository: Backtest read repository
        """
        self.repository = repository
    
    async def handle(self, query: ListBacktestsQuery) -> BacktestListResponse:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            List of backtests
        """
        logger.info(f"Handling ListBacktestsQuery: {query}")
        
        # Build criteria
        criteria = {}
        if query.strategy_id:
            criteria["strategy_id"] = query.strategy_id
        if query.symbol:
            criteria["symbol"] = query.symbol
        
        # Get backtests by criteria
        backtests = await self.repository.get_by_criteria(criteria)
        
        # Apply pagination
        total = len(backtests)
        backtests = backtests[query.offset:query.offset + query.limit]
        
        return BacktestListResponse(
            backtests=backtests,
            total=total,
            limit=query.limit,
            offset=query.offset
        )


class GetOptimizationQueryHandler(QueryHandler):
    """Handler for GetOptimizationQuery."""
    
    def __init__(self, repository: OptimizationReadRepository):
        """
        Initialize the handler.
        
        Args:
            repository: Optimization read repository
        """
        self.repository = repository
    
    async def handle(self, query: GetOptimizationQuery) -> Optional[OptimizationResult]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            The optimization result or None if not found
        """
        logger.info(f"Handling GetOptimizationQuery: {query}")
        
        return await self.repository.get_by_id(query.optimization_id)


class ListOptimizationsQueryHandler(QueryHandler):
    """Handler for ListOptimizationsQuery."""
    
    def __init__(self, repository: OptimizationReadRepository):
        """
        Initialize the handler.
        
        Args:
            repository: Optimization read repository
        """
        self.repository = repository
    
    async def handle(self, query: ListOptimizationsQuery) -> List[OptimizationResult]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            List of optimizations
        """
        logger.info(f"Handling ListOptimizationsQuery: {query}")
        
        # Build criteria
        criteria = {}
        if query.strategy_id:
            criteria["strategy_id"] = query.strategy_id
        if query.symbol:
            criteria["symbol"] = query.symbol
        
        # Get optimizations by criteria
        optimizations = await self.repository.get_by_criteria(criteria)
        
        # Apply pagination
        optimizations = optimizations[query.offset:query.offset + query.limit]
        
        return optimizations


class GetWalkForwardTestQueryHandler(QueryHandler):
    """Handler for GetWalkForwardTestQuery."""
    
    def __init__(self, repository: WalkForwardReadRepository):
        """
        Initialize the handler.
        
        Args:
            repository: Walk-forward test read repository
        """
        self.repository = repository
    
    async def handle(self, query: GetWalkForwardTestQuery) -> Optional[WalkForwardTestResult]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            The walk-forward test result or None if not found
        """
        logger.info(f"Handling GetWalkForwardTestQuery: {query}")
        
        return await self.repository.get_by_id(query.test_id)


class ListWalkForwardTestsQueryHandler(QueryHandler):
    """Handler for ListWalkForwardTestsQuery."""
    
    def __init__(self, repository: WalkForwardReadRepository):
        """
        Initialize the handler.
        
        Args:
            repository: Walk-forward test read repository
        """
        self.repository = repository
    
    async def handle(self, query: ListWalkForwardTestsQuery) -> List[WalkForwardTestResult]:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            List of walk-forward tests
        """
        logger.info(f"Handling ListWalkForwardTestsQuery: {query}")
        
        # Build criteria
        criteria = {}
        if query.strategy_id:
            criteria["strategy_id"] = query.strategy_id
        if query.symbol:
            criteria["symbol"] = query.symbol
        
        # Get walk-forward tests by criteria
        tests = await self.repository.get_by_criteria(criteria)
        
        # Apply pagination
        tests = tests[query.offset:query.offset + query.limit]
        
        return tests


class ListStrategiesQueryHandler(QueryHandler):
    """Handler for ListStrategiesQuery."""
    
    def __init__(self, strategy_service: StrategyService):
        """
        Initialize the handler.
        
        Args:
            strategy_service: Strategy service
        """
        self.strategy_service = strategy_service
    
    async def handle(self, query: ListStrategiesQuery) -> StrategyListResponse:
        """
        Handle the query.
        
        Args:
            query: The query
            
        Returns:
            List of strategies
        """
        logger.info(f"Handling ListStrategiesQuery: {query}")
        
        # Get strategies
        strategies = await self.strategy_service.get_strategies(
            category=query.category,
            limit=query.limit,
            offset=query.offset
        )
        
        return strategies