"""
Strategy Service

This module provides the service layer for strategy management.
"""
import logging
from typing import Dict, List, Any, Optional

from backtesting_service.models.backtest_models import (
    StrategyMetadata,
    StrategyListResponse
)
from backtesting_service.repositories.strategy_repository import StrategyRepository
from common_lib.resilience.decorators import with_standard_resilience

logger = logging.getLogger(__name__)


class StrategyService:
    """
    Service for strategy management.
    """
    
    def __init__(
        self,
        strategy_repository: Optional[StrategyRepository] = None
    ):
        """
        Initialize the strategy service.
        
        Args:
            strategy_repository: Repository for strategy metadata
        """
        self.strategy_repository = strategy_repository or StrategyRepository()
    
    @with_standard_resilience()
    async def get_strategy(self, strategy_id: str) -> Optional[StrategyMetadata]:
        """
        Get a strategy by ID.
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            Strategy metadata or None if not found
        """
        return await self.strategy_repository.get_strategy(strategy_id)
    
    @with_standard_resilience()
    async def list_strategies(
        self,
        category: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> StrategyListResponse:
        """
        List strategies with optional filtering.
        
        Args:
            category: Optional category to filter by
            limit: Maximum number of results to return
            offset: Offset for pagination
            
        Returns:
            List of strategy metadata
        """
        strategies = await self.strategy_repository.list_strategies(
            category=category,
            limit=limit,
            offset=offset
        )
        
        total_count = await self.strategy_repository.count_strategies(category=category)
        
        return StrategyListResponse(
            strategies=strategies,
            total_count=total_count,
            limit=limit,
            offset=offset
        )
    
    @with_standard_resilience()
    async def get_strategy_parameters(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get the parameters for a strategy.
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            Dictionary of parameter metadata
        """
        strategy = await self.get_strategy(strategy_id)
        
        if not strategy:
            return {}
        
        return strategy.parameters
    
    @with_standard_resilience()
    async def get_strategy_code(self, strategy_id: str) -> Optional[str]:
        """
        Get the code for a strategy.
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            Strategy code or None if not found
        """
        return await self.strategy_repository.get_strategy_code(strategy_id)
    
    @with_standard_resilience()
    async def get_strategy_performance(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get the performance metrics for a strategy.
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            Dictionary of performance metrics
        """
        return await self.strategy_repository.get_strategy_performance(strategy_id)