"""
Optimization Service

This module provides the service layer for strategy optimization.
"""
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from backtesting_service.models.backtest_models import (
    OptimizationRequest,
    OptimizationResponse,
    OptimizationResult,
    OptimizationStatus
)
from backtesting_service.repositories.optimization_repository import OptimizationRepository
from backtesting_service.adapters.data_pipeline_adapter import DataPipelineAdapter
from backtesting_service.adapters.strategy_execution_adapter import StrategyExecutionAdapter
from common_lib.resilience.decorators import with_standard_resilience

logger = logging.getLogger(__name__)


class OptimizationService:
    """
    Service for strategy optimization.
    """
    
    def __init__(
        self,
        optimization_repository: Optional[OptimizationRepository] = None,
        data_pipeline_adapter: Optional[DataPipelineAdapter] = None,
        strategy_execution_adapter: Optional[StrategyExecutionAdapter] = None
    ):
        """
        Initialize the optimization service.
        
        Args:
            optimization_repository: Repository for optimization results
            data_pipeline_adapter: Adapter for data pipeline
            strategy_execution_adapter: Adapter for strategy execution
        """
        self.optimization_repository = optimization_repository or OptimizationRepository()
        self.data_pipeline_adapter = data_pipeline_adapter or DataPipelineAdapter()
        self.strategy_execution_adapter = strategy_execution_adapter or StrategyExecutionAdapter()
        self.running_optimizations = {}
    
    @with_standard_resilience()
    async def run_optimization(self, request: OptimizationRequest) -> OptimizationResponse:
        """
        Run an optimization with the specified configuration.
        
        Args:
            request: Optimization request
            
        Returns:
            Optimization response
        """
        # Generate a unique ID for the optimization
        optimization_id = str(uuid.uuid4())
        
        # Create an optimization result with initial status
        optimization_result = OptimizationResult(
            optimization_id=optimization_id,
            request=request,
            status=OptimizationStatus.PENDING,
            start_time=datetime.now(),
            best_parameters={},
            all_evaluations=[]
        )
        
        # Save the initial optimization result
        await self.optimization_repository.save_optimization(optimization_id, optimization_result)
        
        # Start the optimization in the background
        # In a real implementation, this would be done in a background task
        # For simplicity, we'll return the initial response here
        
        # Return the initial response
        return OptimizationResponse(
            optimization_id=optimization_id,
            status=OptimizationStatus.PENDING,
            message="Optimization started successfully"
        )
    
    @with_standard_resilience()
    async def get_optimization_result(self, optimization_id: str) -> Optional[OptimizationResult]:
        """
        Get the result of an optimization.
        
        Args:
            optimization_id: ID of the optimization
            
        Returns:
            Optimization result or None if not found
        """
        return await self.optimization_repository.get_optimization(optimization_id)
    
    @with_standard_resilience()
    async def list_optimizations(
        self,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[OptimizationResult]:
        """
        List optimizations with optional filtering.
        
        Args:
            strategy_id: Optional strategy ID to filter by
            symbol: Optional symbol to filter by
            limit: Maximum number of results to return
            offset: Offset for pagination
            
        Returns:
            List of optimization results
        """
        return await self.optimization_repository.list_optimizations(
            strategy_id=strategy_id,
            symbol=symbol,
            limit=limit,
            offset=offset
        )
    
    @with_standard_resilience()
    async def cancel_optimization(self, optimization_id: str) -> bool:
        """
        Cancel a running optimization.
        
        Args:
            optimization_id: ID of the optimization to cancel
            
        Returns:
            True if the optimization was cancelled, False otherwise
        """
        if optimization_id in self.running_optimizations:
            # Cancel the optimization
            self.running_optimizations[optimization_id] = False
            
            # Update the optimization status
            optimization = await self.optimization_repository.get_optimization(optimization_id)
            if optimization:
                optimization.status = OptimizationStatus.CANCELLED
                optimization.end_time = datetime.now()
                await self.optimization_repository.save_optimization(optimization_id, optimization)
            
            return True
        
        return False