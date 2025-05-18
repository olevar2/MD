"""
Walk Forward Service

This module provides the service layer for walk-forward testing.
"""
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from backtesting_service.models.backtest_models import (
    WalkForwardTestRequest,
    WalkForwardTestResponse,
    WalkForwardTestResult,
    WalkForwardTestStatus
)
from backtesting_service.repositories.walk_forward_repository import WalkForwardRepository
from backtesting_service.adapters.data_pipeline_adapter import DataPipelineAdapter
from backtesting_service.adapters.strategy_execution_adapter import StrategyExecutionAdapter
from common_lib.resilience.decorators import with_standard_resilience

logger = logging.getLogger(__name__)


class WalkForwardService:
    """
    Service for walk-forward testing.
    """
    
    def __init__(
        self,
        walk_forward_repository: Optional[WalkForwardRepository] = None,
        data_pipeline_adapter: Optional[DataPipelineAdapter] = None,
        strategy_execution_adapter: Optional[StrategyExecutionAdapter] = None
    ):
        """
        Initialize the walk-forward service.
        
        Args:
            walk_forward_repository: Repository for walk-forward test results
            data_pipeline_adapter: Adapter for data pipeline
            strategy_execution_adapter: Adapter for strategy execution
        """
        self.walk_forward_repository = walk_forward_repository or WalkForwardRepository()
        self.data_pipeline_adapter = data_pipeline_adapter or DataPipelineAdapter()
        self.strategy_execution_adapter = strategy_execution_adapter or StrategyExecutionAdapter()
        self.running_tests = {}
    
    @with_standard_resilience()
    async def run_walk_forward_test(self, request: WalkForwardTestRequest) -> WalkForwardTestResponse:
        """
        Run a walk-forward test with the specified configuration.
        
        Args:
            request: Walk-forward test request
            
        Returns:
            Walk-forward test response
        """
        # Generate a unique ID for the test
        test_id = str(uuid.uuid4())
        
        # Create a test result with initial status
        test_result = WalkForwardTestResult(
            test_id=test_id,
            request=request,
            status=WalkForwardTestStatus.PENDING,
            start_time=datetime.now(),
            windows=[],
            parameters_by_window={}
        )
        
        # Save the initial test result
        await self.walk_forward_repository.save_walk_forward_test(test_id, test_result)
        
        # Start the test in the background
        # In a real implementation, this would be done in a background task
        # For simplicity, we'll return the initial response here
        
        # Return the initial response
        return WalkForwardTestResponse(
            test_id=test_id,
            status=WalkForwardTestStatus.PENDING,
            message="Walk-forward test started successfully"
        )
    
    @with_standard_resilience()
    async def get_walk_forward_test_result(self, test_id: str) -> Optional[WalkForwardTestResult]:
        """
        Get the result of a walk-forward test.
        
        Args:
            test_id: ID of the test
            
        Returns:
            Walk-forward test result or None if not found
        """
        return await self.walk_forward_repository.get_walk_forward_test(test_id)
    
    @with_standard_resilience()
    async def list_walk_forward_tests(
        self,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[WalkForwardTestResult]:
        """
        List walk-forward tests with optional filtering.
        
        Args:
            strategy_id: Optional strategy ID to filter by
            symbol: Optional symbol to filter by
            limit: Maximum number of results to return
            offset: Offset for pagination
            
        Returns:
            List of walk-forward test results
        """
        return await self.walk_forward_repository.list_walk_forward_tests(
            strategy_id=strategy_id,
            symbol=symbol,
            limit=limit,
            offset=offset
        )
    
    @with_standard_resilience()
    async def cancel_walk_forward_test(self, test_id: str) -> bool:
        """
        Cancel a running walk-forward test.
        
        Args:
            test_id: ID of the test to cancel
            
        Returns:
            True if the test was cancelled, False otherwise
        """
        if test_id in self.running_tests:
            # Cancel the test
            self.running_tests[test_id] = False
            
            # Update the test status
            test = await self.walk_forward_repository.get_walk_forward_test(test_id)
            if test:
                test.status = WalkForwardTestStatus.CANCELLED
                test.end_time = datetime.now()
                await self.walk_forward_repository.save_walk_forward_test(test_id, test)
            
            return True
        
        return False