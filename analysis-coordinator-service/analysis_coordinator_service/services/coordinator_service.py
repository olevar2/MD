from typing import Dict, List, Any, Optional
import logging
import uuid
import asyncio
from datetime import datetime, timedelta, UTC

from analysis_coordinator_service.models.coordinator_models import (
    IntegratedAnalysisRequest,
    IntegratedAnalysisResponse,
    AnalysisTaskRequest,
    AnalysisTaskResponse,
    AnalysisTaskStatus,
    AnalysisTaskResult,
    AnalysisServiceType,
    AnalysisTaskStatusEnum
)
from analysis_coordinator_service.adapters.market_analysis_adapter import MarketAnalysisAdapter
from analysis_coordinator_service.adapters.causal_analysis_adapter import CausalAnalysisAdapter
from analysis_coordinator_service.adapters.backtesting_adapter import BacktestingAdapter
from analysis_coordinator_service.repositories.task_repository import TaskRepository
from analysis_coordinator_service.utils.resilience import with_retry, with_circuit_breaker
from analysis_coordinator_service.workflows.workflow_factory import WorkflowFactory

logger = logging.getLogger(__name__)

class CoordinatorService:
    """
    Service for coordinating analysis tasks across multiple analysis services.
    """

    def __init__(
        self,
        market_analysis_adapter: MarketAnalysisAdapter,
        causal_analysis_adapter: CausalAnalysisAdapter,
        backtesting_adapter: BacktestingAdapter,
        task_repository: TaskRepository
    ):
        self.market_analysis_adapter = market_analysis_adapter
        self.causal_analysis_adapter = causal_analysis_adapter
        self.backtesting_adapter = backtesting_adapter
        self.task_repository = task_repository
        
        # Create workflow factory
        self.workflow_factory = WorkflowFactory(
            task_repository=task_repository,
            market_analysis_adapter=market_analysis_adapter,
            causal_analysis_adapter=causal_analysis_adapter,
            backtesting_adapter=backtesting_adapter
        )

    async def run_integrated_analysis(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        services: List[str] = None,
        parameters: Dict[str, Any] = None
    ) -> IntegratedAnalysisResponse:
        """
        Run an integrated analysis across multiple analysis services.
        """
        if services is None:
            services = ["market_analysis", "causal_analysis"]
        
        if parameters is None:
            parameters = {}
            
        logger.info(f"Running integrated analysis for symbol {symbol} with services {services}")

        # Create a new task
        task_id = str(uuid.uuid4())
        estimated_completion_time = datetime.now(UTC) + timedelta(minutes=5)  # Estimate 5 minutes for completion

        # Create task in repository
        await self.task_repository.create_integrated_task(
            task_id=task_id,
            services=services,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            parameters=parameters
        )

        # Start the task asynchronously using asyncio.create_task
        asyncio.create_task(self._execute_integrated_analysis(
            task_id=task_id,
            services=services,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            parameters=parameters
        ))

        return IntegratedAnalysisResponse(
            task_id=task_id,
            status=AnalysisTaskStatusEnum.PENDING,
            created_at=datetime.now(UTC),
            estimated_completion_time=estimated_completion_time
        )

    async def create_analysis_task(
        self,
        service_type: str,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        parameters: Dict[str, Any] = None
    ) -> AnalysisTaskResponse:
        """
        Create a new analysis task.
        """
        if parameters is None:
            parameters = {}
            
        logger.info(f"Creating {service_type} analysis task for symbol {symbol}")

        # Create a new task
        task_id = str(uuid.uuid4())
        estimated_completion_time = datetime.now(UTC) + timedelta(minutes=2)  # Estimate 2 minutes for completion

        # Create task in repository
        await self.task_repository.create_task(
            task_id=task_id,
            service_type=service_type,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            parameters=parameters
        )

        # Start the task asynchronously using asyncio.create_task
        asyncio.create_task(self._execute_analysis_task(
            task_id=task_id,
            service_type=service_type,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            parameters=parameters
        ))

        return AnalysisTaskResponse(
            task_id=task_id,
            service_type=service_type,
            status=AnalysisTaskStatusEnum.PENDING,
            created_at=datetime.now(UTC),
            estimated_completion_time=estimated_completion_time
        )

    async def get_task_result(self, task_id: str) -> Optional[AnalysisTaskResult]:
        """
        Get the result of a previously created analysis task.
        """
        logger.info(f"Getting result for task {task_id}")

        # Get task from repository
        # Try get_task first, then fall back to get_by_id for compatibility
        try:
            task = await self.task_repository.get_task(task_id)
        except AttributeError:
            try:
                task = await self.task_repository.get_by_id(task_id)
            except AttributeError:
                return None
                
        if not task:
            return None

        return task

    async def get_task_status(self, task_id: str) -> Optional[AnalysisTaskStatus]:
        """
        Get the status of a previously created analysis task.
        """
        logger.info(f"Getting status for task {task_id}")

        # Get task from repository
        try:
            task = await self.task_repository.get_task_status(task_id)
        except AttributeError:
            return None
            
        if not task:
            return None

        # Convert dictionary to AnalysisTaskStatus if needed
        if isinstance(task, dict):
            return AnalysisTaskStatus(
                task_id=task.get("task_id", task_id),
                service_type=task.get("service_type", "unknown"),
                status=task.get("status", AnalysisTaskStatusEnum.PENDING),
                created_at=task.get("created_at", datetime.now(UTC)),
                progress=task.get("progress", 0.0),
                message=task.get("message", "")
            )
            
        return task

    async def delete_task(self, task_id: str) -> bool:
        """
        Delete a previously created analysis task.
        """
        logger.info(f"Deleting task {task_id}")

        # Delete task from repository
        try:
            return await self.task_repository.delete_task(task_id)
        except AttributeError:
            return False

    async def list_tasks(
        self,
        service_type: Optional[str] = None,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List all analysis tasks with optional filtering.
        """
        logger.info(f"Listing tasks with limit {limit}, offset {offset}, status {status}")

        # Get tasks from repository
        try:
            tasks = await self.task_repository.list_tasks(limit, offset, status)
        except AttributeError:
            # Try to use get_by_criteria for compatibility
            try:
                criteria = {}
                if service_type:
                    criteria["service_type"] = service_type
                if status:
                    criteria["status"] = status
                    
                tasks = await self.task_repository.get_by_criteria(criteria)
                
                # Apply pagination
                tasks = tasks[offset:offset + limit]
            except AttributeError:
                tasks = []

        return tasks

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running analysis task.
        """
        logger.info(f"Cancelling task {task_id}")

        # Get task from repository
        try:
            return await self.task_repository.cancel_task(task_id)
        except AttributeError:
            # Try to update task status for compatibility
            try:
                task = await self.get_task_status(task_id)
                if not task:
                    return False
                    
                # Check if task can be cancelled
                if isinstance(task, dict):
                    status = task.get("status", AnalysisTaskStatusEnum.PENDING)
                else:
                    status = task.status
                    
                if status not in [AnalysisTaskStatusEnum.PENDING, AnalysisTaskStatusEnum.RUNNING]:
                    return False
                    
                # Update task status in repository
                await self.task_repository.update_task_status(
                    task_id=task_id,
                    status=AnalysisTaskStatusEnum.CANCELLED,
                    message="Task cancelled by user"
                )
                
                return True
            except AttributeError:
                return False

    async def get_available_services(self) -> Dict[str, Any]:
        """
        Get available analysis services and their capabilities.
        """
        logger.info("Getting available services")

        # In a real implementation, this would involve calling each service to get its capabilities
        # For simplicity, we'll return a static list

        return {
            "market_analysis": [
                "pattern_recognition",
                "support_resistance",
                "market_regime",
                "correlation_analysis"
            ],
            "causal_analysis": [
                "causal_graph",
                "intervention_effect",
                "counterfactual_scenario"
            ],
            "backtesting": [
                "strategy_backtest",
                "performance_analysis",
                "optimization"
            ]
        }

    @with_retry(max_retries=3, backoff_factor=0.5)
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    async def _execute_analysis_task(
        self,
        task_id: str,
        service_type: AnalysisServiceType,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        parameters: Dict[str, Any] = None
    ) -> None:
        """
        Execute an analysis task based on service type.

        Args:
            task_id: Task ID
            service_type: Type of analysis service
            symbol: Symbol to analyze
            timeframe: Timeframe to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            parameters: Additional parameters for the analysis
        """
        if parameters is None:
            parameters = {}

        try:
            # Create workflow for the service type
            workflow = self.workflow_factory.create_workflow(service_type)
            
            # Execute the workflow
            await workflow.execute(
                task_id=task_id,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                parameters=parameters
            )
            
        except Exception as e:
            logger.error(f"Error executing {service_type} analysis task: {str(e)}")

            # Update task status to failed
            await self.task_repository.update_task_status(
                task_id=task_id,
                status=AnalysisTaskStatusEnum.FAILED,
                message=f"Failed to execute {service_type} analysis: {str(e)}",
                error=str(e)
            )

    @with_retry(max_retries=3, backoff_factor=0.5)
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    async def _execute_integrated_analysis(
        self,
        task_id: str,
        services: List[AnalysisServiceType],
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        parameters: Dict[str, Any] = None
    ) -> None:
        """
        Execute an integrated analysis across multiple services.

        Args:
            task_id: Task ID
            services: List of services to use for analysis
            symbol: Symbol to analyze
            timeframe: Timeframe to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            parameters: Additional parameters for the analysis
        """
        if parameters is None:
            parameters = {}

        try:
            # Create integrated analysis workflow
            workflow = self.workflow_factory.create_workflow("integrated_analysis")
            
            # Add services to parameters
            parameters["services"] = services
            
            # Execute the workflow
            await workflow.execute(
                task_id=task_id,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                parameters=parameters
            )
            
        except Exception as e:
            logger.error(f"Error executing integrated analysis task: {str(e)}")

            # Update task status to failed
            await self.task_repository.update_task_status(
                task_id=task_id,
                status=AnalysisTaskStatusEnum.FAILED,
                message=f"Failed to execute integrated analysis: {str(e)}",
                error=str(e)
            )

    def _integrate_analysis_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate results from multiple analysis services.
        
        This method is kept for backward compatibility but delegates to the
        IntegratedAnalysisWorkflow for actual implementation.

        Args:
            results: Dictionary of results from different services

        Returns:
            Integrated analysis results
        """
        # Create integrated analysis workflow
        workflow = self.workflow_factory.create_workflow("integrated_analysis")
        
        # Use the workflow's aggregate_results method
        return workflow.aggregate_results(results)