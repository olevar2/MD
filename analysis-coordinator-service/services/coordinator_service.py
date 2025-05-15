from typing import Dict, List, Any, Optional
import logging
import uuid
import asyncio
from datetime import datetime, timedelta

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

    async def run_integrated_analysis(self, request: IntegratedAnalysisRequest) -> IntegratedAnalysisResponse:
        """
        Run an integrated analysis across multiple analysis services.
        """
        logger.info(f"Running integrated analysis for symbol {request.symbol} with services {request.services}")

        # Create a new task
        task_id = str(uuid.uuid4())
        estimated_completion_time = datetime.utcnow() + timedelta(minutes=5)  # Estimate 5 minutes for completion

        # Create task in repository
        await self.task_repository.create_integrated_task(
            task_id=task_id,
            services=request.services,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            parameters=request.parameters
        )

        # Start the task asynchronously using asyncio.create_task
        asyncio.create_task(self._execute_integrated_analysis(
            task_id=task_id,
            services=request.services,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            parameters=request.parameters
        ))

        return IntegratedAnalysisResponse(
            task_id=task_id,
            status=AnalysisTaskStatusEnum.PENDING,
            created_at=datetime.utcnow(),
            estimated_completion_time=estimated_completion_time
        )

    async def create_analysis_task(self, request: AnalysisTaskRequest) -> AnalysisTaskResponse:
        """
        Create a new analysis task.
        """
        logger.info(f"Creating {request.service_type} analysis task for symbol {request.symbol}")

        # Create a new task
        task_id = str(uuid.uuid4())
        estimated_completion_time = datetime.utcnow() + timedelta(minutes=2)  # Estimate 2 minutes for completion

        # Create task in repository
        await self.task_repository.create_task(
            task_id=task_id,
            service_type=request.service_type,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            parameters=request.parameters
        )

        # Start the task asynchronously using asyncio.create_task
        asyncio.create_task(self._execute_analysis_task(
            task_id=task_id,
            service_type=request.service_type,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            parameters=request.parameters
        ))

        return AnalysisTaskResponse(
            task_id=task_id,
            service_type=request.service_type,
            status=AnalysisTaskStatusEnum.PENDING,
            created_at=datetime.utcnow(),
            estimated_completion_time=estimated_completion_time
        )

    async def get_task_result(self, task_id: str) -> Optional[AnalysisTaskResult]:
        """
        Get the result of a previously created analysis task.
        """
        logger.info(f"Getting result for task {task_id}")

        # Get task from repository
        task = await self.task_repository.get_task(task_id)
        if not task:
            return None

        return task

    async def get_task_status(self, task_id: str) -> Optional[AnalysisTaskStatus]:
        """
        Get the status of a previously created analysis task.
        """
        logger.info(f"Getting status for task {task_id}")

        # Get task from repository
        task = await self.task_repository.get_task_status(task_id)
        if not task:
            return None

        return task

    async def delete_task(self, task_id: str) -> bool:
        """
        Delete a previously created analysis task.
        """
        logger.info(f"Deleting task {task_id}")

        # Delete task from repository
        return await self.task_repository.delete_task(task_id)

    async def list_tasks(
        self,
        limit: int = 10,
        offset: int = 0,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all analysis tasks with optional filtering.
        """
        logger.info(f"Listing tasks with limit {limit}, offset {offset}, status {status}")

        # Get tasks from repository
        tasks = await self.task_repository.list_tasks(limit, offset, status)

        return tasks

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running analysis task.
        """
        logger.info(f"Cancelling task {task_id}")

        # Get task from repository
        task = await self.task_repository.get_task_status(task_id)
        if not task:
            return False

        # Check if task can be cancelled
        if task.status not in [AnalysisTaskStatusEnum.PENDING, AnalysisTaskStatusEnum.RUNNING]:
            return False

        # Cancel task based on service type
        # In a real implementation, this would involve calling the appropriate service
        # For simplicity, we'll just update the task status

        # Update task status in repository
        await self.task_repository.update_task_status(
            task_id=task_id,
            status=AnalysisTaskStatusEnum.CANCELLED,
            message="Task cancelled by user"
        )

        return True

    async def get_available_services(self) -> Dict[str, List[str]]:
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
            # Update task status to running
            await self.task_repository.update_task_status(
                task_id=task_id,
                status=AnalysisTaskStatusEnum.RUNNING,
                progress=0.1,
                message=f"Starting {service_type} analysis for {symbol}"
            )

            # Execute task based on service type
            result = None
            if service_type == AnalysisServiceType.MARKET_ANALYSIS:
                # Update progress
                await self.task_repository.update_task_status(
                    task_id=task_id,
                    progress=0.3,
                    message=f"Running market analysis for {symbol}"
                )

                # Call market analysis service
                result = await self.market_analysis_adapter.analyze_market(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    parameters=parameters
                )

            elif service_type == AnalysisServiceType.CAUSAL_ANALYSIS:
                # Update progress
                await self.task_repository.update_task_status(
                    task_id=task_id,
                    progress=0.3,
                    message=f"Running causal analysis for {symbol}"
                )

                # Call causal analysis service
                result = await self.causal_analysis_adapter.generate_causal_graph(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    parameters=parameters
                )

            elif service_type == AnalysisServiceType.BACKTESTING:
                # Update progress
                await self.task_repository.update_task_status(
                    task_id=task_id,
                    progress=0.3,
                    message=f"Running backtesting for {symbol}"
                )

                # Call backtesting service
                result = await self.backtesting_adapter.run_backtest(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    parameters=parameters
                )

            # Update task status to completed
            await self.task_repository.update_task_status(
                task_id=task_id,
                status=AnalysisTaskStatusEnum.COMPLETED,
                progress=1.0,
                message=f"Completed {service_type} analysis for {symbol}",
                result=result
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
            # Update task status to running
            await self.task_repository.update_task_status(
                task_id=task_id,
                status=AnalysisTaskStatusEnum.RUNNING,
                progress=0.1,
                message=f"Starting integrated analysis for {symbol}"
            )

            # Get subtasks
            task = await self.task_repository.get_task_status(task_id)
            if not task:
                logger.error(f"Task {task_id} not found")
                return

            # Execute each subtask
            results = {}
            total_services = len(services)

            for i, service in enumerate(services):
                # Update progress
                progress = 0.1 + (0.8 * (i / total_services))
                await self.task_repository.update_task_status(
                    task_id=task_id,
                    progress=progress,
                    message=f"Running {service} analysis for {symbol} ({i+1}/{total_services})"
                )

                # Get service parameters
                service_params = parameters.get(service, {})

                # Create a subtask for this service
                subtask_request = AnalysisTaskRequest(
                    service_type=service,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    parameters=service_params
                )

                # Execute the subtask
                subtask_response = await self.create_analysis_task(subtask_request)

                # Wait for the subtask to complete
                subtask_result = None
                max_wait_time = 300  # 5 minutes
                wait_interval = 2  # 2 seconds
                total_wait_time = 0

                while total_wait_time < max_wait_time:
                    # Get subtask status
                    subtask_status = await self.get_task_status(subtask_response.task_id)
                    if not subtask_status:
                        break

                    if subtask_status.status == AnalysisTaskStatusEnum.COMPLETED:
                        # Get subtask result
                        subtask_result = await self.get_task_result(subtask_response.task_id)
                        if subtask_result and subtask_result.result:
                            results[service] = subtask_result.result
                        break

                    if subtask_status.status == AnalysisTaskStatusEnum.FAILED:
                        logger.error(f"Subtask {subtask_response.task_id} failed: {subtask_status.message}")
                        break

                    # Wait for a bit
                    await asyncio.sleep(wait_interval)
                    total_wait_time += wait_interval

            # Integrate results
            integrated_result = self._integrate_analysis_results(results)

            # Update task status to completed
            await self.task_repository.update_task_status(
                task_id=task_id,
                status=AnalysisTaskStatusEnum.COMPLETED,
                progress=1.0,
                message=f"Completed integrated analysis for {symbol}",
                result=integrated_result
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

        Args:
            results: Dictionary of results from different services

        Returns:
            Integrated analysis results
        """
        # In a real implementation, this would involve more sophisticated integration logic
        # For simplicity, we'll just combine the results

        integrated_result = {
            "integrated_timestamp": datetime.utcnow().isoformat(),
            "services_used": list(results.keys()),
            "results": results
        }

        # Add some basic insights based on the combined results
        insights = []

        # Check if we have market analysis results
        if AnalysisServiceType.MARKET_ANALYSIS in results:
            market_result = results[AnalysisServiceType.MARKET_ANALYSIS]

            # Check for patterns
            if "patterns" in market_result and market_result["patterns"]:
                for pattern in market_result["patterns"]:
                    insights.append(f"Detected {pattern['type']} pattern with {pattern['confidence']*100:.1f}% confidence")

            # Check for market regime
            if "market_regime" in market_result:
                regime = market_result["market_regime"]
                insights.append(f"Market is in a {regime['type']} regime with {regime['strength']*100:.1f}% strength")

        # Check if we have causal analysis results
        if AnalysisServiceType.CAUSAL_ANALYSIS in results:
            causal_result = results[AnalysisServiceType.CAUSAL_ANALYSIS]

            # Check for causal relationships
            if "relationships" in causal_result and causal_result["relationships"]:
                for rel in causal_result["relationships"][:3]:  # Just take the top 3
                    insights.append(f"Found causal relationship: {rel['cause']} -> {rel['effect']} (strength: {rel['strength']*100:.1f}%)")

        # Check if we have backtesting results
        if AnalysisServiceType.BACKTESTING in results:
            backtest_result = results[AnalysisServiceType.BACKTESTING]

            # Check for performance metrics
            if "performance" in backtest_result:
                perf = backtest_result["performance"]
                if "profit_factor" in perf:
                    insights.append(f"Strategy has profit factor of {perf['profit_factor']:.2f}")
                if "sharpe_ratio" in perf:
                    insights.append(f"Strategy has Sharpe ratio of {perf['sharpe_ratio']:.2f}")

        integrated_result["insights"] = insights

        return integrated_result