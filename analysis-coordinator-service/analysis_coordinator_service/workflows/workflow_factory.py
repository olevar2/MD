"""
Workflow factory for creating analysis workflows.
"""
from typing import Dict, Any, Optional

from analysis_coordinator_service.workflows.base_workflow import BaseWorkflow
from analysis_coordinator_service.workflows.market_analysis_workflow import MarketAnalysisWorkflow
from analysis_coordinator_service.workflows.causal_analysis_workflow import CausalAnalysisWorkflow
from analysis_coordinator_service.workflows.backtesting_workflow import BacktestingWorkflow
from analysis_coordinator_service.workflows.integrated_analysis_workflow import IntegratedAnalysisWorkflow
from analysis_coordinator_service.models.coordinator_models import AnalysisServiceType
from analysis_coordinator_service.repositories.task_repository import TaskRepository
from analysis_coordinator_service.adapters.market_analysis_adapter import MarketAnalysisAdapter
from analysis_coordinator_service.adapters.causal_analysis_adapter import CausalAnalysisAdapter
from analysis_coordinator_service.adapters.backtesting_adapter import BacktestingAdapter


class WorkflowFactory:
    """
    Factory for creating analysis workflows.
    """

    def __init__(
        self,
        task_repository: TaskRepository,
        market_analysis_adapter: MarketAnalysisAdapter,
        causal_analysis_adapter: CausalAnalysisAdapter,
        backtesting_adapter: BacktestingAdapter
    ):
        """
        Initialize the workflow factory.

        Args:
            task_repository: Task repository for storing and retrieving tasks
            market_analysis_adapter: Adapter for the market analysis service
            causal_analysis_adapter: Adapter for the causal analysis service
            backtesting_adapter: Adapter for the backtesting service
        """
        self.task_repository = task_repository
        self.market_analysis_adapter = market_analysis_adapter
        self.causal_analysis_adapter = causal_analysis_adapter
        self.backtesting_adapter = backtesting_adapter

    def create_workflow(self, service_type: str) -> BaseWorkflow:
        """
        Create a workflow for the specified service type.

        Args:
            service_type: Type of analysis service

        Returns:
            Analysis workflow
        """
        if service_type == AnalysisServiceType.MARKET_ANALYSIS:
            return MarketAnalysisWorkflow(
                task_repository=self.task_repository,
                market_analysis_adapter=self.market_analysis_adapter
            )
        elif service_type == AnalysisServiceType.CAUSAL_ANALYSIS:
            return CausalAnalysisWorkflow(
                task_repository=self.task_repository,
                causal_analysis_adapter=self.causal_analysis_adapter
            )
        elif service_type == AnalysisServiceType.BACKTESTING:
            return BacktestingWorkflow(
                task_repository=self.task_repository,
                backtesting_adapter=self.backtesting_adapter
            )
        elif service_type == "integrated_analysis":
            return IntegratedAnalysisWorkflow(
                task_repository=self.task_repository,
                market_analysis_adapter=self.market_analysis_adapter,
                causal_analysis_adapter=self.causal_analysis_adapter,
                backtesting_adapter=self.backtesting_adapter
            )
        else:
            raise ValueError(f"Unknown service type: {service_type}")