"""
Base workflow class for analysis workflows.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from analysis_coordinator_service.models.coordinator_models import (
    AnalysisTaskResult,
    AnalysisTaskStatusEnum
)
from analysis_coordinator_service.repositories.task_repository import TaskRepository

logger = logging.getLogger(__name__)


class BaseWorkflow(ABC):
    """
    Base class for analysis workflows.
    """

    def __init__(self, task_repository: TaskRepository):
        """
        Initialize the workflow.

        Args:
            task_repository: Task repository for storing and retrieving tasks
        """
        self.task_repository = task_repository

    @abstractmethod
    async def execute(
        self,
        task_id: str,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        parameters: Dict[str, Any] = None
    ) -> AnalysisTaskResult:
        """
        Execute the workflow.

        Args:
            task_id: Task ID
            symbol: Symbol to analyze
            timeframe: Timeframe to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            parameters: Additional parameters for the analysis

        Returns:
            Analysis task result
        """
        pass

    async def update_task_status(
        self,
        task_id: str,
        status: AnalysisTaskStatusEnum = None,
        progress: float = None,
        message: str = None,
        result: Dict[str, Any] = None,
        error: str = None
    ) -> None:
        """
        Update the status of a task.

        Args:
            task_id: Task ID
            status: Task status
            progress: Task progress (0-1)
            message: Status message
            result: Task result
            error: Error message
        """
        await self.task_repository.update_task_status(
            task_id=task_id,
            status=status,
            progress=progress,
            message=message,
            result=result,
            error=error
        )

    def aggregate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate results from multiple analysis steps.

        Args:
            results: Dictionary of results from different steps

        Returns:
            Aggregated results
        """
        # Default implementation just returns the results as is
        return results