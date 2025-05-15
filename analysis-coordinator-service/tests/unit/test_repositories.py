"""
Test repositories for the Analysis Coordinator Service.

This module provides test repositories for the Analysis Coordinator Service.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, UTC
import uuid

from analysis_coordinator_service.models.coordinator_models import (
    AnalysisTaskStatusEnum,
    AnalysisTaskStatus,
    AnalysisTaskResult,
    AnalysisServiceType
)

class TestTaskRepository:
    """
    Test repository for tasks.
    """
    
    def __init__(self):
        """
        Initialize the repository.
        """
        self.tasks = {}
        self.task_statuses = {}
    
    async def create_task(
        self,
        task_id: str,
        service_type: str,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        parameters: Dict[str, Any] = None
    ) -> None:
        """
        Create a new task.
        
        Args:
            task_id: Task ID
            service_type: Type of analysis service
            symbol: Symbol to analyze
            timeframe: Timeframe to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            parameters: Additional parameters for the analysis
        """
        self.tasks[task_id] = AnalysisTaskResult(
            task_id=task_id,
            service_type=service_type,
            status=AnalysisTaskStatusEnum.PENDING,
            created_at=datetime.now(UTC),
            completed_at=None,
            result=None,
            error=None
        )
        
        self.task_statuses[task_id] = AnalysisTaskStatus(
            task_id=task_id,
            service_type=service_type,
            status=AnalysisTaskStatusEnum.PENDING,
            created_at=datetime.now(UTC),
            progress=0.0,
            message="Task created"
        )
    
    async def create_integrated_task(
        self,
        task_id: str,
        services: List[str],
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        parameters: Dict[str, Any] = None
    ) -> None:
        """
        Create a new integrated task.
        
        Args:
            task_id: Task ID
            services: List of services to use for analysis
            symbol: Symbol to analyze
            timeframe: Timeframe to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            parameters: Additional parameters for the analysis
        """
        # Use a valid service type from the enum
        service_type = AnalysisServiceType.MARKET_ANALYSIS
        
        self.tasks[task_id] = AnalysisTaskResult(
            task_id=task_id,
            service_type=service_type,
            status=AnalysisTaskStatusEnum.PENDING,
            created_at=datetime.now(UTC),
            completed_at=None,
            result=None,
            error=None
        )
        
        self.task_statuses[task_id] = AnalysisTaskStatus(
            task_id=task_id,
            service_type=service_type,
            status=AnalysisTaskStatusEnum.PENDING,
            created_at=datetime.now(UTC),
            progress=0.0,
            message="Integrated task created"
        )
    
    async def get_task(self, task_id: str) -> Optional[AnalysisTaskResult]:
        """
        Get a task by ID.
        
        Args:
            task_id: Task ID
            
        Returns:
            The task or None if not found
        """
        return self.tasks.get(task_id)
    
    async def get_task_status(self, task_id: str) -> Optional[AnalysisTaskStatus]:
        """
        Get a task status by ID.
        
        Args:
            task_id: Task ID
            
        Returns:
            The task status or None if not found
        """
        return self.task_statuses.get(task_id)
    
    async def update_task_status(
        self,
        task_id: str,
        status: Optional[AnalysisTaskStatusEnum] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Update a task status.
        
        Args:
            task_id: Task ID
            status: New status
            progress: New progress
            message: New message
            result: New result
            error: New error
        """
        # Update task status
        task_status = self.task_statuses.get(task_id)
        if task_status:
            if status is not None:
                task_status.status = status
            if progress is not None:
                task_status.progress = progress
            if message is not None:
                task_status.message = message
        
        # Update task
        task = self.tasks.get(task_id)
        if task:
            if status is not None:
                task.status = status
            if result is not None:
                task.result = result
            if error is not None:
                task.error = error
            if status == AnalysisTaskStatusEnum.COMPLETED:
                task.completed_at = datetime.now(UTC)
    
    async def list_tasks(
        self,
        limit: int = 10,
        offset: int = 0,
        status: Optional[str] = None
    ) -> List[AnalysisTaskResult]:
        """
        List tasks.
        
        Args:
            limit: Maximum number of tasks to return
            offset: Offset for pagination
            status: Filter by status
            
        Returns:
            List of tasks
        """
        tasks = list(self.tasks.values())
        
        # Filter by status
        if status:
            tasks = [task for task in tasks if task.status == status]
        
        # Apply pagination
        tasks = tasks[offset:offset + limit]
        
        return tasks
    
    async def delete_task(self, task_id: str) -> bool:
        """
        Delete a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if the task was deleted, False otherwise
        """
        if task_id in self.tasks:
            del self.tasks[task_id]
            if task_id in self.task_statuses:
                del self.task_statuses[task_id]
            return True
        return False
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if the task was cancelled, False otherwise
        """
        task_status = self.task_statuses.get(task_id)
        if task_status and task_status.status in [AnalysisTaskStatusEnum.PENDING, AnalysisTaskStatusEnum.RUNNING]:
            task_status.status = AnalysisTaskStatusEnum.CANCELLED
            task_status.message = "Task cancelled by user"
            
            task = self.tasks.get(task_id)
            if task:
                task.status = AnalysisTaskStatusEnum.CANCELLED
            
            return True
        return False
    
    async def get_by_id(self, task_id: str) -> Optional[AnalysisTaskResult]:
        """
        Get a task by ID.
        
        Args:
            task_id: Task ID
            
        Returns:
            The task or None if not found
        """
        return self.tasks.get(task_id)
    
    async def get_by_criteria(self, criteria: Dict[str, Any]) -> List[AnalysisTaskResult]:
        """
        Get tasks by criteria.
        
        Args:
            criteria: Criteria to filter by
            
        Returns:
            List of tasks
        """
        tasks = list(self.tasks.values())
        
        # Filter by criteria
        for key, value in criteria.items():
            tasks = [task for task in tasks if getattr(task, key, None) == value]
        
        return tasks