"""
Task read repository.

This module provides the read repository for analysis tasks.
"""
import logging
from analysis_coordinator_service.utils.cache_factory import cache_factory
from common_lib.caching.decorators import cached
from typing import Dict, List, Any, Optional
from datetime import datetime

from common_lib.cqrs.repositories import ReadRepository
from analysis_coordinator_service.models.coordinator_models import (
    AnalysisTaskStatusEnum,
    AnalysisTaskStatus,
    AnalysisTaskResult,
    AnalysisServiceType
)

logger = logging.getLogger(__name__)


class TaskReadRepository(ReadRepository):
    """
    Read repository for analysis tasks.
    """
    
    def __init__(self, connection_string: str):
        """
        Initialize the repository with a connection string.
        
        Args:
            connection_string: Database connection string
        """
        self.connection_string = connection_string
        # In-memory storage for tasks (in a real implementation, this would use a database)
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.integrated_tasks: Dict[str, Dict[str, Any]] = {}
        self.cache = cache_factory.get_cache()
    
    @cached(cache_factory.get_cache(), "task", ttl=3600)
    async def get_by_id(self, id: str) -> Optional[AnalysisTaskResult]:
        """
        Get a task by ID.
        
        Args:
            id: Task ID
            
        Returns:
            The task or None if not found
        """
        logger.info(f"Getting task {id}")
        
        # Check if it's a regular task
        task = self.tasks.get(id)
        if task:
            return AnalysisTaskResult(
                task_id=task["task_id"],
                service_type=task["service_type"],
                status=task["status"],
                created_at=task["created_at"],
                completed_at=task["completed_at"],
                result=task["result"],
                error=task["error"]
            )
        
        # Check if it's an integrated task
        integrated_task = self.integrated_tasks.get(id)
        if integrated_task:
            return AnalysisTaskResult(
                task_id=integrated_task["task_id"],
                service_type="integrated",
                status=integrated_task["status"],
                created_at=integrated_task["created_at"],
                completed_at=integrated_task["completed_at"],
                result=integrated_task["result"],
                error=integrated_task["error"]
            )
        
        logger.warning(f"Task {id} not found")
        return None
    
    async def get_all(self) -> List[AnalysisTaskResult]:
        """
        Get all tasks.
        
        Returns:
            List of all tasks
        """
        logger.info("Getting all tasks")
        
        tasks = []
        
        # Get regular tasks
        for task_id, task in self.tasks.items():
            tasks.append(AnalysisTaskResult(
                task_id=task["task_id"],
                service_type=task["service_type"],
                status=task["status"],
                created_at=task["created_at"],
                completed_at=task["completed_at"],
                result=task["result"],
                error=task["error"]
            ))
        
        # Get integrated tasks
        for task_id, task in self.integrated_tasks.items():
            tasks.append(AnalysisTaskResult(
                task_id=task["task_id"],
                service_type="integrated",
                status=task["status"],
                created_at=task["created_at"],
                completed_at=task["completed_at"],
                result=task["result"],
                error=task["error"]
            ))
        
        return tasks
    
    async def get_by_criteria(self, criteria: Dict[str, Any]) -> List[AnalysisTaskResult]:
        """
        Get tasks by criteria.
        
        Args:
            criteria: Criteria to filter by
            
        Returns:
            List of tasks matching the criteria
        """
        logger.info(f"Getting tasks with criteria: {criteria}")
        
        # Get all tasks
        all_tasks = await self.get_all()
        
        # Filter by criteria
        filtered_tasks = []
        for task in all_tasks:
            match = True
            for key, value in criteria.items():
                if hasattr(task, key):
                    if getattr(task, key) != value:
                        match = False
                        break
                else:
                    match = False
                    break
            
            if match:
                filtered_tasks.append(task)
        
        return filtered_tasks
    
    async def get_task_status(self, task_id: str) -> Optional[AnalysisTaskStatus]:
        """
        Get the status of a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            The task status or None if not found
        """
        logger.info(f"Getting status for task {task_id}")
        
        # Check if it's a regular task
        task = self.tasks.get(task_id)
        if task:
            return AnalysisTaskStatus(
                task_id=task["task_id"],
                status=task["status"],
                progress=task["progress"],
                message=task["message"],
                updated_at=task["updated_at"]
            )
        
        # Check if it's an integrated task
        integrated_task = self.integrated_tasks.get(task_id)
        if integrated_task:
            return AnalysisTaskStatus(
                task_id=integrated_task["task_id"],
                status=integrated_task["status"],
                progress=integrated_task["progress"],
                message=integrated_task["message"],
                updated_at=integrated_task["updated_at"]
            )
        
        logger.warning(f"Task {task_id} not found")
        return None