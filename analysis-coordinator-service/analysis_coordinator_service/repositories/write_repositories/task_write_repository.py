"""
Task write repository.

This module provides the write repository for analysis tasks.
"""
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, UTC

from common_lib.cqrs.repositories import WriteRepository
from analysis_coordinator_service.models.coordinator_models import (
    AnalysisTaskStatusEnum,
    AnalysisTaskStatus,
    AnalysisTaskResult,
    AnalysisServiceType
)

logger = logging.getLogger(__name__)


class TaskWriteRepository(WriteRepository):
    """
    Write repository for analysis tasks.
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
        self.lock = asyncio.Lock()
    
    async def add(self, entity: Dict[str, Any]) -> str:
        """
        Add a task.
        
        Args:
            entity: Task to add
            
        Returns:
            ID of the added task
        """
        logger.info(f"Adding task {entity.get('task_id')}")
        
        async with self.lock:
            if entity.get("service_type") == "integrated":
                self.integrated_tasks[entity["task_id"]] = entity
            else:
                self.tasks[entity["task_id"]] = entity
        
        return entity["task_id"]
    
    async def update(self, entity: Dict[str, Any]) -> None:
        """
        Update a task.
        
        Args:
            entity: Task to update
        """
        logger.info(f"Updating task {entity.get('task_id')}")
        
        async with self.lock:
            if entity.get("service_type") == "integrated":
                if entity["task_id"] in self.integrated_tasks:
                    self.integrated_tasks[entity["task_id"]] = entity
                else:
                    logger.warning(f"Integrated task {entity['task_id']} not found for update")
            else:
                if entity["task_id"] in self.tasks:
                    self.tasks[entity["task_id"]] = entity
                else:
                    logger.warning(f"Task {entity['task_id']} not found for update")
    
    async def delete(self, id: str) -> None:
        """
        Delete a task by ID.
        
        Args:
            id: ID of the task to delete
        """
        logger.info(f"Deleting task {id}")
        
        async with self.lock:
            if id in self.tasks:
                del self.tasks[id]
            elif id in self.integrated_tasks:
                del self.integrated_tasks[id]
            else:
                logger.warning(f"Task {id} not found for deletion")
    
    async def add_batch(self, entities: List[Dict[str, Any]]) -> List[str]:
        """
        Add multiple tasks in a batch.
        
        Args:
            entities: Tasks to add
            
        Returns:
            IDs of the added tasks
        """
        logger.info(f"Adding {len(entities)} tasks in batch")
        
        ids = []
        for entity in entities:
            id = await self.add(entity)
            ids.append(id)
        
        return ids
    
    async def create_task(
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
        Create a new analysis task.
        
        Args:
            task_id: Task ID
            service_type: Type of analysis service
            symbol: Symbol to analyze
            timeframe: Timeframe for analysis
            start_date: Start date for analysis
            end_date: End date for analysis
            parameters: Parameters for the analysis
        """
        if parameters is None:
            parameters = {}

        logger.info(f"Creating task {task_id} for {service_type} analysis of {symbol}")

        task = {
            "task_id": task_id,
            "service_type": service_type,
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date,
            "parameters": parameters,
            "status": AnalysisTaskStatusEnum.PENDING,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "completed_at": None,
            "result": None,
            "error": None,
            "progress": 0.0,
            "message": "Task created"
        }
        
        await self.add(task)
    
    async def create_integrated_task(
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
        Create a new integrated analysis task.
        
        Args:
            task_id: Task ID
            services: List of services to use for analysis
            symbol: Symbol to analyze
            timeframe: Timeframe for analysis
            start_date: Start date for analysis
            end_date: End date for analysis
            parameters: Parameters for the analysis
        """
        if parameters is None:
            parameters = {}

        logger.info(f"Creating integrated task {task_id} for {symbol} with services {services}")

        task = {
            "task_id": task_id,
            "service_type": "integrated",
            "services": services,
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date,
            "parameters": parameters,
            "status": AnalysisTaskStatusEnum.PENDING,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "completed_at": None,
            "result": None,
            "error": None,
            "progress": 0.0,
            "message": "Task created",
            "service_tasks": {}
        }
        
        await self.add(task)
    
    async def update_task_status(
        self,
        task_id: str,
        status: AnalysisTaskStatusEnum,
        progress: float = None,
        message: str = None,
        result: Dict[str, Any] = None,
        error: str = None
    ) -> None:
        """
        Update the status of a task.
        
        Args:
            task_id: Task ID
            status: New status
            progress: Progress (0.0 to 1.0)
            message: Status message
            result: Task result
            error: Error message
        """
        logger.info(f"Updating status for task {task_id} to {status}")

        # Check if it's a regular task
        task = self.tasks.get(task_id)
        if task:
            task["status"] = status
            task["updated_at"] = datetime.now(UTC)
            
            if progress is not None:
                task["progress"] = progress
            
            if message is not None:
                task["message"] = message
            
            if result is not None:
                task["result"] = result
            
            if error is not None:
                task["error"] = error
            
            if status in [AnalysisTaskStatusEnum.COMPLETED, AnalysisTaskStatusEnum.FAILED, AnalysisTaskStatusEnum.CANCELLED]:
                task["completed_at"] = datetime.now(UTC)
            
            await self.update(task)
            return
        
        # Check if it's an integrated task
        integrated_task = self.integrated_tasks.get(task_id)
        if integrated_task:
            integrated_task["status"] = status
            integrated_task["updated_at"] = datetime.now(UTC)
            
            if progress is not None:
                integrated_task["progress"] = progress
            
            if message is not None:
                integrated_task["message"] = message
            
            if result is not None:
                integrated_task["result"] = result
            
            if error is not None:
                integrated_task["error"] = error
            
            if status in [AnalysisTaskStatusEnum.COMPLETED, AnalysisTaskStatusEnum.FAILED, AnalysisTaskStatusEnum.CANCELLED]:
                integrated_task["completed_at"] = datetime.now(UTC)
            
            await self.update(integrated_task)
            return
        
        logger.warning(f"Task {task_id} not found for status update")
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if the task was cancelled, False otherwise
        """
        logger.info(f"Cancelling task {task_id}")

        # Check if it's a regular task
        task = self.tasks.get(task_id)
        if task:
            if task["status"] in [AnalysisTaskStatusEnum.PENDING, AnalysisTaskStatusEnum.RUNNING]:
                await self.update_task_status(
                    task_id=task_id,
                    status=AnalysisTaskStatusEnum.CANCELLED,
                    message="Task cancelled by user"
                )
                return True
            return False
        
        # Check if it's an integrated task
        integrated_task = self.integrated_tasks.get(task_id)
        if integrated_task:
            if integrated_task["status"] in [AnalysisTaskStatusEnum.PENDING, AnalysisTaskStatusEnum.RUNNING]:
                await self.update_task_status(
                    task_id=task_id,
                    status=AnalysisTaskStatusEnum.CANCELLED,
                    message="Task cancelled by user"
                )
                return True
            return False
        
        logger.warning(f"Task {task_id} not found for cancellation")
        return False