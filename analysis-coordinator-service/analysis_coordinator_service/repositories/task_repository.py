import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, UTC
import uuid

from analysis_coordinator_service.models.coordinator_models import (
    AnalysisTaskStatusEnum,
    AnalysisTaskStatus,
    AnalysisTaskResult,
    AnalysisServiceType
)

logger = logging.getLogger(__name__)

class TaskRepository:
    """
    Repository for storing and retrieving analysis tasks.

    Note: This is a simplified in-memory implementation.
    In a real application, this would use a database.
    """

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        # In-memory storage for tasks
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.integrated_tasks: Dict[str, Dict[str, Any]] = {}

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

        self.tasks[task_id] = task

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
        """
        if parameters is None:
            parameters = {}

        logger.info(f"Creating integrated task {task_id} for {services} analysis of {symbol}")

        task = {
            "task_id": task_id,
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
            "result": {},
            "error": None,
            "progress": 0.0,
            "message": "Task created",
            "subtasks": {}
        }

        # Create subtasks for each service
        for service in services:
            subtask_id = str(uuid.uuid4())
            service_parameters = parameters.get(service, {})

            await self.create_task(
                task_id=subtask_id,
                service_type=service,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                parameters=service_parameters
            )

            task["subtasks"][service] = subtask_id

        self.integrated_tasks[task_id] = task

    async def get_task(self, task_id: str) -> Optional[AnalysisTaskResult]:
        """
        Get a task by ID.
        """
        logger.info(f"Getting task {task_id}")

        # Check if it's a regular task
        task = self.tasks.get(task_id)
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
        task = self.integrated_tasks.get(task_id)
        if task:
            # For integrated tasks, we need to combine results from all subtasks
            result = {}
            for service, subtask_id in task["subtasks"].items():
                subtask = self.tasks.get(subtask_id)
                if subtask and subtask["result"]:
                    result[service] = subtask["result"]

            return AnalysisTaskResult(
                task_id=task["task_id"],
                service_type=AnalysisServiceType.MARKET_ANALYSIS,  # Placeholder for integrated tasks
                status=task["status"],
                created_at=task["created_at"],
                completed_at=task["completed_at"],
                result=result,
                error=task["error"]
            )

        return None

    async def get_task_status(self, task_id: str) -> Optional[AnalysisTaskStatus]:
        """
        Get the status of a task by ID.
        """
        logger.info(f"Getting status for task {task_id}")

        # Check if it's a regular task
        task = self.tasks.get(task_id)
        if task:
            return AnalysisTaskStatus(
                task_id=task["task_id"],
                service_type=task["service_type"],
                status=task["status"],
                created_at=task["created_at"],
                updated_at=task["updated_at"],
                progress=task["progress"],
                message=task["message"]
            )

        # Check if it's an integrated task
        task = self.integrated_tasks.get(task_id)
        if task:
            # For integrated tasks, we calculate progress as average of subtasks
            progress = 0.0
            completed_subtasks = 0
            for service, subtask_id in task["subtasks"].items():
                subtask = self.tasks.get(subtask_id)
                if subtask:
                    progress += subtask["progress"]
                    if subtask["status"] == AnalysisTaskStatusEnum.COMPLETED:
                        completed_subtasks += 1

            if task["subtasks"]:
                progress /= len(task["subtasks"])

            # Determine overall status
            status = task["status"]
            if completed_subtasks == len(task["subtasks"]):
                status = AnalysisTaskStatusEnum.COMPLETED
            elif any(self.tasks.get(subtask_id, {}).get("status") == AnalysisTaskStatusEnum.FAILED
                    for subtask_id in task["subtasks"].values()):
                status = AnalysisTaskStatusEnum.FAILED
            elif any(self.tasks.get(subtask_id, {}).get("status") == AnalysisTaskStatusEnum.RUNNING
                    for subtask_id in task["subtasks"].values()):
                status = AnalysisTaskStatusEnum.RUNNING

            return AnalysisTaskStatus(
                task_id=task["task_id"],
                service_type=AnalysisServiceType.MARKET_ANALYSIS,  # Placeholder for integrated tasks
                status=status,
                created_at=task["created_at"],
                updated_at=task["updated_at"],
                progress=progress,
                message=task["message"]
            )

        return None

    async def update_task_status(
        self,
        task_id: str,
        status: AnalysisTaskStatusEnum,
        progress: float = None,
        message: str = None,
        result: Dict[str, Any] = None,
        error: str = None
    ) -> bool:
        """
        Update the status of a task.
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

            if status == AnalysisTaskStatusEnum.COMPLETED or status == AnalysisTaskStatusEnum.FAILED:
                task["completed_at"] = datetime.now(UTC)

            return True

        # Check if it's an integrated task
        task = self.integrated_tasks.get(task_id)
        if task:
            task["status"] = status
            task["updated_at"] = datetime.now(UTC)

            if progress is not None:
                task["progress"] = progress

            if message is not None:
                task["message"] = message

            if error is not None:
                task["error"] = error

            if status == AnalysisTaskStatusEnum.COMPLETED or status == AnalysisTaskStatusEnum.FAILED:
                task["completed_at"] = datetime.now(UTC)

            return True

        return False

    async def delete_task(self, task_id: str) -> bool:
        """
        Delete a task by ID.
        """
        logger.info(f"Deleting task {task_id}")

        # Check if it's a regular task
        if task_id in self.tasks:
            del self.tasks[task_id]
            return True

        # Check if it's an integrated task
        if task_id in self.integrated_tasks:
            # Delete all subtasks
            for subtask_id in self.integrated_tasks[task_id]["subtasks"].values():
                if subtask_id in self.tasks:
                    del self.tasks[subtask_id]

            del self.integrated_tasks[task_id]
            return True

        return False

    async def list_tasks(
        self,
        limit: int = 10,
        offset: int = 0,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all tasks with optional filtering.
        """
        logger.info(f"Listing tasks with limit {limit}, offset {offset}, status {status}")

        # Combine regular and integrated tasks
        all_tasks = []

        # Add regular tasks
        for task in self.tasks.values():
            if status is None or task["status"] == status:
                all_tasks.append({
                    "task_id": task["task_id"],
                    "service_type": task["service_type"],
                    "symbol": task["symbol"],
                    "timeframe": task["timeframe"],
                    "status": task["status"],
                    "created_at": task["created_at"],
                    "updated_at": task["updated_at"],
                    "progress": task["progress"],
                    "message": task["message"]
                })

        # Add integrated tasks
        for task in self.integrated_tasks.values():
            if status is None or task["status"] == status:
                all_tasks.append({
                    "task_id": task["task_id"],
                    "service_type": "integrated",
                    "services": task["services"],
                    "symbol": task["symbol"],
                    "timeframe": task["timeframe"],
                    "status": task["status"],
                    "created_at": task["created_at"],
                    "updated_at": task["updated_at"],
                    "progress": task["progress"],
                    "message": task["message"]
                })

        # Sort by created_at (newest first)
        all_tasks.sort(key=lambda x: x["created_at"], reverse=True)

        # Apply pagination
        paginated_tasks = all_tasks[offset:offset + limit]

        return paginated_tasks