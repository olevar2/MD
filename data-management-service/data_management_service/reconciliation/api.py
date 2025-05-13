"""
Data Reconciliation API.

This module provides the API endpoints for the Data Reconciliation system.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from data_management_service.reconciliation.models import (
    ReconciliationConfig,
    ReconciliationTask,
    ReconciliationResult,
    ReconciliationIssue,
    ReconciliationStatus,
    ReconciliationSeverity,
    ReconciliationType,
    ReconciliationRule,
    DataSourceConfig,
    ReconciliationQuery
)
from data_management_service.reconciliation.repository import ReconciliationRepository
from data_management_service.reconciliation.service import ReconciliationService
from data_management_service.historical.service import HistoricalDataService
from data_management_service.historical.repository import HistoricalDataRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reconciliation", tags=["reconciliation"])


# Dependency for database connection
async def get_db_engine() -> AsyncEngine:
    """Get database engine."""
    # Get database connection parameters from environment variables
    import os
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    # Get database connection parameters
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "postgres")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "forex_platform")

    # Create connection string
    connection_string = f"postgresql+asyncpg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    return create_async_engine(connection_string)


# Dependency for repositories
async def get_repositories(engine: AsyncEngine = Depends(get_db_engine)) -> tuple:
    """Get repositories."""
    reconciliation_repository = ReconciliationRepository(engine)
    await reconciliation_repository.initialize()

    historical_repository = HistoricalDataRepository(engine)
    await historical_repository.initialize()

    return reconciliation_repository, historical_repository


# Dependency for services
async def get_services(
    repositories: tuple = Depends(get_repositories)
) -> tuple:
    """Get services."""
    reconciliation_repository, historical_repository = repositories

    historical_service = HistoricalDataService(historical_repository)
    await historical_service.initialize()

    reconciliation_service = ReconciliationService(reconciliation_repository, historical_service)
    await reconciliation_service.initialize()

    return reconciliation_service, historical_service


# Dependency for reconciliation service
async def get_reconciliation_service(
    services: tuple = Depends(get_services)
) -> ReconciliationService:
    """Get reconciliation service."""
    reconciliation_service, _ = services
    return reconciliation_service


# Request/Response models
class CreateConfigRequest(BaseModel):
    """Request model for creating a reconciliation configuration."""
    name: str
    reconciliation_type: ReconciliationType
    primary_source: DataSourceConfig
    secondary_source: Optional[DataSourceConfig] = None
    rules: List[ReconciliationRule] = []
    description: Optional[str] = None
    schedule: Optional[str] = None
    enabled: bool = True
    metadata: Optional[Dict[str, Any]] = None


class ConfigResponse(BaseModel):
    """Response model for configuration ID."""
    config_id: str


class ScheduleTaskRequest(BaseModel):
    """Request model for scheduling a reconciliation task."""
    config_id: str
    scheduled_time: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class TaskResponse(BaseModel):
    """Response model for task ID."""
    task_id: str


class ResultResponse(BaseModel):
    """Response model for result ID."""
    result_id: str


# API endpoints
@router.post("/configs", response_model=ConfigResponse)
async def create_config(
    request: CreateConfigRequest,
    service: ReconciliationService = Depends(get_reconciliation_service)
) -> ConfigResponse:
    """Create a reconciliation configuration."""
    try:
        config_id = await service.create_config(
            name=request.name,
            reconciliation_type=request.reconciliation_type,
            primary_source=request.primary_source,
            secondary_source=request.secondary_source,
            rules=request.rules,
            description=request.description,
            schedule=request.schedule,
            enabled=request.enabled,
            metadata=request.metadata
        )

        return ConfigResponse(config_id=config_id)
    except Exception as e:
        logger.error(f"Failed to create configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks", response_model=TaskResponse)
async def schedule_task(
    request: ScheduleTaskRequest,
    service: ReconciliationService = Depends(get_reconciliation_service)
) -> TaskResponse:
    """Schedule a reconciliation task."""
    try:
        task_id = await service.schedule_task(
            config_id=request.config_id,
            scheduled_time=request.scheduled_time,
            metadata=request.metadata
        )

        return TaskResponse(task_id=task_id)
    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to schedule task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks/{task_id}/run", response_model=ResultResponse)
async def run_task(
    task_id: str,
    service: ReconciliationService = Depends(get_reconciliation_service)
) -> ResultResponse:
    """Run a reconciliation task."""
    try:
        result_id = await service.run_task(task_id=task_id)

        return ResultResponse(result_id=result_id)
    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to run task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/configs/{config_id}", response_model=ReconciliationConfig)
async def get_config(
    config_id: str,
    service: ReconciliationService = Depends(get_reconciliation_service)
) -> ReconciliationConfig:
    """Get a reconciliation configuration."""
    try:
        config = await service.get_config(config_id=config_id)

        if config is None:
            raise HTTPException(status_code=404, detail=f"Configuration not found: {config_id}")

        return config
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}", response_model=ReconciliationTask)
async def get_task(
    task_id: str,
    service: ReconciliationService = Depends(get_reconciliation_service)
) -> ReconciliationTask:
    """Get a reconciliation task."""
    try:
        task = await service.get_task(task_id=task_id)

        if task is None:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

        return task
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{result_id}", response_model=ReconciliationResult)
async def get_result(
    result_id: str,
    service: ReconciliationService = Depends(get_reconciliation_service)
) -> ReconciliationResult:
    """Get a reconciliation result."""
    try:
        result = await service.get_result(result_id=result_id)

        if result is None:
            raise HTTPException(status_code=404, detail=f"Result not found: {result_id}")

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get result: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/configs", response_model=List[ReconciliationConfig])
async def get_configs(
    enabled: Optional[bool] = Query(None, description="Filter by enabled status"),
    reconciliation_type: Optional[ReconciliationType] = Query(None, description="Filter by reconciliation type"),
    limit: int = Query(100, description="Maximum number of records to return"),
    offset: int = Query(0, description="Offset for pagination"),
    service: ReconciliationService = Depends(get_reconciliation_service)
) -> List[ReconciliationConfig]:
    """Get reconciliation configurations."""
    try:
        configs = await service.get_configs(
            enabled=enabled,
            reconciliation_type=reconciliation_type,
            limit=limit,
            offset=offset
        )

        return configs
    except Exception as e:
        logger.error(f"Failed to get configurations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks", response_model=List[ReconciliationTask])
async def get_tasks(
    config_id: Optional[str] = Query(None, description="Filter by config ID"),
    status: Optional[ReconciliationStatus] = Query(None, description="Filter by status"),
    start_date: Optional[datetime] = Query(None, description="Filter by scheduled time (start)"),
    end_date: Optional[datetime] = Query(None, description="Filter by scheduled time (end)"),
    limit: int = Query(100, description="Maximum number of records to return"),
    offset: int = Query(0, description="Offset for pagination"),
    service: ReconciliationService = Depends(get_reconciliation_service)
) -> List[ReconciliationTask]:
    """Get reconciliation tasks."""
    try:
        tasks = await service.get_tasks(
            config_id=config_id,
            status=status,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset
        )

        return tasks
    except Exception as e:
        logger.error(f"Failed to get tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results", response_model=List[ReconciliationResult])
async def get_results(
    config_id: Optional[str] = Query(None, description="Filter by config ID"),
    status: Optional[ReconciliationStatus] = Query(None, description="Filter by status"),
    start_date: Optional[datetime] = Query(None, description="Filter by start time (start)"),
    end_date: Optional[datetime] = Query(None, description="Filter by start time (end)"),
    limit: int = Query(100, description="Maximum number of records to return"),
    offset: int = Query(0, description="Offset for pagination"),
    service: ReconciliationService = Depends(get_reconciliation_service)
) -> List[ReconciliationResult]:
    """Get reconciliation results."""
    try:
        results = await service.get_results(
            config_id=config_id,
            status=status,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset
        )

        return results
    except Exception as e:
        logger.error(f"Failed to get results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/configs/{config_id}/disable", response_model=bool)
async def disable_config(
    config_id: str,
    service: ReconciliationService = Depends(get_reconciliation_service)
) -> bool:
    """Disable a reconciliation configuration."""
    try:
        # Get the configuration
        config = await service.get_config(config_id=config_id)

        if config is None:
            raise HTTPException(status_code=404, detail=f"Configuration not found: {config_id}")

        # Update the configuration
        config_dict = config.dict()
        config_dict["enabled"] = False

        # Create a new configuration with the updated values
        updated_config = ReconciliationConfig(**config_dict)

        # Store the updated configuration
        await service.repository.store_config(updated_config)

        return True
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to disable configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/configs/{config_id}/enable", response_model=bool)
async def enable_config(
    config_id: str,
    service: ReconciliationService = Depends(get_reconciliation_service)
) -> bool:
    """Enable a reconciliation configuration."""
    try:
        # Get the configuration
        config = await service.get_config(config_id=config_id)

        if config is None:
            raise HTTPException(status_code=404, detail=f"Configuration not found: {config_id}")

        # Update the configuration
        config_dict = config.dict()
        config_dict["enabled"] = True

        # Create a new configuration with the updated values
        updated_config = ReconciliationConfig(**config_dict)

        # Store the updated configuration
        await service.repository.store_config(updated_config)

        return True
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to enable configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/configs/{config_id}", response_model=bool)
async def delete_config(
    config_id: str,
    service: ReconciliationService = Depends(get_reconciliation_service)
) -> bool:
    """Delete a reconciliation configuration."""
    try:
        # Get the configuration
        config = await service.get_config(config_id=config_id)

        if config is None:
            raise HTTPException(status_code=404, detail=f"Configuration not found: {config_id}")

        # Update the configuration
        config_dict = config.dict()
        config_dict["enabled"] = False

        # Create a new configuration with the updated values
        updated_config = ReconciliationConfig(**config_dict)

        # Store the updated configuration
        await service.repository.store_config(updated_config)

        return True
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))
