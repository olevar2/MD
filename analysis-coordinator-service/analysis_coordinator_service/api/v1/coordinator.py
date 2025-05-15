"""
Coordinator API routes.

This module provides the API routes for the Analysis Coordinator Service.
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, UTC

from common_lib.cqrs.commands import CommandBus
from common_lib.cqrs.queries import QueryBus
from analysis_coordinator_service.utils.dependency_injection import get_command_bus, get_query_bus, get_coordinator_service
from analysis_coordinator_service.cqrs.commands import (
    RunIntegratedAnalysisCommand,
    CreateAnalysisTaskCommand,
    CancelAnalysisTaskCommand,
    DeleteAnalysisTaskCommand
)
from analysis_coordinator_service.cqrs.queries import (
    GetAnalysisTaskQuery,
    ListAnalysisTasksQuery,
    GetIntegratedAnalysisTaskQuery,
    ListIntegratedAnalysisTasksQuery,
    GetAnalysisTaskStatusQuery,
    GetAvailableServicesQuery
)
from analysis_coordinator_service.models.coordinator_models import (
    IntegratedAnalysisRequest,
    IntegratedAnalysisResponse,
    AnalysisTaskRequest,
    AnalysisTaskResponse,
    AnalysisTaskStatus,
    AnalysisTaskResult,
    AnalysisTaskStatusEnum,
    AnalysisServiceType
)
from analysis_coordinator_service.services.coordinator_service import CoordinatorService

router = APIRouter(prefix="/coordinator", tags=["analysis-coordinator"])


@router.post("/integrated-analysis", response_model=IntegratedAnalysisResponse)
async def run_integrated_analysis(
    request: IntegratedAnalysisRequest,
    coordinator_service: CoordinatorService = Depends(get_coordinator_service)
):
    """
    Run an integrated analysis across multiple analysis services.
    """
    try:
        # Run integrated analysis
        result = await coordinator_service.run_integrated_analysis(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            services=request.services,
            parameters=request.parameters
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run integrated analysis: {str(e)}")


@router.post("/tasks", response_model=AnalysisTaskResponse)
async def create_analysis_task(
    request: AnalysisTaskRequest,
    coordinator_service: CoordinatorService = Depends(get_coordinator_service)
):
    """
    Create a new analysis task.
    """
    try:
        # Create analysis task
        result = await coordinator_service.create_analysis_task(
            service_type=request.service_type,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            parameters=request.parameters
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create analysis task: {str(e)}")


@router.get("/tasks/{task_id}", response_model=AnalysisTaskResult)
async def get_task_result(
    task_id: str,
    coordinator_service: CoordinatorService = Depends(get_coordinator_service)
):
    """
    Get the result of an analysis task.
    """
    try:
        # Get task result
        result = await coordinator_service.get_task_result(task_id)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get task result: {str(e)}")


@router.get("/tasks/{task_id}/status", response_model=AnalysisTaskStatus)
async def get_task_status(
    task_id: str,
    coordinator_service: CoordinatorService = Depends(get_coordinator_service)
):
    """
    Get the status of an analysis task.
    """
    try:
        # Get task status
        status = await coordinator_service.get_task_status(task_id)
        
        if not status:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        return status
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")


@router.get("/tasks", response_model=List[AnalysisTaskResult])
async def list_tasks(
    service_type: Optional[str] = None,
    symbol: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
    coordinator_service: CoordinatorService = Depends(get_coordinator_service)
):
    """
    List analysis tasks.
    """
    try:
        # List tasks
        tasks = await coordinator_service.list_tasks(
            service_type=service_type,
            symbol=symbol,
            status=status,
            limit=limit,
            offset=offset
        )
        
        return tasks
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(
    task_id: str,
    coordinator_service: CoordinatorService = Depends(get_coordinator_service)
):
    """
    Cancel a running task.
    """
    try:
        # Cancel task
        cancelled = await coordinator_service.cancel_task(task_id)
        
        if not cancelled:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found or could not be cancelled")
        
        return {"success": True, "message": f"Task {task_id} cancelled successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")


@router.delete("/tasks/{task_id}")
async def delete_task(
    task_id: str,
    coordinator_service: CoordinatorService = Depends(get_coordinator_service)
):
    """
    Delete a task.
    """
    try:
        # Delete task
        deleted = await coordinator_service.delete_task(task_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        return {"success": True, "message": f"Task {task_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete task: {str(e)}")


@router.get("/available-services")
async def get_available_services(
    coordinator_service: CoordinatorService = Depends(get_coordinator_service)
):
    """
    Get available analysis services.
    """
    try:
        # Get available services
        services = await coordinator_service.get_available_services()
        
        return services
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get available services: {str(e)}")