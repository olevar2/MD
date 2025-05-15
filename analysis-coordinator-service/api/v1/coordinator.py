from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Any, Optional

from analysis_coordinator_service.core.service_dependencies import get_coordinator_service
from analysis_coordinator_service.services.coordinator_service import CoordinatorService
from analysis_coordinator_service.models.coordinator_models import (
    IntegratedAnalysisRequest,
    IntegratedAnalysisResponse,
    AnalysisTaskRequest,
    AnalysisTaskResponse,
    AnalysisTaskStatus,
    AnalysisTaskResult
)

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
        return await coordinator_service.run_integrated_analysis(request)
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
        return await coordinator_service.create_analysis_task(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create analysis task: {str(e)}")

@router.get("/tasks/{task_id}", response_model=AnalysisTaskResult)
async def get_task_result(
    task_id: str,
    coordinator_service: CoordinatorService = Depends(get_coordinator_service)
):
    """
    Get the result of a previously created analysis task.
    """
    try:
        result = await coordinator_service.get_task_result(task_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found")
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
    Get the status of a previously created analysis task.
    """
    try:
        status = await coordinator_service.get_task_status(task_id)
        if not status:
            raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found")
        return status
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")

@router.delete("/tasks/{task_id}", response_model=Dict[str, bool])
async def delete_task(
    task_id: str,
    coordinator_service: CoordinatorService = Depends(get_coordinator_service)
):
    """
    Delete a previously created analysis task.
    """
    try:
        success = await coordinator_service.delete_task(task_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete task: {str(e)}")

@router.get("/tasks", response_model=List[Dict[str, Any]])
async def list_tasks(
    limit: int = 10,
    offset: int = 0,
    status: Optional[str] = None,
    coordinator_service: CoordinatorService = Depends(get_coordinator_service)
):
    """
    List all analysis tasks with optional filtering.
    """
    try:
        return await coordinator_service.list_tasks(limit, offset, status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")

@router.post("/tasks/{task_id}/cancel", response_model=Dict[str, bool])
async def cancel_task(
    task_id: str,
    coordinator_service: CoordinatorService = Depends(get_coordinator_service)
):
    """
    Cancel a running analysis task.
    """
    try:
        success = await coordinator_service.cancel_task(task_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found or already completed")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")

@router.get("/available-services", response_model=Dict[str, List[str]])
async def get_available_services(
    coordinator_service: CoordinatorService = Depends(get_coordinator_service)
):
    """
    Get available analysis services and their capabilities.
    """
    try:
        return await coordinator_service.get_available_services()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get available services: {str(e)}")