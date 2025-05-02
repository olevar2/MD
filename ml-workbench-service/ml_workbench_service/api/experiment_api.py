"""
Experiment API Module.

Provides API endpoints for managing machine learning experiments.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Depends, Query, Body, UploadFile, File, Path, BackgroundTasks
from fastapi.responses import FileResponse

from ml_workbench_service.models.experiment_models import (
    Experiment, 
    ExperimentCreate,
    ExperimentUpdate,
    ModelVersionCreate,
    MetricLog,
    ExperimentStatus,
    ModelType
)
from ml_workbench_service.services.experiment_service import ExperimentService
from ml_workbench_service.repositories.experiment_repository import ExperimentRepository
from ml_workbench_service.clients.feature_store_client import FeatureStoreClient

# Create router
router = APIRouter()

# Dependencies
async def get_experiment_service() -> ExperimentService:
    """
    Dependency for getting the experiment service.
    
    Returns:
        Experiment service
    """
    # Configuration should come from environment in production
    mongo_url = "mongodb://localhost:27017"
    feature_store_url = "http://localhost:8001"
    
    repository = ExperimentRepository(mongo_url)
    feature_store = FeatureStoreClient(feature_store_url)
    return ExperimentService(repository, feature_store)


@router.post(
    "/",
    response_model=Experiment,
    summary="Create a new experiment",
    description="Create a new machine learning experiment."
)
async def create_experiment(
    experiment_data: ExperimentCreate,
    service: ExperimentService = Depends(get_experiment_service)
):
    """
    Create a new experiment.
    
    Args:
        experiment_data: Data for the new experiment
        service: Experiment service
    
    Returns:
        Created experiment
    """
    try:
        experiment = await service.create_experiment(experiment_data)
        return experiment
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/",
    response_model=List[Experiment],
    summary="List experiments",
    description="List experiments with optional filtering."
)
async def list_experiments(
    skip: int = Query(0, description="Number of experiments to skip"),
    limit: int = Query(100, description="Maximum number of experiments to return"),
    status: Optional[str] = Query(None, description="Filter by experiment status"),
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    sort_by: str = Query("created_at", description="Field to sort by"),
    sort_order: str = Query("desc", description="Sort order (asc or desc)"),
    service: ExperimentService = Depends(get_experiment_service)
):
    """
    List experiments with optional filtering.
    
    Args:
        skip: Number of experiments to skip
        limit: Maximum number of experiments to return
        status: Optional filter by experiment status
        model_type: Optional filter by model type
        tags: Optional filter by tags (all must match)
        sort_by: Field to sort by
        sort_order: Sort order (asc or desc)
        service: Experiment service
    
    Returns:
        List of experiments
    """
    try:
        experiments = await service.list_experiments(
            skip=skip,
            limit=limit,
            status=status,
            model_type=model_type,
            tags=tags,
            sort_by=sort_by,
            sort_order=sort_order
        )
        return experiments
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{experiment_id}",
    response_model=Experiment,
    summary="Get experiment",
    description="Get a specific experiment by ID."
)
async def get_experiment(
    experiment_id: str = Path(..., description="ID of the experiment to retrieve"),
    service: ExperimentService = Depends(get_experiment_service)
):
    """
    Get a specific experiment by ID.
    
    Args:
        experiment_id: ID of the experiment to retrieve
        service: Experiment service
    
    Returns:
        Experiment
    """
    try:
        experiment = await service.get_experiment(experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
        return experiment
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch(
    "/{experiment_id}",
    response_model=Experiment,
    summary="Update experiment",
    description="Update an existing experiment."
)
async def update_experiment(
    experiment_id: str = Path(..., description="ID of the experiment to update"),
    update_data: ExperimentUpdate = Body(..., description="Data to update"),
    service: ExperimentService = Depends(get_experiment_service)
):
    """
    Update an experiment.
    
    Args:
        experiment_id: ID of the experiment to update
        update_data: Data to update
        service: Experiment service
    
    Returns:
        Updated experiment
    """
    try:
        # Convert Pydantic model to dict
        update_dict = update_data.dict(exclude_unset=True)
        
        # Update the experiment
        experiment = await service.update_experiment(experiment_id, update_dict)
        
        if not experiment:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
            
        return experiment
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/{experiment_id}",
    status_code=204,
    summary="Delete experiment",
    description="Delete an experiment."
)
async def delete_experiment(
    experiment_id: str = Path(..., description="ID of the experiment to delete"),
    service: ExperimentService = Depends(get_experiment_service)
):
    """
    Delete an experiment.
    
    Args:
        experiment_id: ID of the experiment to delete
        service: Experiment service
    """
    try:
        success = await service.delete_experiment(experiment_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/{experiment_id}/start",
    response_model=Experiment,
    summary="Start experiment",
    description="Mark an experiment as running."
)
async def start_experiment(
    experiment_id: str = Path(..., description="ID of the experiment to start"),
    service: ExperimentService = Depends(get_experiment_service)
):
    """
    Start an experiment.
    
    Args:
        experiment_id: ID of the experiment to start
        service: Experiment service
    
    Returns:
        Updated experiment
    """
    try:
        experiment = await service.start_experiment(experiment_id)
        
        if not experiment:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
            
        return experiment
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/{experiment_id}/complete",
    response_model=Experiment,
    summary="Complete experiment",
    description="Mark an experiment as completed."
)
async def complete_experiment(
    experiment_id: str = Path(..., description="ID of the experiment to complete"),
    best_version: Optional[str] = Query(None, description="ID of the best model version"),
    service: ExperimentService = Depends(get_experiment_service)
):
    """
    Complete an experiment.
    
    Args:
        experiment_id: ID of the experiment to complete
        best_version: Optional ID of the best model version
        service: Experiment service
    
    Returns:
        Updated experiment
    """
    try:
        experiment = await service.complete_experiment(
            experiment_id=experiment_id,
            best_version=best_version
        )
        
        if not experiment:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
            
        return experiment
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/{experiment_id}/versions",
    summary="Add model version",
    description="Add a new model version to an experiment."
)
async def add_model_version(
    experiment_id: str = Path(..., description="ID of the experiment"),
    version_data: ModelVersionCreate = Body(..., description="Data for the new model version"),
    service: ExperimentService = Depends(get_experiment_service)
):
    """
    Add a new model version to an experiment.
    
    Args:
        experiment_id: ID of the experiment
        version_data: Data for the new model version
        service: Experiment service
    
    Returns:
        ID of the created model version
    """
    try:
        version_id = await service.add_model_version(experiment_id, version_data)
        
        if not version_id:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
            
        return {"version_id": version_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/{experiment_id}/metrics",
    summary="Log metric",
    description="Log a metric value for an experiment."
)
async def log_metric(
    experiment_id: str = Path(..., description="ID of the experiment"),
    metric_log: MetricLog = Body(..., description="Metric data to log"),
    service: ExperimentService = Depends(get_experiment_service)
):
    """
    Log a metric value for an experiment.
    
    Args:
        experiment_id: ID of the experiment
        metric_log: Metric data to log
        service: Experiment service
    
    Returns:
        Success status
    """
    try:
        success = await service.log_metric(experiment_id, metric_log)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
            
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/{experiment_id}/versions/{version_id}/artifact",
    summary="Upload model artifact",
    description="Upload a model artifact file for a specific model version."
)
async def upload_model_artifact(
    experiment_id: str = Path(..., description="ID of the experiment"),
    version_id: str = Path(..., description="ID of the model version"),
    model_file: UploadFile = File(..., description="Model artifact file"),
    service: ExperimentService = Depends(get_experiment_service)
):
    """
    Upload a model artifact file.
    
    Args:
        experiment_id: ID of the experiment
        version_id: ID of the model version
        model_file: Model artifact file
        service: Experiment service
    
    Returns:
        Path to the saved artifact
    """
    try:
        # Read the uploaded file
        model_bytes = await model_file.read()
        
        # Get the model version
        version = await service.repository.get_model_version(experiment_id, version_id)
        
        if not version:
            raise HTTPException(
                status_code=404,
                detail=f"Model version {version_id} not found for experiment {experiment_id}"
            )
        
        # Save the model artifact
        artifact_path = await service.repository.save_model_artifact(
            experiment_id=experiment_id,
            version_id=version_id,
            artifact=model_bytes
        )
        
        return {"artifact_path": artifact_path}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{experiment_id}/versions/{version_id}/artifact",
    summary="Download model artifact",
    description="Download a model artifact file for a specific model version."
)
async def download_model_artifact(
    experiment_id: str = Path(..., description="ID of the experiment"),
    version_id: str = Path(..., description="ID of the model version"),
    service: ExperimentService = Depends(get_experiment_service)
):
    """
    Download a model artifact file.
    
    Args:
        experiment_id: ID of the experiment
        version_id: ID of the model version
        service: Experiment service
    
    Returns:
        Model artifact file
    """
    try:
        # Get the model version
        version = await service.repository.get_model_version(experiment_id, version_id)
        
        if not version:
            raise HTTPException(
                status_code=404,
                detail=f"Model version {version_id} not found for experiment {experiment_id}"
            )
        
        if "artifact_path" not in version:
            raise HTTPException(
                status_code=404,
                detail=f"No artifact found for model version {version_id}"
            )
        
        artifact_path = version["artifact_path"]
        
        # Check if the file exists
        if not os.path.exists(artifact_path):
            raise HTTPException(
                status_code=404,
                detail=f"Artifact file not found for model version {version_id}"
            )
        
        # Return the file as a response
        return FileResponse(
            path=artifact_path,
            filename=f"model_{experiment_id}_{version_id}.pkl",
            media_type="application/octet-stream"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))