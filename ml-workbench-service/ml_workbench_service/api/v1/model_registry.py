"""
Model Registry API Router

This module implements the API endpoints for the ML Model Registry.
It uses the custom exceptions from common-lib for standardized error handling.
"""

from typing import Dict, List, Any, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import joblib
import io
import json
import pandas as pd
from datetime import datetime
import logging
import os

from ml_workbench_service.model_registry.model_registry_service import ModelRegistryService
from ml_workbench_service.model_registry.model_stage import ModelStage
from ml_workbench_service.model_registry.model_metadata import ModelMetadata
from ml_workbench_service.model_registry.model_version import ModelVersion
from ml_workbench_service.model_registry.registry_exceptions import (
    ModelRegistryException, 
    ModelNotFoundException,
    ModelVersionNotFoundException,
    InvalidModelException
)

# Import common-lib exceptions
from common_lib.exceptions import (
    ForexTradingPlatformError,
    ConfigurationError,
    DataError,
    DataValidationError,
    DataFetchError,
    DataStorageError,
    DataTransformationError,
    ServiceError,
    ServiceUnavailableError,
    ServiceTimeoutError,
    ModelError,
    ModelTrainingError,
    ModelPredictionError
)

# Initialize logger
logger = logging.getLogger(__name__)

# Models for API requests and responses

class ModelRegistrationRequest(BaseModel):
    """Request model for model registration"""
    model_name: str = Field(..., description="Name of the model")
    model_type: str = Field(..., description="Type of model (e.g., 'classification', 'regression', 'forecasting')")
    description: str = Field(..., description="Model description")
    version_desc: Optional[str] = Field(None, description="Description for this version")
    tags: Optional[Dict[str, str]] = Field(None, description="Tags for model categorization")
    metrics: Optional[Dict[str, float]] = Field(None, description="Performance metrics")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Model parameters/hyperparameters")
    framework: str = Field("sklearn", description="ML framework used")
    feature_names: Optional[List[str]] = Field(None, description="Names of features used")
    target_names: Optional[List[str]] = Field(None, description="Names of target variables")

class ModelRegistrationResponse(BaseModel):
    """Response model for model registration"""
    model_name: str
    model_type: str
    version: int
    version_id: str
    creation_time: str
    stage: str

class ModelMetadataResponse(BaseModel):
    """Response model for model metadata"""
    name: str
    model_type: str
    description: str
    tags: Dict[str, str]
    creation_time: str
    latest_version: int
    version_count: int

class ModelVersionResponse(BaseModel):
    """Response model for model version details"""
    version: int
    version_id: str
    creation_time: str
    stage: str
    metrics: Optional[Dict[str, float]] = None
    parameters: Optional[Dict[str, Any]] = None
    feature_names: Optional[List[str]] = None
    target_names: Optional[List[str]] = None
    description: Optional[str] = None
    artifact_paths: Dict[str, str] = {}

class StageUpdateRequest(BaseModel):
    """Request model for updating model version stage"""
    stage: str = Field(..., description="New stage for the model version")

class MetricsUpdateRequest(BaseModel):
    """Request model for updating model version metrics"""
    metrics: Dict[str, float] = Field(..., description="Updated metrics for the model version")

class ABTestRequest(BaseModel):
    """Request model for setting up an A/B test"""
    version_a: int = Field(..., description="First version to compare")
    version_b: int = Field(..., description="Second version to compare")
    test_name: str = Field(..., description="Name for the A/B test")
    traffic_split: float = Field(0.5, description="Fraction of traffic to route to version B (0.0-1.0)")
    description: Optional[str] = Field(None, description="Description of the A/B test")

class ABTestUpdateRequest(BaseModel):
    """Request model for updating an A/B test"""
    status: Optional[str] = Field(None, description="New status for the test ('active', 'completed', 'cancelled')")
    traffic_split: Optional[float] = Field(None, description="Updated traffic split (0.0-1.0)")

# Dependency for getting service instance

def get_model_registry_service():
    """Dependency for getting the ModelRegistryService instance."""
    # In a real application, this would be configured properly
    # and possibly use dependency injection
    registry_path = os.environ.get("MODEL_REGISTRY_PATH", "./model_registry")
    return ModelRegistryService(registry_path)

# Create router

router = APIRouter(
    prefix="/model-registry",
    tags=["model-registry"],
    responses={404: {"description": "Not found"}},
)

# Model Routes

@router.post("/models", response_model=ModelRegistrationResponse)
async def register_model(
    request: ModelRegistrationRequest = Body(...),
    model_file: UploadFile = File(...),
    registry_service: ModelRegistryService = Depends(get_model_registry_service)
):
    """
    Register a new model or new version of an existing model.
    
    Requires a model file upload and metadata.
    """
    try:
        # Read model from file
        model_bytes = await model_file.read()
        model = joblib.load(io.BytesIO(model_bytes))
        
        # Register model
        metadata = registry_service.register_model(
            model=model,
            model_name=request.model_name,
            model_type=request.model_type,
            description=request.description,
            version_desc=request.version_desc,
            tags=request.tags,
            metrics=request.metrics,
            parameters=request.parameters,
            framework=request.framework,
            feature_names=request.feature_names,
            target_names=request.target_names
        )
        
        # Get latest version info
        latest_version = metadata.versions[-1]
        
        return {
            "model_name": metadata.name,
            "model_type": metadata.model_type,
            "version": latest_version.version,
            "version_id": latest_version.version_id,
            "creation_time": latest_version.creation_time,
            "stage": latest_version.stage.value
        }
    except DataValidationError as e:
        logger.error(f"Data validation error during model registration: {e.message}")
        raise HTTPException(status_code=400, detail=f"Data validation error: {e.message}")
    except ModelError as e:
        logger.error(f"Model error during registration: {e.message}")
        raise HTTPException(status_code=400, detail=f"Model error: {e.message}")
    except DataStorageError as e:
        logger.error(f"Storage error during model registration: {e.message}")
        raise HTTPException(status_code=500, detail=f"Storage error: {e.message}")
    except ForexTradingPlatformError as e:
        logger.error(f"Platform error during model registration: {e.message}")
        raise HTTPException(status_code=500, detail=f"Platform error: {e.message}")
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to register model: {str(e)}")

@router.post("/models/{model_name}/artifacts", response_model=dict)
async def add_model_artifacts(
    model_name: str = Path(..., description="Name of the model"),
    version: int = Query(..., description="Model version"),
    artifact_name: str = Query(..., description="Name for the artifact"),
    artifact_file: UploadFile = File(...),
    registry_service: ModelRegistryService = Depends(get_model_registry_service)
):
    """
    Add an artifact to a model version.
    
    Artifacts can include feature importance plots, confusion matrices, etc.
    """
    try:
        # Get model metadata
        metadata = registry_service.get_model_metadata(model_name)
        
        # Find the specific version
        version_info = None
        for v in metadata.versions:
            if v.version == version:
                version_info = v
                break
                
        if version_info is None:
            raise ModelVersionNotFoundException(f"Version {version} not found for model '{model_name}'")
        
        # Create artifact directory if it doesn't exist
        artifact_dir = os.path.join(registry_service.registry_path, model_name, f"version-{version}", "artifacts")
        os.makedirs(artifact_dir, exist_ok=True)
        
        # Save artifact file
        artifact_path = os.path.join(artifact_dir, artifact_name)
        artifact_bytes = await artifact_file.read()
        with open(artifact_path, "wb") as f:
            f.write(artifact_bytes)
        
        # Update version info
        version_info.artifact_paths[artifact_name] = str(artifact_path)
        registry_service._save_metadata(metadata)
        
        return {
            "status": "success",
            "message": f"Artifact '{artifact_name}' added to model '{model_name}' version {version}",
            "artifact_path": str(artifact_path)
        }
    except ModelNotFoundException as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ModelVersionNotFoundException as e:
        logger.error(f"Model version not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except DataStorageError as e:
        logger.error(f"Storage error during artifact addition: {e.message}")
        raise HTTPException(status_code=500, detail=f"Storage error: {e.message}")
    except Exception as e:
        logger.error(f"Failed to add artifact: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add artifact: {str(e)}")

@router.get("/models", response_model=List[ModelMetadataResponse])
async def list_models(
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    tag_name: Optional[str] = Query(None, description="Filter by tag name"),
    tag_value: Optional[str] = Query(None, description="Filter by tag value"),
    registry_service: ModelRegistryService = Depends(get_model_registry_service)
):
    """
    List all models in the registry with optional filtering.
    """
    try:
        # Prepare tag filter if both tag_name and tag_value are provided
        tag_filter = None
        if tag_name and tag_value:
            tag_filter = {tag_name: tag_value}
        
        # Get models
        models = registry_service.list_models(
            model_type=model_type,
            tag_filter=tag_filter
        )
        
        # Convert to response format
        response = []
        for metadata in models:
            response.append(
                ModelMetadataResponse(
                    name=metadata.name,
                    model_type=metadata.model_type,
                    description=metadata.description,
                    tags=metadata.tags or {},
                    creation_time=metadata.creation_time,
                    latest_version=metadata.latest_version,
                    version_count=len(metadata.versions)
                )
            )
        
        return response
    except DataFetchError as e:
        logger.error(f"Data fetch error during model listing: {e.message}")
        raise HTTPException(status_code=500, detail=f"Data fetch error: {e.message}")
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@router.get("/models/search", response_model=List[Dict[str, Any]])
async def search_models(
    name_contains: Optional[str] = Query(None, description="Filter models whose name contains this string"),
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    tag_name: Optional[str] = Query(None, description="Filter by tag name"),
    tag_value: Optional[str] = Query(None, description="Filter by tag value"),
    min_metric_name: Optional[str] = Query(None, description="Name of metric for minimum threshold"),
    min_metric_value: Optional[float] = Query(None, description="Minimum value for the metric"),
    production_only: bool = Query(False, description="Only include models with production versions"),
    registry_service: ModelRegistryService = Depends(get_model_registry_service)
):
    """
    Search for models with advanced filtering options.
    """
    try:
        # Prepare tag filter if both tag_name and tag_value are provided
        tag_filter = None
        if tag_name and tag_value:
            tag_filter = {tag_name: tag_value}
        
        # Prepare metric filter if both min_metric_name and min_metric_value are provided
        min_metric_filter = None
        if min_metric_name and min_metric_value is not None:
            min_metric_filter = {min_metric_name: min_metric_value}
        
        # Search models
        models = registry_service.search_models(
            name_contains=name_contains,
            model_type=model_type,
            tag_filter=tag_filter,
            min_metric_filter=min_metric_filter,
            production_only=production_only
        )
        
        return models
    except DataFetchError as e:
        logger.error(f"Data fetch error during model search: {e.message}")
        raise HTTPException(status_code=500, detail=f"Data fetch error: {e.message}")
    except Exception as e:
        logger.error(f"Failed to search models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search models: {str(e)}")

@router.get("/models/{model_name}", response_model=ModelMetadataResponse)
async def get_model_metadata(
    model_name: str = Path(..., description="Name of the model"),
    registry_service: ModelRegistryService = Depends(get_model_registry_service)
):
    """
    Get metadata for a model.
    """
    try:
        metadata = registry_service.get_model_metadata(model_name)
        
        return ModelMetadataResponse(
            name=metadata.name,
            model_type=metadata.model_type,
            description=metadata.description,
            tags=metadata.tags or {},
            creation_time=metadata.creation_time,
            latest_version=metadata.latest_version,
            version_count=len(metadata.versions)
        )
    except ModelNotFoundException as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except DataFetchError as e:
        logger.error(f"Data fetch error during model metadata retrieval: {e.message}")
        raise HTTPException(status_code=500, detail=f"Data fetch error: {e.message}")
    except Exception as e:
        logger.error(f"Failed to get model metadata: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model metadata: {str(e)}")

@router.get("/models/{model_name}/versions/{version}", response_model=ModelVersionResponse)
async def get_model_version(
    model_name: str = Path(..., description="Name of the model"),
    version: int = Path(..., description="Model version"),
    registry_service: ModelRegistryService = Depends(get_model_registry_service)
):
    """
    Get details for a specific model version.
    """
    try:
        metadata = registry_service.get_model_metadata(model_name)
        
        # Find the specific version
        version_info = None
        for v in metadata.versions:
            if v.version == version:
                version_info = v
                break
                
        if version_info is None:
            raise ModelVersionNotFoundException(f"Version {version} not found for model '{model_name}'")
        
        return ModelVersionResponse(
            version=version_info.version,
            version_id=version_info.version_id,
            creation_time=version_info.creation_time,
            stage=version_info.stage.value,
            metrics=version_info.metrics,
            parameters=version_info.parameters,
            feature_names=version_info.feature_names,
            target_names=version_info.target_names,
            description=version_info.description,
            artifact_paths=version_info.artifact_paths
        )
    except ModelNotFoundException as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ModelVersionNotFoundException as e:
        logger.error(f"Model version not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except DataFetchError as e:
        logger.error(f"Data fetch error during model version retrieval: {e.message}")
        raise HTTPException(status_code=500, detail=f"Data fetch error: {e.message}")
    except Exception as e:
        logger.error(f"Failed to get model version: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model version: {str(e)}")

@router.patch("/models/{model_name}/versions/{version}/stage", response_model=ModelVersionResponse)
async def update_model_version_stage(
    model_name: str = Path(..., description="Name of the model"),
    version: int = Path(..., description="Model version"),
    request: StageUpdateRequest = Body(...),
    registry_service: ModelRegistryService = Depends(get_model_registry_service)
):
    """
    Update the stage of a model version.
    """
    try:
        # Convert string to enum
        try:
            stage = ModelStage(request.stage)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid stage '{request.stage}'. Must be one of: development, staging, production, archived"
            )
        
        # Update stage
        metadata = registry_service.update_model_version_stage(
            model_name=model_name,
            version=version,
            stage=stage
        )
        
        # Find the updated version
        version_info = None
        for v in metadata.versions:
            if v.version == version:
                version_info = v
                break
        
        return ModelVersionResponse(
            version=version_info.version,
            version_id=version_info.version_id,
            creation_time=version_info.creation_time,
            stage=version_info.stage.value,
            metrics=version_info.metrics,
            parameters=version_info.parameters,
            feature_names=version_info.feature_names,
            target_names=version_info.target_names,
            description=version_info.description,
            artifact_paths=version_info.artifact_paths
        )
    except ModelNotFoundException as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ModelVersionNotFoundException as e:
        logger.error(f"Model version not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except DataValidationError as e:
        logger.error(f"Data validation error during stage update: {e.message}")
        raise HTTPException(status_code=400, detail=f"Data validation error: {e.message}")
    except Exception as e:
        logger.error(f"Failed to update model version stage: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update model version stage: {str(e)}")

@router.patch("/models/{model_name}/versions/{version}/metrics", response_model=Dict[str, Any])
async def update_model_metrics(
    model_name: str = Path(..., description="Name of the model"),
    version: int = Path(..., description="Model version"),
    request: MetricsUpdateRequest = Body(...),
    registry_service: ModelRegistryService = Depends(get_model_registry_service)
):
    """
    Update metrics for a model version.
    """
    try:
        metadata = registry_service.update_model_version_metrics(
            model_name=model_name,
            version=version,
            metrics=request.metrics
        )
        
        # Find the updated version
        version_info = None
        for v in metadata.versions:
            if v.version == version:
                version_info = v
                break
        
        return {
            "status": "success",
            "message": f"Metrics updated for model '{model_name}' version {version}",
            "metrics": version_info.metrics
        }
    except ModelNotFoundException as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ModelVersionNotFoundException as e:
        logger.error(f"Model version not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except DataValidationError as e:
        logger.error(f"Data validation error during metrics update: {e.message}")
        raise HTTPException(status_code=400, detail=f"Data validation error: {e.message}")
    except Exception as e:
        logger.error(f"Failed to update model metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update model metrics: {str(e)}")

@router.delete("/models/{model_name}")
async def delete_model(
    model_name: str = Path(..., description="Name of the model"),
    registry_service: ModelRegistryService = Depends(get_model_registry_service)
):
    """
    Delete a model and all its versions.
    """
    try:
        success = registry_service.delete_model(model_name)
        
        if success:
            return {"status": "success", "message": f"Model '{model_name}' deleted successfully"}
        else:
            return {"status": "error", "message": f"Failed to delete model '{model_name}'"}
    except ModelNotFoundException as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except DataStorageError as e:
        logger.error(f"Storage error during model deletion: {e.message}")
        raise HTTPException(status_code=500, detail=f"Storage error: {e.message}")
    except Exception as e:
        logger.error(f"Failed to delete model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")

@router.post("/models/{model_name}/ab-tests", response_model=Dict[str, Any])
async def create_ab_test(
    model_name: str = Path(..., description="Name of the model"),
    request: ABTestRequest = Body(...),
    registry_service: ModelRegistryService = Depends(get_model_registry_service)
):
    """
    Set up an A/B test between two model versions.
    """
    try:
        ab_test = registry_service.setup_ab_test(
            model_name=model_name,
            version_a=request.version_a,
            version_b=request.version_b,
            test_name=request.test_name,
            traffic_split=request.traffic_split,
            description=request.description
        )
        
        return ab_test
    except ModelNotFoundException as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ModelVersionNotFoundException as e:
        logger.error(f"Model version not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except DataValidationError as e:
        logger.error(f"Data validation error during A/B test creation: {e.message}")
        raise HTTPException(status_code=400, detail=f"Data validation error: {e.message}")
    except Exception as e:
        logger.error(f"Failed to create A/B test: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create A/B test: {str(e)}")

@router.get("/ab-tests", response_model=List[Dict[str, Any]])
async def list_ab_tests(
    model_name: Optional[str] = Query(None, description="Filter tests for a specific model"),
    status: Optional[str] = Query(None, description="Filter by test status ('active', 'completed', 'cancelled')"),
    registry_service: ModelRegistryService = Depends(get_model_registry_service)
):
    """
    List A/B tests with optional filtering.
    """
    try:
        tests = registry_service.list_ab_tests(
            model_name=model_name,
            status=status
        )
        
        return tests
    except DataFetchError as e:
        logger.error(f"Data fetch error during A/B test listing: {e.message}")
        raise HTTPException(status_code=500, detail=f"Data fetch error: {e.message}")
    except Exception as e:
        logger.error(f"Failed to list A/B tests: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list A/B tests: {str(e)}")

@router.get("/ab-tests/{test_id}", response_model=Dict[str, Any])
async def get_ab_test(
    test_id: str = Path(..., description="ID of the A/B test"),
    registry_service: ModelRegistryService = Depends(get_model_registry_service)
):
    """
    Get A/B test configuration.
    """
    try:
        test = registry_service.get_ab_test(test_id)
        
        return test
    except ModelRegistryException as e:
        logger.error(f"Registry exception: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except DataFetchError as e:
        logger.error(f"Data fetch error during A/B test retrieval: {e.message}")
        raise HTTPException(status_code=500, detail=f"Data fetch error: {e.message}")
    except Exception as e:
        logger.error(f"Failed to get A/B test: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get A/B test: {str(e)}")

@router.patch("/ab-tests/{test_id}", response_model=Dict[str, Any])
async def update_ab_test(
    test_id: str = Path(..., description="ID of the A/B test"),
    request: ABTestUpdateRequest = Body(...),
    registry_service: ModelRegistryService = Depends(get_model_registry_service)
):
    """
    Update an A/B test.
    """
    try:
        test = registry_service.update_ab_test(
            test_id=test_id,
            status=request.status,
            traffic_split=request.traffic_split
        )
        
        return test
    except ModelRegistryException as e:
        logger.error(f"Registry exception: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except DataValidationError as e:
        logger.error(f"Data validation error during A/B test update: {e.message}")
        raise HTTPException(status_code=400, detail=f"Data validation error: {e.message}")
    except Exception as e:
        logger.error(f"Failed to update A/B test: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update A/B test: {str(e)}")

@router.get("/models/{model_name}/compare", response_model=Dict[str, Any])
async def compare_model_versions(
    model_name: str = Path(..., description="Name of the model"),
    version1: int = Query(..., description="First version to compare"),
    version2: int = Query(..., description="Second version to compare"),
    registry_service: ModelRegistryService = Depends(get_model_registry_service)
):
    """
    Compare two versions of a model.
    """
    try:
        comparison = registry_service.compare_versions(
            model_name=model_name,
            version1=version1,
            version2=version2
        )
        
        return comparison
    except ModelNotFoundException as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ModelVersionNotFoundException as e:
        logger.error(f"Model version not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except DataFetchError as e:
        logger.error(f"Data fetch error during model version comparison: {e.message}")
        raise HTTPException(status_code=500, detail=f"Data fetch error: {e.message}")
    except Exception as e:
        logger.error(f"Failed to compare model versions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to compare model versions: {str(e)}")
