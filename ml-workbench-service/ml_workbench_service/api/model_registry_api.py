"""
Model Registry API Module.

This module provides API endpoints for ML model registry functionality, including:
- Model registration and versioning
- Model lifecycle management (staging, production, archived)
- Model metadata and version management
- Model artifact storage and retrieval
"""
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import uuid
import os
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks, UploadFile, File, Form, Body, Path, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, validator
from core_foundations.utils.logger import get_logger
from ml_workbench_service.model_registry.model_registry_service import ModelRegistryService
from ml_workbench_service.model_registry.registry import ModelMetadata, ModelVersion, ModelStatus, ModelFramework, ModelType, ModelMetrics, HyperParameters
from ml_workbench_service.model_registry.registry_exceptions import ModelRegistryException, ModelNotFoundException, ModelVersionNotFoundException, InvalidModelException
logger = get_logger('model-registry-api')
router = APIRouter()
REGISTRY_ROOT = os.environ.get('MODEL_REGISTRY_PATH', '/data/model_registry')
model_registry_service = ModelRegistryService(REGISTRY_ROOT)


from ml_workbench_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ModelRegistryCreateRequest(BaseModel):
    """Request model for creating a new model in the registry."""
    name: str
    description: Optional[str] = None
    model_type: ModelType
    created_by: str
    tags: List[str] = Field(default_factory=list)
    business_domain: Optional[str] = None
    purpose: Optional[str] = None
    training_frequency: Optional[str] = None
    monitoring_config: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelRegistryResponse(BaseModel):
    """Response model for model registry operations."""
    model_id: str
    name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    created_by: str
    model_type: str
    tags: List[str] = Field(default_factory=list)
    latest_version_number: int
    latest_version_id: Optional[str] = None
    production_version_id: Optional[str] = None
    staging_version_id: Optional[str] = None
    business_domain: Optional[str] = None
    purpose: Optional[str] = None


class ModelVersionCreateRequest(BaseModel):
    """Request model for creating a new model version."""
    model_id: str
    description: Optional[str] = None
    created_by: str
    framework: ModelFramework
    framework_version: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    training_dataset_id: Optional[str] = None
    validation_dataset_id: Optional[str] = None
    test_dataset_id: Optional[str] = None
    feature_columns: List[str] = Field(default_factory=list)
    target_column: Optional[str] = None
    preprocessing_steps: List[Dict[str, Any]] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    experiment_id: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelVersionResponse(BaseModel):
    """Response model for model version operations."""
    version_id: str
    model_id: str
    version_number: int
    created_at: datetime
    updated_at: datetime
    created_by: str
    description: Optional[str] = None
    status: str
    framework: str
    framework_version: str
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    file_path: Optional[str] = None
    tags: List[str]


class ModelListResponse(BaseModel):
    """Response model for listing models."""
    models: List[ModelRegistryResponse]
    total: int
    page: int
    page_size: int


class ModelVersionListResponse(BaseModel):
    """Response model for listing model versions."""
    versions: List[ModelVersionResponse]
    total: int
    page: int
    page_size: int


class StageTransitionRequest(BaseModel):
    """Request model for transitioning a model version to a new stage."""
    version_id: str
    stage: ModelStatus
    reason: Optional[str] = None


@router.post('/models', response_model=ModelRegistryResponse, status_code=
    status.HTTP_201_CREATED)
@async_with_exception_handling
async def create_model(model_data: ModelRegistryCreateRequest):
    """
    Create a new model in the registry.
    """
    try:
        model_metadata = ModelMetadata(name=model_data.name, description=
            model_data.description, model_type=model_data.model_type,
            created_by=model_data.created_by, tags=model_data.tags,
            business_domain=model_data.business_domain, purpose=model_data.
            purpose, training_frequency=model_data.training_frequency,
            monitoring_config=model_data.monitoring_config, metadata=
            model_data.metadata)
        registered_model = model_registry_service.register_model(model_metadata
            )
        return ModelRegistryResponse(**registered_model.to_dict())
    except ModelRegistryException as e:
        logger.error(f'Error creating model: {str(e)}')
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f'Unexpected error creating model: {str(e)}')
        raise HTTPException(status_code=500, detail=
            'An unexpected error occurred')


@router.get('/models', response_model=ModelListResponse)
@async_with_exception_handling
async def list_models(page: int=Query(1, ge=1, description='Page number'),
    page_size: int=Query(10, ge=1, le=100, description='Page size'), name:
    Optional[str]=Query(None, description='Filter by model name'),
    model_type: Optional[ModelType]=Query(None, description=
    'Filter by model type'), tags: Optional[List[str]]=Query(None,
    description='Filter by tags'), created_by: Optional[str]=Query(None,
    description='Filter by creator')):
    """
    List models in the registry with optional filtering.
    """
    try:
        filters = {}
        if name:
            filters['name'] = name
        if model_type:
            filters['model_type'] = model_type
        if tags:
            filters['tags'] = tags
        if created_by:
            filters['created_by'] = created_by
        models, total = model_registry_service.list_models(page=page,
            page_size=page_size, filters=filters)
        return ModelListResponse(models=[ModelRegistryResponse(**m.to_dict(
            )) for m in models], total=total, page=page, page_size=page_size)
    except Exception as e:
        logger.error(f'Error listing models: {str(e)}')
        raise HTTPException(status_code=500, detail=
            'An error occurred while listing models')


@router.get('/models/{model_id}', response_model=ModelRegistryResponse)
@async_with_exception_handling
async def get_model(model_id: str=Path(..., description=
    'ID of the model to retrieve')):
    """
    Get a specific model by ID.
    """
    try:
        model = model_registry_service.get_model(model_id)
        return ModelRegistryResponse(**model.to_dict())
    except ModelNotFoundException as e:
        raise HTTPException(status_code=404, detail=
            f'Model not found: {model_id}')
    except Exception as e:
        logger.error(f'Error retrieving model {model_id}: {str(e)}')
        raise HTTPException(status_code=500, detail=
            'An error occurred while retrieving the model')


@router.put('/models/{model_id}', response_model=ModelRegistryResponse)
@async_with_exception_handling
async def update_model(model_id: str=Path(..., description=
    'ID of the model to update'), model_data: ModelRegistryCreateRequest=
    Body(..., description='Updated model data')):
    """
    Update an existing model's metadata.
    """
    try:
        existing_model = model_registry_service.get_model(model_id)
        existing_model.name = model_data.name
        existing_model.description = model_data.description
        existing_model.model_type = model_data.model_type
        existing_model.tags = model_data.tags
        existing_model.business_domain = model_data.business_domain
        existing_model.purpose = model_data.purpose
        existing_model.training_frequency = model_data.training_frequency
        existing_model.monitoring_config = model_data.monitoring_config
        existing_model.metadata = model_data.metadata
        existing_model.updated_at = datetime.utcnow()
        updated_model = model_registry_service.update_model(existing_model)
        return ModelRegistryResponse(**updated_model.to_dict())
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=
            f'Model not found: {model_id}')
    except Exception as e:
        logger.error(f'Error updating model {model_id}: {str(e)}')
        raise HTTPException(status_code=500, detail=
            'An error occurred while updating the model')


@router.delete('/models/{model_id}', status_code=status.HTTP_204_NO_CONTENT)
@async_with_exception_handling
async def delete_model(model_id: str=Path(..., description=
    'ID of the model to delete')):
    """
    Delete a model and all its versions.
    """
    try:
        model_registry_service.delete_model(model_id)
        return {}
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=
            f'Model not found: {model_id}')
    except Exception as e:
        logger.error(f'Error deleting model {model_id}: {str(e)}')
        raise HTTPException(status_code=500, detail=
            'An error occurred while deleting the model')


@router.post('/models/{model_id}/versions', response_model=
    ModelVersionResponse, status_code=status.HTTP_201_CREATED)
@async_with_exception_handling
async def create_model_version(model_id: str=Path(..., description=
    'ID of the model'), version_data: ModelVersionCreateRequest=Body(...,
    description='Model version data'), model_file: Optional[UploadFile]=
    File(None, description='Model artifact file')):
    """
    Create a new version of a model.
    """
    try:
        model_registry_service.get_model(model_id)
        hyperparameters = HyperParameters(values=version_data.hyperparameters)
        metrics = ModelMetrics(**version_data.metrics)
        model_version = ModelVersion(model_id=model_id, created_by=
            version_data.created_by, description=version_data.description,
            framework=version_data.framework, framework_version=
            version_data.framework_version, hyperparameters=hyperparameters,
            metrics=metrics, training_dataset_id=version_data.
            training_dataset_id, validation_dataset_id=version_data.
            validation_dataset_id, test_dataset_id=version_data.
            test_dataset_id, feature_columns=version_data.feature_columns,
            target_column=version_data.target_column, preprocessing_steps=
            version_data.preprocessing_steps, tags=version_data.tags, notes
            =version_data.notes, experiment_id=version_data.experiment_id,
            metadata=version_data.metadata)
        temp_file_path = None
        if model_file:
            temp_file_path = f'/tmp/model_upload_{uuid.uuid4()}.bin'
            with open(temp_file_path, 'wb') as buffer:
                buffer.write(await model_file.read())
        registered_version = model_registry_service.create_model_version(
            model_version, model_file_path=temp_file_path)
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        return ModelVersionResponse(**registered_version.to_dict())
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=
            f'Model not found: {model_id}')
    except Exception as e:
        logger.error(f'Error creating version for model {model_id}: {str(e)}')
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        raise HTTPException(status_code=500, detail=
            f'An error occurred while creating the model version: {str(e)}')


@router.get('/models/{model_id}/versions', response_model=
    ModelVersionListResponse)
@async_with_exception_handling
async def list_model_versions(model_id: str=Path(..., description=
    'ID of the model'), page: int=Query(1, ge=1, description='Page number'),
    page_size: int=Query(10, ge=1, le=100, description='Page size'), status:
    Optional[ModelStatus]=Query(None, description='Filter by version status')):
    """
    List versions of a specific model.
    """
    try:
        model_registry_service.get_model(model_id)
        filters = {'model_id': model_id}
        if status:
            filters['status'] = status
        versions, total = model_registry_service.list_model_versions(model_id
            =model_id, page=page, page_size=page_size, filters=filters)
        return ModelVersionListResponse(versions=[ModelVersionResponse(**v.
            to_dict()) for v in versions], total=total, page=page,
            page_size=page_size)
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=
            f'Model not found: {model_id}')
    except Exception as e:
        logger.error(f'Error listing versions for model {model_id}: {str(e)}')
        raise HTTPException(status_code=500, detail=
            'An error occurred while listing model versions')


@router.get('/models/{model_id}/versions/{version_id}', response_model=
    ModelVersionResponse)
@async_with_exception_handling
async def get_model_version(model_id: str=Path(..., description=
    'ID of the model'), version_id: str=Path(..., description=
    'ID of the version to retrieve')):
    """
    Get a specific model version.
    """
    try:
        version = model_registry_service.get_model_version(model_id, version_id
            )
        return ModelVersionResponse(**version.to_dict())
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=
            f'Model not found: {model_id}')
    except ModelVersionNotFoundException:
        raise HTTPException(status_code=404, detail=
            f'Version not found: {version_id}')
    except Exception as e:
        logger.error(
            f'Error retrieving version {version_id} for model {model_id}: {str(e)}'
            )
        raise HTTPException(status_code=500, detail=
            'An error occurred while retrieving the model version')


@router.post('/models/{model_id}/versions/{version_id}/stage',
    response_model=ModelVersionResponse)
@async_with_exception_handling
async def transition_model_version_stage(model_id: str=Path(...,
    description='ID of the model'), version_id: str=Path(..., description=
    'ID of the version'), stage_data: StageTransitionRequest=Body(...,
    description='Stage transition data')):
    """
    Transition a model version to a new stage (staging, production, etc.).
    """
    try:
        updated_version = (model_registry_service.
            transition_model_version_stage(model_id=model_id, version_id=
            version_id, new_stage=stage_data.stage, reason=stage_data.reason))
        return ModelVersionResponse(**updated_version.to_dict())
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=
            f'Model not found: {model_id}')
    except ModelVersionNotFoundException:
        raise HTTPException(status_code=404, detail=
            f'Version not found: {version_id}')
    except Exception as e:
        logger.error(
            f'Error transitioning version {version_id} for model {model_id}: {str(e)}'
            )
        raise HTTPException(status_code=500, detail=
            f'An error occurred during stage transition: {str(e)}')


@router.get('/models/{model_id}/versions/{version_id}/download')
@async_with_exception_handling
async def download_model_artifact(model_id: str=Path(..., description=
    'ID of the model'), version_id: str=Path(..., description=
    'ID of the version to download')):
    """
    Download a model artifact.
    """
    try:
        model_path = model_registry_service.get_model_artifact_path(model_id,
            version_id)
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=
                'Model artifact file not found')
        return FileResponse(path=model_path, filename=
            f'model_{model_id}_v{version_id}.bin', media_type=
            'application/octet-stream')
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=
            f'Model not found: {model_id}')
    except ModelVersionNotFoundException:
        raise HTTPException(status_code=404, detail=
            f'Version not found: {version_id}')
    except Exception as e:
        logger.error(
            f'Error downloading version {version_id} for model {model_id}: {str(e)}'
            )
        raise HTTPException(status_code=500, detail=
            'An error occurred while downloading the model artifact')


@router.put('/models/{model_id}/versions/{version_id}/metrics',
    response_model=ModelVersionResponse)
@async_with_exception_handling
async def update_model_version_metrics(model_id: str=Path(..., description=
    'ID of the model'), version_id: str=Path(..., description=
    'ID of the version'), metrics: Dict[str, float]=Body(..., description=
    'Updated metrics')):
    """
    Update metrics for a model version.
    """
    try:
        version = model_registry_service.get_model_version(model_id, version_id
            )
        metrics_obj = ModelMetrics(**metrics)
        version.metrics = metrics_obj
        version.updated_at = datetime.utcnow()
        updated_version = model_registry_service.update_model_version(version)
        return ModelVersionResponse(**updated_version.to_dict())
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=
            f'Model not found: {model_id}')
    except ModelVersionNotFoundException:
        raise HTTPException(status_code=404, detail=
            f'Version not found: {version_id}')
    except Exception as e:
        logger.error(
            f'Error updating metrics for version {version_id}: {str(e)}')
        raise HTTPException(status_code=500, detail=
            'An error occurred while updating model metrics')


@router.delete('/models/{model_id}/versions/{version_id}', status_code=
    status.HTTP_204_NO_CONTENT)
@async_with_exception_handling
async def delete_model_version(model_id: str=Path(..., description=
    'ID of the model'), version_id: str=Path(..., description=
    'ID of the version to delete')):
    """
    Delete a specific version of a model.
    """
    try:
        model_registry_service.delete_model_version(model_id, version_id)
        return {}
    except ModelNotFoundException:
        raise HTTPException(status_code=404, detail=
            f'Model not found: {model_id}')
    except ModelVersionNotFoundException:
        raise HTTPException(status_code=404, detail=
            f'Version not found: {version_id}')
    except Exception as e:
        logger.error(
            f'Error deleting version {version_id} for model {model_id}: {str(e)}'
            )
        raise HTTPException(status_code=500, detail=
            'An error occurred while deleting the model version')


@router.get('/models/search', response_model=ModelListResponse)
@async_with_exception_handling
async def search_models(query: str=Query(..., description=
    'Search query string'), page: int=Query(1, ge=1, description=
    'Page number'), page_size: int=Query(10, ge=1, le=100, description=
    'Page size')):
    """
    Search for models based on a query string.
    """
    try:
        models, total = model_registry_service.search_models(query=query,
            page=page, page_size=page_size)
        return ModelListResponse(models=[ModelRegistryResponse(**m.to_dict(
            )) for m in models], total=total, page=page, page_size=page_size)
    except Exception as e:
        logger.error(f'Error searching models: {str(e)}')
        raise HTTPException(status_code=500, detail=
            'An error occurred while searching for models')
