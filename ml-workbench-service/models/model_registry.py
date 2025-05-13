"""
Model Registry API Routes

This module implements the API endpoints for the ML Model Registry.
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
from services.model_registry_service import ModelRegistryService
from models.model_stage import ModelStage
from models.model_metadata import ModelMetadata
from models.model_version import ModelVersion
from models.registry_exceptions import ModelRegistryException, ModelNotFoundException, ModelVersionNotFoundException
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)


from ml_workbench_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ModelRegistrationRequest(BaseModel):
    """Request model for model registration"""
    model_name: str = Field(..., description='Name of the model')
    model_type: str = Field(..., description=
        "Type of model (e.g., 'classification', 'regression', 'forecasting')")
    description: str = Field(..., description='Model description')
    version_desc: Optional[str] = Field(None, description=
        'Description for this version')
    tags: Optional[Dict[str, str]] = Field(None, description=
        'Tags for model categorization')
    metrics: Optional[Dict[str, float]] = Field(None, description=
        'Performance metrics')
    parameters: Optional[Dict[str, Any]] = Field(None, description=
        'Model parameters/hyperparameters')
    framework: str = Field('sklearn', description='ML framework used')
    feature_names: Optional[List[str]] = Field(None, description=
        'Names of features used')
    target_names: Optional[List[str]] = Field(None, description=
        'Names of target variables')


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
    description: str
    metrics: Dict[str, Any]
    parameters: Dict[str, Any]
    framework: str
    stage: str
    feature_names: List[str]
    target_names: List[str]
    artifact_names: List[str]


class StageUpdateRequest(BaseModel):
    """Request model for updating model stage"""
    stage: str = Field(..., description=
        "New stage ('development', 'staging', 'production', 'archived')")


class MetricsUpdateRequest(BaseModel):
    """Request model for updating model metrics"""
    metrics: Dict[str, float] = Field(..., description='Updated metrics')


class ABTestRequest(BaseModel):
    """Request model for creating an A/B test"""
    test_name: str = Field(..., description='Name for the A/B test')
    version_a: int = Field(..., description='First version for testing (A)')
    version_b: int = Field(..., description='Second version for testing (B)')
    traffic_split: float = Field(0.5, description=
        'Portion of traffic to send to version B (0.0-1.0)')
    description: Optional[str] = Field('', description=
        'Description of the A/B test')


class ABTestUpdateRequest(BaseModel):
    """Request model for updating an A/B test"""
    status: Optional[str] = Field(None, description=
        "New status ('active', 'completed', 'cancelled')")
    traffic_split: Optional[float] = Field(None, description=
        'New traffic split')


def get_model_registry_service():
    """Dependency for getting the ModelRegistryService instance."""
    registry_path = 'd:/ML/model_registry'
    return ModelRegistryService(registry_path)


router = APIRouter(prefix='/api/v1/model-registry', tags=['model-registry'],
    responses={(404): {'description': 'Not found'}})


@router.post('/models', response_model=ModelRegistrationResponse)
@async_with_exception_handling
async def register_model(request: ModelRegistrationRequest=Body(...),
    model_file: UploadFile=File(...), registry_service:
    ModelRegistryService=Depends(get_model_registry_service)):
    """
    Register a new model or new version of an existing model.
    
    Requires a model file upload and metadata.
    """
    try:
        model_bytes = await model_file.read()
        model = joblib.load(io.BytesIO(model_bytes))
        metadata = registry_service.register_model(model=model, model_name=
            request.model_name, model_type=request.model_type, description=
            request.description, version_desc=request.version_desc, tags=
            request.tags, metrics=request.metrics, parameters=request.
            parameters, framework=request.framework, feature_names=request.
            feature_names, target_names=request.target_names)
        latest_version = metadata.versions[-1]
        return {'model_name': metadata.name, 'model_type': metadata.
            model_type, 'version': latest_version.version, 'version_id':
            latest_version.version_id, 'creation_time': latest_version.
            creation_time, 'stage': latest_version.stage.value}
    except Exception as e:
        logger.error(f'Failed to register model: {e}')
        raise HTTPException(status_code=500, detail=
            f'Failed to register model: {str(e)}')


@router.post('/models/{model_name}/artifacts', response_model=dict)
@async_with_exception_handling
async def add_model_artifacts(model_name: str=Path(..., description=
    'Name of the model'), version: int=Query(..., description=
    'Model version'), artifact_name: str=Query(..., description=
    'Name for the artifact'), artifact_file: UploadFile=File(...),
    registry_service: ModelRegistryService=Depends(get_model_registry_service)
    ):
    """
    Add an artifact to a model version.
    
    This could be feature importance data, example inputs/outputs, or other related files.
    """
    try:
        metadata = registry_service.get_model_metadata(model_name)
        version_info = None
        for v in metadata.versions:
            if v.version == version:
                version_info = v
                break
        if version_info is None:
            raise ModelVersionNotFoundException(
                f"Version {version} not found for model '{model_name}'")
        artifact_bytes = await artifact_file.read()
        artifact_content = None
        if artifact_file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(artifact_bytes))
            artifact_content = df
        elif artifact_file.filename.endswith('.json'):
            artifact_content = json.loads(artifact_bytes.decode('utf-8'))
        else:
            artifact_content = joblib.load(io.BytesIO(artifact_bytes))
        safe_model_name = registry_service._sanitize_name(model_name)
        artifact_dir = (registry_service.models_path / safe_model_name /
            f'v{version}' / 'artifacts')
        artifact_dir.mkdir(exist_ok=True)
        artifact_path = artifact_dir / artifact_name
        if isinstance(artifact_content, pd.DataFrame):
            artifact_path = artifact_path.with_suffix('.csv')
            artifact_content.to_csv(artifact_path, index=False)
        elif isinstance(artifact_content, (dict, list)):
            artifact_path = artifact_path.with_suffix('.json')
            with open(artifact_path, 'w') as f:
                json.dump(artifact_content, f, indent=2)
        else:
            artifact_path = artifact_path.with_suffix('.joblib')
            joblib.dump(artifact_content, artifact_path)
        version_info.artifact_paths[artifact_name] = str(artifact_path)
        registry_service._save_metadata(metadata)
        return {'status': 'success', 'message':
            f"Artifact '{artifact_name}' added to model '{model_name}' version {version}"
            , 'artifact_path': str(artifact_path)}
    except ModelNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelVersionNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f'Failed to add artifact: {e}')
        raise HTTPException(status_code=500, detail=
            f'Failed to add artifact: {str(e)}')


@router.get('/models', response_model=List[ModelMetadataResponse])
@async_with_exception_handling
async def list_models(model_type: Optional[str]=Query(None, description=
    'Filter by model type'), tag_name: Optional[str]=Query(None,
    description='Filter by tag name'), tag_value: Optional[str]=Query(None,
    description='Filter by tag value'), registry_service:
    ModelRegistryService=Depends(get_model_registry_service)):
    """
    List all models in the registry with optional filtering.
    """
    try:
        tag_filter = None
        if tag_name and tag_value:
            tag_filter = {tag_name: tag_value}
        models = registry_service.list_models(model_type=model_type,
            tag_filter=tag_filter)
        return [ModelMetadataResponse(name=model['name'], model_type=model[
            'model_type'], description=model['description'], tags=model[
            'tags'], creation_time=model['creation_time'], latest_version=
            model['latest_version'], version_count=model['version_count']) for
            model in models]
    except Exception as e:
        logger.error(f'Failed to list models: {e}')
        raise HTTPException(status_code=500, detail=
            f'Failed to list models: {str(e)}')


@router.get('/models/search', response_model=List[Dict[str, Any]])
@async_with_exception_handling
async def search_models(name_contains: Optional[str]=Query(None,
    description='Filter models whose name contains this string'),
    model_type: Optional[str]=Query(None, description=
    'Filter by model type'), tag_name: Optional[str]=Query(None,
    description='Filter by tag name'), tag_value: Optional[str]=Query(None,
    description='Filter by tag value'), min_metric_name: Optional[str]=
    Query(None, description='Name of metric for minimum threshold'),
    min_metric_value: Optional[float]=Query(None, description=
    'Minimum value for the metric'), production_only: bool=Query(False,
    description='Only include models with production versions'),
    registry_service: ModelRegistryService=Depends(get_model_registry_service)
    ):
    """
    Search for models with advanced filtering options.
    """
    try:
        tag_filter = None
        if tag_name and tag_value:
            tag_filter = {tag_name: tag_value}
        min_metric_filter = None
        if min_metric_name and min_metric_value is not None:
            min_metric_filter = {min_metric_name: min_metric_value}
        models = registry_service.search_models(name_contains=name_contains,
            model_type=model_type, tag_filter=tag_filter, min_metric_filter
            =min_metric_filter, production_only=production_only)
        return models
    except Exception as e:
        logger.error(f'Failed to search models: {e}')
        raise HTTPException(status_code=500, detail=
            f'Failed to search models: {str(e)}')


@router.get('/models/{model_name}', response_model=ModelMetadataResponse)
@async_with_exception_handling
async def get_model_metadata(model_name: str=Path(..., description=
    'Name of the model'), registry_service: ModelRegistryService=Depends(
    get_model_registry_service)):
    """
    Get metadata for a model.
    """
    try:
        metadata = registry_service.get_model_metadata(model_name)
        return ModelMetadataResponse(name=metadata.name, model_type=
            metadata.model_type, description=metadata.description, tags=
            metadata.tags, creation_time=metadata.creation_time,
            latest_version=metadata.latest_version, version_count=len(
            metadata.versions))
    except ModelNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f'Failed to get model metadata: {e}')
        raise HTTPException(status_code=500, detail=
            f'Failed to get model metadata: {str(e)}')


@router.get('/models/{model_name}/versions/{version}', response_model=
    ModelVersionResponse)
@async_with_exception_handling
async def get_model_version(model_name: str=Path(..., description=
    'Name of the model'), version: int=Path(..., description=
    'Model version'), registry_service: ModelRegistryService=Depends(
    get_model_registry_service)):
    """
    Get details for a specific model version.
    """
    try:
        metadata = registry_service.get_model_metadata(model_name)
        version_info = None
        for v in metadata.versions:
            if v.version == version:
                version_info = v
                break
        if version_info is None:
            raise ModelVersionNotFoundException(
                f"Version {version} not found for model '{model_name}'")
        return ModelVersionResponse(version=version_info.version,
            version_id=version_info.version_id, creation_time=version_info.
            creation_time, description=version_info.description, metrics=
            version_info.metrics.as_dict(), parameters=version_info.
            parameters, framework=version_info.framework, stage=
            version_info.stage.value, feature_names=version_info.
            feature_names, target_names=version_info.target_names,
            artifact_names=list(version_info.artifact_paths.keys()))
    except ModelNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelVersionNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f'Failed to get model version: {e}')
        raise HTTPException(status_code=500, detail=
            f'Failed to get model version: {str(e)}')


@router.patch('/models/{model_name}/versions/{version}/stage',
    response_model=ModelVersionResponse)
@async_with_exception_handling
async def update_model_version_stage(model_name: str=Path(..., description=
    'Name of the model'), version: int=Path(..., description=
    'Model version'), request: StageUpdateRequest=Body(...),
    registry_service: ModelRegistryService=Depends(get_model_registry_service)
    ):
    """
    Update the stage of a model version.
    """
    try:
        try:
            stage = ModelStage(request.stage)
        except ValueError:
            raise HTTPException(status_code=400, detail=
                f"Invalid stage '{request.stage}'. Must be one of: development, staging, production, archived"
                )
        metadata = registry_service.update_model_version_stage(model_name=
            model_name, version=version, stage=stage)
        version_info = None
        for v in metadata.versions:
            if v.version == version:
                version_info = v
                break
        if version_info is None:
            raise ModelVersionNotFoundException(
                f"Version {version} not found for model '{model_name}'")
        return ModelVersionResponse(version=version_info.version,
            version_id=version_info.version_id, creation_time=version_info.
            creation_time, description=version_info.description, metrics=
            version_info.metrics.as_dict(), parameters=version_info.
            parameters, framework=version_info.framework, stage=
            version_info.stage.value, feature_names=version_info.
            feature_names, target_names=version_info.target_names,
            artifact_names=list(version_info.artifact_paths.keys()))
    except ModelNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelVersionNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelRegistryException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f'Failed to update model version stage: {e}')
        raise HTTPException(status_code=500, detail=
            f'Failed to update model version stage: {str(e)}')


@router.patch('/models/{model_name}/versions/{version}/metrics',
    response_model=Dict[str, Any])
@async_with_exception_handling
async def update_model_metrics(model_name: str=Path(..., description=
    'Name of the model'), version: int=Path(..., description=
    'Model version'), request: MetricsUpdateRequest=Body(...),
    registry_service: ModelRegistryService=Depends(get_model_registry_service)
    ):
    """
    Update metrics for a model version.
    """
    try:
        metadata = registry_service.update_model_version_metrics(model_name
            =model_name, version=version, metrics=request.metrics)
        version_info = None
        for v in metadata.versions:
            if v.version == version:
                version_info = v
                break
        return {'model_name': model_name, 'version': version, 'metrics': 
            version_info.metrics.as_dict() if version_info else {}}
    except ModelNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelVersionNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f'Failed to update model metrics: {e}')
        raise HTTPException(status_code=500, detail=
            f'Failed to update model metrics: {str(e)}')


@router.delete('/models/{model_name}/versions/{version}')
@async_with_exception_handling
async def delete_model_version(model_name: str=Path(..., description=
    'Name of the model'), version: int=Path(..., description=
    'Model version'), registry_service: ModelRegistryService=Depends(
    get_model_registry_service)):
    """
    Delete a model version.
    """
    try:
        registry_service.delete_model_version(model_name=model_name,
            version=version)
        return {'status': 'success', 'message':
            f"Deleted version {version} of model '{model_name}'"}
    except ModelNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelVersionNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelRegistryException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f'Failed to delete model version: {e}')
        raise HTTPException(status_code=500, detail=
            f'Failed to delete model version: {str(e)}')


@router.delete('/models/{model_name}')
@async_with_exception_handling
async def delete_model(model_name: str=Path(..., description=
    'Name of the model'), registry_service: ModelRegistryService=Depends(
    get_model_registry_service)):
    """
    Delete a model and all its versions.
    """
    try:
        success = registry_service.delete_model(model_name)
        if success:
            return {'status': 'success', 'message':
                f"Deleted model '{model_name}'"}
        else:
            return {'status': 'error', 'message':
                f"Failed to delete model '{model_name}'"}
    except ModelNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelRegistryException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f'Failed to delete model: {e}')
        raise HTTPException(status_code=500, detail=
            f'Failed to delete model: {str(e)}')


@router.post('/models/{model_name}/ab-tests', response_model=Dict[str, Any])
@async_with_exception_handling
async def create_ab_test(model_name: str=Path(..., description=
    'Name of the model'), request: ABTestRequest=Body(...),
    registry_service: ModelRegistryService=Depends(get_model_registry_service)
    ):
    """
    Set up an A/B test between two model versions.
    """
    try:
        ab_test = registry_service.setup_ab_test(model_name=model_name,
            version_a=request.version_a, version_b=request.version_b,
            test_name=request.test_name, traffic_split=request.
            traffic_split, description=request.description)
        return ab_test
    except ModelNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelVersionNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelRegistryException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f'Failed to create A/B test: {e}')
        raise HTTPException(status_code=500, detail=
            f'Failed to create A/B test: {str(e)}')


@router.get('/ab-tests', response_model=List[Dict[str, Any]])
@async_with_exception_handling
async def list_ab_tests(model_name: Optional[str]=Query(None, description=
    'Filter tests for a specific model'), status: Optional[str]=Query(None,
    description=
    "Filter by test status ('active', 'completed', 'cancelled')"),
    registry_service: ModelRegistryService=Depends(get_model_registry_service)
    ):
    """
    List A/B tests with optional filtering.
    """
    try:
        tests = registry_service.list_ab_tests(model_name=model_name,
            status=status)
        return tests
    except Exception as e:
        logger.error(f'Failed to list A/B tests: {e}')
        raise HTTPException(status_code=500, detail=
            f'Failed to list A/B tests: {str(e)}')


@router.get('/ab-tests/{test_id}', response_model=Dict[str, Any])
@async_with_exception_handling
async def get_ab_test(test_id: str=Path(..., description=
    'ID of the A/B test'), registry_service: ModelRegistryService=Depends(
    get_model_registry_service)):
    """
    Get A/B test configuration.
    """
    try:
        test = registry_service.get_ab_test(test_id)
        return test
    except ModelRegistryException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f'Failed to get A/B test: {e}')
        raise HTTPException(status_code=500, detail=
            f'Failed to get A/B test: {str(e)}')


@router.patch('/ab-tests/{test_id}', response_model=Dict[str, Any])
@async_with_exception_handling
async def update_ab_test(test_id: str=Path(..., description=
    'ID of the A/B test'), request: ABTestUpdateRequest=Body(...),
    registry_service: ModelRegistryService=Depends(get_model_registry_service)
    ):
    """
    Update an A/B test.
    """
    try:
        test = registry_service.update_ab_test(test_id=test_id, status=
            request.status, traffic_split=request.traffic_split)
        return test
    except ModelRegistryException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f'Failed to update A/B test: {e}')
        raise HTTPException(status_code=500, detail=
            f'Failed to update A/B test: {str(e)}')


@router.get('/models/{model_name}/compare', response_model=Dict[str, Any])
@async_with_exception_handling
async def compare_model_versions(model_name: str=Path(..., description=
    'Name of the model'), version1: int=Query(..., description=
    'First version to compare'), version2: int=Query(..., description=
    'Second version to compare'), registry_service: ModelRegistryService=
    Depends(get_model_registry_service)):
    """
    Compare two versions of a model.
    """
    try:
        comparison = registry_service.compare_versions(model_name=
            model_name, version1=version1, version2=version2)
        return comparison
    except ModelNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelVersionNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f'Failed to compare model versions: {e}')
        raise HTTPException(status_code=500, detail=
            f'Failed to compare model versions: {str(e)}')
