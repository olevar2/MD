"""
Model API Module.

Provides API endpoints for ML model management, training, and deployment.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import uuid
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from core_foundations.utils.logger import get_logger
from ml_workbench_service.services.model_service import ModelService
from models.model_models import Model, ModelVersion, ModelStage, ModelFramework, ModelEvaluation
logger = get_logger('model-api')
router = APIRouter()
model_service = ModelService()


from ml_workbench_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ModelCreate(BaseModel):
    """Model for creating a new ML model."""
    name: str = Field(..., description='Name of the model')
    description: Optional[str] = Field(None, description=
        'Description of the model')
    framework: ModelFramework = Field(..., description=
        'Framework used for this model')
    tags: Optional[Dict[str, str]] = Field(None, description=
        'Tags for the model')
    metadata: Optional[Dict[str, Any]] = Field(None, description=
        'Additional metadata for the model')


class ModelVersionCreate(BaseModel):
    """Model for creating a new model version."""
    model_id: str = Field(..., description=
        'ID of the model this version belongs to')
    experiment_run_id: Optional[str] = Field(None, description=
        'ID of the experiment run that created this version')
    description: Optional[str] = Field(None, description=
        'Description of the model version')
    source_uri: str = Field(..., description=
        'URI where the model artifacts are stored')
    tags: Optional[Dict[str, str]] = Field(None, description=
        'Tags for the model version')
    metadata: Optional[Dict[str, Any]] = Field(None, description=
        'Additional metadata for the model version')
    parameters: Optional[Dict[str, Any]] = Field(None, description=
        'Parameters used to train this model version')


class ModelEvaluationCreate(BaseModel):
    """Model for creating an evaluation for a model version."""
    dataset_id: str = Field(..., description=
        'ID of the dataset used for evaluation')
    metrics: Dict[str, float] = Field(..., description='Evaluation metrics')
    timestamp: Optional[datetime] = Field(None, description=
        'Time of evaluation')
    metadata: Optional[Dict[str, Any]] = Field(None, description=
        'Additional metadata for the evaluation')


@router.post('/', response_model=Model, summary='Create model', description
    ='Create a new ML model.')
@async_with_exception_handling
async def create_model(model: ModelCreate):
    """
    Create a new ML model.
    
    Args:
        model: Model to create
        
    Returns:
        The created model
    """
    try:
        created_model = model_service.create_model(name=model.name,
            description=model.description, framework=model.framework, tags=
            model.tags, metadata=model.metadata)
        return created_model
    except Exception as e:
        logger.error(f'Error creating model: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/', response_model=List[Model], summary='List models',
    description='List all ML models.')
@async_with_exception_handling
async def list_models(name: Optional[str]=Query(None, description=
    'Filter by name'), framework: Optional[ModelFramework]=Query(None,
    description='Filter by framework'), tag: Optional[str]=Query(None,
    description='Filter by tag'), limit: int=Query(100, description=
    'Maximum number of models to return'), offset: int=Query(0, description
    ='Number of models to skip')):
    """
    List all ML models, optionally filtered.
    
    Args:
        name: Filter models by name
        framework: Filter models by framework
        tag: Filter models by tag
        limit: Maximum number of models to return
        offset: Number of models to skip
        
    Returns:
        List of models
    """
    try:
        models = model_service.list_models(name=name, framework=framework,
            tag=tag, limit=limit, offset=offset)
        return models
    except Exception as e:
        logger.error(f'Error listing models: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/{model_id}', response_model=Model, summary='Get model',
    description='Get a specific ML model by ID.')
@async_with_exception_handling
async def get_model(model_id: str):
    """
    Get a specific ML model by ID.
    
    Args:
        model_id: ID of the model to get
        
    Returns:
        The model
    """
    try:
        model = model_service.get_model(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail=
                f'Model {model_id} not found')
        return model
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error getting model {model_id}: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


@router.patch('/{model_id}', response_model=Model, summary='Update model',
    description='Update an existing ML model.')
@async_with_exception_handling
async def update_model(model_id: str, model: ModelCreate):
    """
    Update an existing ML model.
    
    Args:
        model_id: ID of the model to update
        model: Updated model data
        
    Returns:
        The updated model
    """
    try:
        updated_model = model_service.update_model(model_id=model_id, name=
            model.name, description=model.description, framework=model.
            framework, tags=model.tags, metadata=model.metadata)
        if updated_model is None:
            raise HTTPException(status_code=404, detail=
                f'Model {model_id} not found')
        return updated_model
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error updating model {model_id}: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


@router.delete('/{model_id}', status_code=204, summary='Delete model',
    description='Delete an ML model and all its versions.')
@async_with_exception_handling
async def delete_model(model_id: str):
    """
    Delete an ML model and all its versions.
    
    Args:
        model_id: ID of the model to delete
    """
    try:
        success = model_service.delete_model(model_id)
        if not success:
            raise HTTPException(status_code=404, detail=
                f'Model {model_id} not found')
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error deleting model {model_id}: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/versions', response_model=ModelVersion, summary=
    'Create model version', description=
    'Create a new version for an existing ML model.')
@async_with_exception_handling
async def create_model_version(version: ModelVersionCreate):
    """
    Create a new version for an existing ML model.
    
    Args:
        version: Model version to create
        
    Returns:
        The created model version
    """
    try:
        created_version = model_service.create_version(model_id=version.
            model_id, experiment_run_id=version.experiment_run_id,
            description=version.description, source_uri=version.source_uri,
            tags=version.tags, metadata=version.metadata, parameters=
            version.parameters)
        return created_version
    except Exception as e:
        logger.error(f'Error creating model version: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/versions/{version_id}', response_model=ModelVersion, summary=
    'Get model version', description='Get a specific model version by ID.')
@async_with_exception_handling
async def get_model_version(version_id: str):
    """
    Get a specific model version by ID.
    
    Args:
        version_id: ID of the model version to get
        
    Returns:
        The model version
    """
    try:
        version = model_service.get_version(version_id)
        if version is None:
            raise HTTPException(status_code=404, detail=
                f'Model version {version_id} not found')
        return version
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error getting model version {version_id}: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/{model_id}/versions', response_model=List[ModelVersion],
    summary='List model versions', description=
    'List all versions of an ML model.')
@async_with_exception_handling
async def list_model_versions(model_id: str, stage: Optional[ModelStage]=
    Query(None, description='Filter by stage'), limit: int=Query(100,
    description='Maximum number of versions to return'), offset: int=Query(
    0, description='Number of versions to skip')):
    """
    List all versions of an ML model, optionally filtered by stage.
    
    Args:
        model_id: ID of the model
        stage: Filter versions by stage
        limit: Maximum number of versions to return
        offset: Number of versions to skip
        
    Returns:
        List of model versions
    """
    try:
        versions = model_service.list_versions(model_id=model_id, stage=
            stage, limit=limit, offset=offset)
        return versions
    except Exception as e:
        logger.error(f'Error listing versions for model {model_id}: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


@router.patch('/versions/{version_id}/stage', response_model=ModelVersion,
    summary='Update model version stage', description=
    'Update the stage of a model version.')
@async_with_exception_handling
async def update_model_version_stage(version_id: str, stage: ModelStage):
    """
    Update the stage of a model version.
    
    Args:
        version_id: ID of the model version
        stage: New stage for the model version
        
    Returns:
        The updated model version
    """
    try:
        updated_version = model_service.update_version_stage(version_id=
            version_id, stage=stage)
        if updated_version is None:
            raise HTTPException(status_code=404, detail=
                f'Model version {version_id} not found')
        return updated_version
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f'Error updating stage for model version {version_id}: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/versions/{version_id}/evaluations', response_model=
    ModelEvaluation, summary='Create model evaluation', description=
    'Create an evaluation for a model version.')
@async_with_exception_handling
async def create_model_evaluation(version_id: str, evaluation:
    ModelEvaluationCreate):
    """
    Create an evaluation for a model version.
    
    Args:
        version_id: ID of the model version
        evaluation: Evaluation to create
        
    Returns:
        The created evaluation
    """
    try:
        created_evaluation = model_service.create_evaluation(version_id=
            version_id, dataset_id=evaluation.dataset_id, metrics=
            evaluation.metrics, timestamp=evaluation.timestamp, metadata=
            evaluation.metadata)
        return created_evaluation
    except Exception as e:
        logger.error(
            f'Error creating evaluation for model version {version_id}: {str(e)}'
            )
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/versions/{version_id}/evaluations', response_model=List[
    ModelEvaluation], summary='List model evaluations', description=
    'List all evaluations of a model version.')
@async_with_exception_handling
async def list_model_evaluations(version_id: str, dataset_id: Optional[str]
    =Query(None, description='Filter by dataset ID'), limit: int=Query(100,
    description='Maximum number of evaluations to return'), offset: int=
    Query(0, description='Number of evaluations to skip')):
    """
    List all evaluations of a model version, optionally filtered by dataset.
    
    Args:
        version_id: ID of the model version
        dataset_id: Filter evaluations by dataset ID
        limit: Maximum number of evaluations to return
        offset: Number of evaluations to skip
        
    Returns:
        List of model evaluations
    """
    try:
        evaluations = model_service.list_evaluations(version_id=version_id,
            dataset_id=dataset_id, limit=limit, offset=offset)
        return evaluations
    except Exception as e:
        logger.error(
            f'Error listing evaluations for model version {version_id}: {str(e)}'
            )
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/versions/{version_id}/deploy', response_model=ModelVersion,
    summary='Deploy model version', description=
    'Deploy a model version to a specific environment.')
@async_with_exception_handling
async def deploy_model_version(version_id: str, environment: str,
    background_tasks: BackgroundTasks):
    """
    Deploy a model version to a specific environment.
    
    Args:
        version_id: ID of the model version to deploy
        environment: Environment to deploy to
        background_tasks: FastAPI background tasks
        
    Returns:
        The updated model version
    """
    try:
        background_tasks.add_task(model_service.deploy_version, version_id=
            version_id, environment=environment)
        version = model_service.get_version(version_id)
        if version is None:
            raise HTTPException(status_code=404, detail=
                f'Model version {version_id} not found')
        return version
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error deploying model version {version_id}: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/serving/{model_name}/versions/{version}', response_model=Dict
    [str, Any], summary='Get serving info', description=
    'Get serving information for a deployed model version.')
@async_with_exception_handling
async def get_serving_info(model_name: str, version: str, environment:
    Optional[str]=Query(None, description=
    'Environment to get serving info for')):
    """
    Get serving information for a deployed model version.
    
    Args:
        model_name: Name of the model
        version: Version of the model
        environment: Environment to get serving info for
        
    Returns:
        Serving information for the deployed model version
    """
    try:
        info = model_service.get_serving_info(model_name=model_name,
            version=version, environment=environment)
        if info is None:
            raise HTTPException(status_code=404, detail=
                f'Serving info for model {model_name} version {version} not found'
                )
        return info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f'Error getting serving info for model {model_name} version {version}: {str(e)}'
            )
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/inference/{model_name}', response_model=Dict[str, Any],
    summary='Get model prediction', description=
    'Get a prediction from a deployed model.')
@async_with_exception_handling
async def get_prediction(model_name: str, data: Dict[str, Any], version:
    Optional[str]=Query(None, description='Model version to use'),
    environment: Optional[str]=Query(None, description=
    'Environment to get prediction from')):
    """
    Get a prediction from a deployed model.
    
    Args:
        model_name: Name of the model
        data: Input data for prediction
        version: Model version to use
        environment: Environment to get prediction from
        
    Returns:
        Model prediction
    """
    try:
        prediction = model_service.get_prediction(model_name=model_name,
            data=data, version=version, environment=environment)
        return prediction
    except Exception as e:
        logger.error(
            f'Error getting prediction from model {model_name}: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))
