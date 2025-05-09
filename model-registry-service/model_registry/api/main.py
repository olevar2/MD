"""
FastAPI application for the Model Registry Service.
"""
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Query, Path, Body
from fastapi.responses import Response, StreamingResponse
import io

from model_registry.domain.model import (
    ModelType,
    ModelStage,
    ModelFramework,
    ModelMetadata,
    ModelVersion
)
from model_registry.core.service import ModelRegistryService
from model_registry.core.exceptions import (
    ModelRegistryError,
    ModelNotFoundError,
    ModelVersionNotFoundError,
    InvalidModelError,
    ArtifactNotFoundError
)
from model_registry.api.models import (
    ModelCreate,
    ModelUpdate,
    VersionCreate,
    StageUpdate,
    ModelResponse,
    VersionResponse,
    ModelsResponse,
    VersionsResponse,
    ErrorResponse,
    ABTestCreate,
    ABTestResponse,
    ABTestList,
    ABTestUpdate
)
from model_registry.infrastructure.filesystem_repository import FilesystemModelRegistry
from model_registry.api.dependencies import get_model_registry_service

# Create FastAPI application
app = FastAPI(
    title="Model Registry Service",
    description="Service for managing ML model versioning and lifecycle",
    version="1.0.0"
)

# Model Registry Routes

@app.post(
    "/models",
    response_model=ModelResponse,
    status_code=201,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def create_model(
    model: ModelCreate,
    service: ModelRegistryService = Depends(get_model_registry_service)
):
    """Create a new model in the registry"""
    try:
        metadata = await service.register_model(
            name=model.name,
            model_type=model.model_type,
            description=model.description,
            created_by=model.created_by,
            tags=model.tags,
            business_domain=model.business_domain,
            purpose=model.purpose,
            metadata=model.metadata
        )
        return ModelResponse.from_domain(metadata)
    except ModelRegistryError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/models",
    response_model=ModelsResponse,
    responses={500: {"model": ErrorResponse}}
)
async def list_models(
    name_filter: Optional[str] = Query(None, description="Filter models by name (case-insensitive contains)"),
    model_type: Optional[ModelType] = Query(None, description="Filter models by type"),
    tags: Optional[Dict[str, str]] = Query(None, description="Filter models by tags"),
    business_domain: Optional[str] = Query(None, description="Filter models by business domain"),
    service: ModelRegistryService = Depends(get_model_registry_service)
):
    """List models in the registry with optional filtering"""
    try:
        models = await service.list_models(
            name_filter=name_filter,
            model_type=model_type,
            tags=tags,
            business_domain=business_domain
        )
        return ModelsResponse(models=[ModelResponse.from_domain(m) for m in models])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/models/{model_id}",
    response_model=ModelResponse,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def get_model(
    model_id: str = Path(..., description="The ID of the model"),
    service: ModelRegistryService = Depends(get_model_registry_service)
):
    """Get details of a specific model"""
    try:
        model = await service.get_model(model_id)
        return ModelResponse.from_domain(model)
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/models/{model_id}/versions",
    response_model=VersionResponse,
    status_code=201,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def create_version(
    model_id: str,
    version: VersionCreate,
    model_file: UploadFile = File(...),
    service: ModelRegistryService = Depends(get_model_registry_service)
):
    """Create a new version of a model"""
    try:
        model_data = await model_file.read()
        version_obj = await service.create_model_version(
            model_id=model_id,
            model_data=model_data,
            framework=version.framework,
            framework_version=version.framework_version,
            created_by=version.created_by,
            description=version.description,
            metrics=version.metrics,
            hyperparameters=version.hyperparameters,
            feature_names=version.feature_names,
            target_names=version.target_names,
            experiment_id=version.experiment_id,
            tags=version.tags,
            metadata=version.metadata,
            stage=version.stage
        )
        return VersionResponse.from_domain(version_obj)
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    except ModelRegistryError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/models/{model_id}/versions",
    response_model=VersionsResponse,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def list_versions(
    model_id: str = Path(..., description="The ID of the model"),
    stage: Optional[ModelStage] = Query(None, description="Filter versions by stage"),
    service: ModelRegistryService = Depends(get_model_registry_service)
):
    """List all versions of a model"""
    try:
        versions = await service.list_model_versions(model_id, stage)
        return VersionsResponse(versions=[VersionResponse.from_domain(v) for v in versions])
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/versions/{version_id}",
    response_model=VersionResponse,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def get_version(
    version_id: str = Path(..., description="The ID of the version"),
    service: ModelRegistryService = Depends(get_model_registry_service)
):
    """Get details of a specific model version"""
    try:
        version = await service.get_model_version(version_id)
        return VersionResponse.from_domain(version)
    except ModelVersionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Version {version_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.patch(
    "/versions/{version_id}/stage",
    response_model=VersionResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def update_stage(
    version_id: str,
    stage_update: StageUpdate,
    service: ModelRegistryService = Depends(get_model_registry_service)
):
    """Update the stage of a model version"""
    try:
        version = await service.update_version_stage(version_id, stage_update.stage)
        return VersionResponse.from_domain(version)
    except ModelVersionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Version {version_id} not found")
    except ModelRegistryError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/versions/{version_id}/artifact",
    response_class=Response,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def download_artifact(
    version_id: str = Path(..., description="The ID of the version"),
    service: ModelRegistryService = Depends(get_model_registry_service)
):
    """Download the model artifact for a version"""
    try:
        version = await service.get_model_version(version_id, with_artifact=True)
        if not version.metadata.get("artifact_data"):
            raise ArtifactNotFoundError(f"No artifact found for version {version_id}")
            
        return StreamingResponse(
            io.BytesIO(version.metadata["artifact_data"]),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="model_{version_id}.joblib"'
            }
        )
    except (ModelVersionNotFoundError, ArtifactNotFoundError):
        raise HTTPException(status_code=404, detail=f"Artifact not found for version {version_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete(
    "/models/{model_id}",
    status_code=204,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def delete_model(
    model_id: str = Path(..., description="The ID of the model"),
    service: ModelRegistryService = Depends(get_model_registry_service)
):
    """Delete a model and all its versions"""
    try:
        await service.delete_model(model_id)
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    except ModelRegistryError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete(
    "/versions/{version_id}",
    status_code=204,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def delete_version(
    version_id: str = Path(..., description="The ID of the version"),
    service: ModelRegistryService = Depends(get_model_registry_service)
):
    """Delete a specific model version"""
    try:
        await service.delete_model_version(version_id)
    except ModelVersionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Version {version_id} not found")
    except ModelRegistryError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# A/B Testing Routes

@app.post(
    "/models/{model_id}/abtests",
    response_model=ABTestResponse,
    status_code=201,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def create_ab_test(
    model_id: str,
    test: ABTestCreate,
    service: ModelRegistryService = Depends(get_model_registry_service)
):
    """Create a new A/B test for a model"""
    try:
        ab_test = await service.create_ab_test(
            model_id=model_id,
            version_ids=test.version_ids,
            traffic_split=test.traffic_split,
            duration_days=test.duration_days,
            description=test.description
        )
        return ABTestResponse(**ab_test.to_dict())
    except ModelRegistryError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/abtests",
    response_model=ABTestList,
    responses={500: {"model": ErrorResponse}}
)
async def list_ab_tests(
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    status: Optional[str] = Query(None, description="Filter by test status"),
    service: ModelRegistryService = Depends(get_model_registry_service)
):
    """List A/B tests matching criteria"""
    try:
        tests = await service.list_ab_tests(model_id=model_id, status=status)
        return ABTestList(tests=[ABTestResponse(**t.to_dict()) for t in tests])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/abtests/{test_id}",
    response_model=ABTestResponse,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def get_ab_test(
    test_id: str = Path(..., description="ID of the A/B test"),
    service: ModelRegistryService = Depends(get_model_registry_service)
):
    """Get details of a specific A/B test"""
    try:
        test = await service.get_ab_test(test_id)
        return ABTestResponse(**test.to_dict())
    except ModelRegistryError:
        raise HTTPException(status_code=404, detail=f"A/B test {test_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.patch(
    "/abtests/{test_id}",
    response_model=ABTestResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def update_ab_test(
    test_id: str,
    update: ABTestUpdate,
    service: ModelRegistryService = Depends(get_model_registry_service)
):
    """Update an A/B test configuration"""
    try:
        test = await service.update_ab_test(
            test_id=test_id,
            traffic_split=update.traffic_split,
            status=update.status
        )
        return ABTestResponse(**test.to_dict())
    except ModelRegistryError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
