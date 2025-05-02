"""
ML Prediction API.

This module provides API endpoints for model predictions,
serving as the interface between clients and the ModelServingEngine.
It uses the custom exceptions from common-lib for standardized error handling.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from ml_workbench_service.services.model_serving_engine import ModelServingEngine
from ml_workbench_service.model_registry.model_registry import ModelRegistry
from ml_workbench_service.services.model_monitor import ModelMonitor

# Import common-lib exceptions
from common_lib.exceptions import (
    ForexTradingPlatformError,
    ModelError,
    ModelPredictionError,
    DataValidationError,
    DataFetchError,
    ServiceError,
    ServiceUnavailableError
)

# Create router
router = APIRouter()

# Get logger
logger = logging.getLogger(__name__)

# Models for request/response
class PredictionRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model to use for prediction")
    inputs: Dict[str, Any] = Field(..., description="Input data for the prediction")
    version_id: Optional[str] = Field(None, description="Optional specific model version to use")

class PredictionResponse(BaseModel):
    prediction: Any = Field(..., description="Prediction result")
    metadata: Dict[str, Any] = Field(..., description="Prediction metadata")

class ModelInfo(BaseModel):
    model_name: str = Field(..., description="Model name")
    versions: List[str] = Field(..., description="Available versions")
    latest_version: str = Field(..., description="Latest production version")
    description: Optional[str] = Field(None, description="Model description")

class ModelsResponse(BaseModel):
    models: List[Dict[str, Any]] = Field(..., description="List of available models")

# Dependency to get ModelServingEngine instance
def get_model_serving_engine():
    registry = ModelRegistry()
    model_monitor = ModelMonitor()
    return ModelServingEngine(model_registry=registry, model_monitor=model_monitor)

@router.post("/predict", response_model=Dict[str, Any], tags=["prediction"])
async def predict(
    request: PredictionRequest,
    serving_engine: ModelServingEngine = Depends(get_model_serving_engine)
) -> Dict[str, Any]:
    """
    Get a prediction from a model.

    Args:
        request: Prediction request with model name and inputs
        serving_engine: ModelServingEngine instance

    Returns:
        Dictionary with prediction result and metadata

    Raises:
        HTTPException: If prediction fails
    """
    try:
        # Log prediction request
        logger.info(f"Prediction request for model: {request.model_name}, "
                  f"version: {request.version_id or 'latest'}")

        # Make prediction
        result = serving_engine.predict(
            model_name=request.model_name,
            inputs=request.inputs,
            version_id=request.version_id
        )

        return result

    except DataValidationError as e:
        logger.error(f"Data validation error during prediction: {e.message}")
        raise HTTPException(status_code=400, detail=f"Data validation error: {e.message}")

    except ModelPredictionError as e:
        logger.error(f"Model prediction error: {e.message}")
        raise HTTPException(status_code=400, detail=f"Model prediction error: {e.message}")

    except ModelError as e:
        logger.error(f"Model error: {e.message}")
        raise HTTPException(status_code=500, detail=f"Model error: {e.message}")

    except ServiceUnavailableError as e:
        logger.error(f"Service unavailable: {e.message}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e.message}")

    except ForexTradingPlatformError as e:
        logger.error(f"Platform error during prediction: {e.message}")
        raise HTTPException(status_code=500, detail=f"Platform error: {e.message}")

    except ValueError as e:
        logger.error(f"Invalid prediction request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to make prediction: {str(e)}")

@router.get("/models", response_model=ModelsResponse, tags=["models"])
async def list_models(
    serving_engine: ModelServingEngine = Depends(get_model_serving_engine)
) -> Dict[str, List[Dict[str, Any]]]:
    """
    List all available models.

    Args:
        serving_engine: ModelServingEngine instance

    Returns:
        Dictionary with list of available models

    Raises:
        HTTPException: If listing models fails
    """
    try:
        # Get all models from registry
        registry = ModelRegistry()
        models = registry.list_models()

        # Get loaded models
        loaded_models = serving_engine.list_loaded_models()
        loaded_model_keys = {f"{model['model_name']}:{model['version_id']}"
                           for model in loaded_models}

        # Enhance model info
        model_infos = []
        for model in models:
            # Get versions
            versions = registry.list_versions(model.name)

            # Get latest production version
            latest_version = registry.get_latest_version(model.name, stage="production")

            # Check which versions are loaded
            version_infos = []
            for version in versions:
                is_loaded = f"{model.name}:{version.version_id}" in loaded_model_keys
                version_infos.append({
                    "version_id": version.version_id,
                    "stage": version.stage,
                    "creation_timestamp": version.creation_timestamp.isoformat(),
                    "is_loaded": is_loaded
                })

            model_infos.append({
                "name": model.name,
                "description": model.description,
                "versions": version_infos,
                "latest_production_version": latest_version.version_id if latest_version else None,
                "version_count": len(versions)
            })

        return {"models": model_infos}

    except DataFetchError as e:
        logger.error(f"Data fetch error during model listing: {e.message}")
        raise HTTPException(status_code=500, detail=f"Data fetch error: {e.message}")

    except ServiceUnavailableError as e:
        logger.error(f"Service unavailable: {e.message}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e.message}")

    except ForexTradingPlatformError as e:
        logger.error(f"Platform error during model listing: {e.message}")
        raise HTTPException(status_code=500, detail=f"Platform error: {e.message}")

    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@router.get("/models/{model_name}", tags=["models"])
async def get_model_info(
    model_name: str,
    serving_engine: ModelServingEngine = Depends(get_model_serving_engine)
) -> Dict[str, Any]:
    """
    Get information about a specific model.

    Args:
        model_name: Name of the model
        serving_engine: ModelServingEngine instance

    Returns:
        Dictionary with model information

    Raises:
        HTTPException: If model doesn't exist
    """
    try:
        # Get model from registry
        registry = ModelRegistry()
        model = registry.get_model(model_name)

        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        # Get versions
        versions = registry.list_versions(model_name)

        # Get latest production version
        latest_version = registry.get_latest_version(model_name, stage="production")

        # Get loaded models
        loaded_models = serving_engine.list_loaded_models()
        loaded_versions = [m["version_id"] for m in loaded_models
                         if m["model_name"] == model_name]

        # Build response
        version_infos = []
        for version in versions:
            metrics = {}
            if serving_engine.model_monitor:
                metrics = serving_engine.model_monitor.get_model_metrics(
                    model_name, version.version_id
                )

            metadata = registry.get_model_metadata(model_name, version.version_id)

            version_infos.append({
                "version_id": version.version_id,
                "stage": version.stage,
                "creation_timestamp": version.creation_timestamp.isoformat(),
                "is_loaded": version.version_id in loaded_versions,
                "metrics": metrics,
                "metadata": metadata.metadata if metadata else {}
            })

        return {
            "name": model.name,
            "description": model.description,
            "versions": version_infos,
            "latest_production_version": latest_version.version_id if latest_version else None
        }

    except HTTPException:
        raise

    except DataFetchError as e:
        logger.error(f"Data fetch error during model info retrieval: {e.message}")
        raise HTTPException(status_code=500, detail=f"Data fetch error: {e.message}")

    except ModelError as e:
        logger.error(f"Model error: {e.message}")
        raise HTTPException(status_code=500, detail=f"Model error: {e.message}")

    except ServiceUnavailableError as e:
        logger.error(f"Service unavailable: {e.message}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e.message}")

    except ForexTradingPlatformError as e:
        logger.error(f"Platform error during model info retrieval: {e.message}")
        raise HTTPException(status_code=500, detail=f"Platform error: {e.message}")

    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@router.get("/models/{model_name}/{version_id}", tags=["models"])
async def get_model_version_info(
    model_name: str,
    version_id: str,
    serving_engine: ModelServingEngine = Depends(get_model_serving_engine)
) -> Dict[str, Any]:
    """
    Get information about a specific model version.

    Args:
        model_name: Name of the model
        version_id: Version ID
        serving_engine: ModelServingEngine instance

    Returns:
        Dictionary with model version information

    Raises:
        HTTPException: If model version doesn't exist
    """
    try:
        return serving_engine.get_model_info(model_name, version_id)

    except DataValidationError as e:
        logger.error(f"Data validation error during model version info retrieval: {e.message}")
        raise HTTPException(status_code=400, detail=f"Data validation error: {e.message}")

    except ModelError as e:
        logger.error(f"Model error: {e.message}")
        raise HTTPException(status_code=500, detail=f"Model error: {e.message}")

    except DataFetchError as e:
        logger.error(f"Data fetch error during model version info retrieval: {e.message}")
        raise HTTPException(status_code=500, detail=f"Data fetch error: {e.message}")

    except ServiceUnavailableError as e:
        logger.error(f"Service unavailable: {e.message}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e.message}")

    except ForexTradingPlatformError as e:
        logger.error(f"Platform error during model version info retrieval: {e.message}")
        raise HTTPException(status_code=500, detail=f"Platform error: {e.message}")

    except ValueError as e:
        logger.error(f"Invalid model version request: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        logger.error(f"Error getting model version info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model version info: {str(e)}")
