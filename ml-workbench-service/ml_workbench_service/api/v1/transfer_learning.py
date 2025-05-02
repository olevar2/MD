"""
Transfer Learning API

This module provides API endpoints for managing transfer learning operations.
It uses the custom exceptions from common-lib for standardized error handling.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, File, UploadFile
from pydantic import BaseModel
import pandas as pd
import json
from datetime import datetime

from ml_workbench_service.services.transfer_learning_service import TransferLearningService
from ml_workbench_service.api.dependencies import get_current_user
from ml_workbench_service.models.user import User

# Import common-lib exceptions
from common_lib.exceptions import (
    ForexTradingPlatformError,
    ModelError,
    ModelTrainingError,
    ModelPredictionError,
    DataValidationError,
    DataFetchError,
    DataTransformationError,
    ServiceError,
    ServiceUnavailableError
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/transfer-learning",
    tags=["transfer-learning"],
    responses={404: {"description": "Not found"}},
)


def get_transfer_learning_service():
    """Dependency for getting the TransferLearningService instance"""
    return TransferLearningService()


# Define API models
class TransferCandidate(BaseModel):
    """Model for transfer learning candidate information"""
    source_symbol: str
    source_timeframe: str
    similarity: float
    models: Optional[List[Dict[str, Any]]] = []


class TransferCandidatesRequest(BaseModel):
    """Request model for finding transfer candidates"""
    target_symbol: str
    target_timeframe: str
    min_similarity: float = 0.7
    include_data: bool = False


class TransferModelRequest(BaseModel):
    """Request model for creating a transfer model"""
    source_model_id: str
    target_symbol: str
    target_timeframe: str
    source_data: List[Dict[str, Any]]
    target_data: List[Dict[str, Any]]
    adapt_layers: Optional[List[str]] = None


class EvaluationRequest(BaseModel):
    """Request model for evaluating a transfer model"""
    model_id: str
    test_data: List[Dict[str, Any]]
    test_labels: List[Any]


class TransformRequest(BaseModel):
    """Request model for transforming features"""
    model_id: str
    features: List[Dict[str, Any]]


@router.get("/models", summary="List transfer learning models")
async def list_models(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    timeframe: Optional[str] = Query(None, description="Filter by timeframe"),
    is_transfer_model: Optional[bool] = Query(None, description="Filter by transfer model status"),
    transfer_learning_service: TransferLearningService = Depends(get_transfer_learning_service),
    current_user: User = Depends(get_current_user)
):
    """
    List available models with optional filtering.
    """
    try:
        models = transfer_learning_service.get_available_models(
            symbol=symbol,
            timeframe=timeframe,
            is_transfer_model=is_transfer_model
        )

        return {"models": models}

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
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list models: {str(e)}"
        )


@router.post("/candidates", summary="Find transfer learning candidates")
async def find_candidates(
    request: TransferCandidatesRequest,
    transfer_learning_service: TransferLearningService = Depends(get_transfer_learning_service),
    current_user: User = Depends(get_current_user)
):
    """
    Find suitable source models for transfer learning based on similarity.

    This endpoint analyzes feature distributions and correlations to identify
    models that could serve as good starting points for transfer learning.
    """
    try:
        # In a real implementation, we would fetch this data from a data service
        # For this POC, we'll return a simplified response

        # Mock source data (in production this would come from a data service)
        source_data = {
            f"{request.target_symbol}_{request.target_timeframe}": pd.DataFrame({
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [0.1, 0.2, 0.3]
            })
        }

        # Add some example correlated symbols
        if request.target_symbol.startswith("EUR"):
            source_data["GBPUSD_1h"] = pd.DataFrame({
                "feature1": [1.1, 2.1, 3.1],
                "feature2": [0.15, 0.25, 0.35]
            })
        elif request.target_symbol.startswith("BTC"):
            source_data["ETHUSD_1h"] = pd.DataFrame({
                "feature1": [10.1, 20.1, 30.1],
                "feature2": [1.5, 2.5, 3.5]
            })

        candidates = transfer_learning_service.find_transfer_candidates(
            target_symbol=request.target_symbol,
            target_timeframe=request.target_timeframe,
            source_data=source_data,
            min_similarity=request.min_similarity
        )

        return {
            "candidates": candidates,
            "target_symbol": request.target_symbol,
            "target_timeframe": request.target_timeframe
        }

    except DataValidationError as e:
        logger.error(f"Data validation error during candidate search: {e.message}")
        raise HTTPException(status_code=400, detail=f"Data validation error: {e.message}")

    except DataFetchError as e:
        logger.error(f"Data fetch error during candidate search: {e.message}")
        raise HTTPException(status_code=500, detail=f"Data fetch error: {e.message}")

    except ModelError as e:
        logger.error(f"Model error during candidate search: {e.message}")
        raise HTTPException(status_code=500, detail=f"Model error: {e.message}")

    except ServiceUnavailableError as e:
        logger.error(f"Service unavailable: {e.message}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e.message}")

    except ForexTradingPlatformError as e:
        logger.error(f"Platform error during candidate search: {e.message}")
        raise HTTPException(status_code=500, detail=f"Platform error: {e.message}")

    except Exception as e:
        logger.error(f"Error finding transfer candidates: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to find transfer candidates: {str(e)}"
        )


@router.post("/models", summary="Create a transfer learning model")
async def create_transfer_model(
    request: TransferModelRequest,
    transfer_learning_service: TransferLearningService = Depends(get_transfer_learning_service),
    current_user: User = Depends(get_current_user)
):
    """
    Create a transfer learning model by adapting a source model to a target domain.

    This endpoint creates a new model by transferring knowledge from a source model
    to a target domain (different instrument or timeframe).
    """
    try:
        # Convert list data to DataFrame
        source_df = pd.DataFrame(request.source_data)
        target_df = pd.DataFrame(request.target_data)

        result = transfer_learning_service.create_transfer_model(
            source_model_id=request.source_model_id,
            source_data=source_df,
            target_data=target_df,
            target_symbol=request.target_symbol,
            target_timeframe=request.target_timeframe,
            adapt_layers=request.adapt_layers
        )

        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Failed to create transfer model")
            )

        return result

    except HTTPException:
        raise

    except DataValidationError as e:
        logger.error(f"Data validation error during transfer model creation: {e.message}")
        raise HTTPException(status_code=400, detail=f"Data validation error: {e.message}")

    except ModelTrainingError as e:
        logger.error(f"Model training error: {e.message}")
        raise HTTPException(status_code=400, detail=f"Model training error: {e.message}")

    except ModelError as e:
        logger.error(f"Model error: {e.message}")
        raise HTTPException(status_code=500, detail=f"Model error: {e.message}")

    except DataTransformationError as e:
        logger.error(f"Data transformation error: {e.message}")
        raise HTTPException(status_code=400, detail=f"Data transformation error: {e.message}")

    except ServiceUnavailableError as e:
        logger.error(f"Service unavailable: {e.message}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e.message}")

    except ForexTradingPlatformError as e:
        logger.error(f"Platform error during transfer model creation: {e.message}")
        raise HTTPException(status_code=500, detail=f"Platform error: {e.message}")

    except Exception as e:
        logger.error(f"Error creating transfer model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create transfer model: {str(e)}"
        )


@router.post("/models/{model_id}/evaluate", summary="Evaluate a transfer model")
async def evaluate_model(
    model_id: str = Path(..., description="ID of the transfer model"),
    request: EvaluationRequest = Body(...),
    transfer_learning_service: TransferLearningService = Depends(get_transfer_learning_service),
    current_user: User = Depends(get_current_user)
):
    """
    Evaluate a transfer model on test data.

    This endpoint measures how well the transfer learning worked by evaluating
    the model on target domain test data.
    """
    try:
        # Convert list data to DataFrame/Series
        test_df = pd.DataFrame(request.test_data)
        test_labels = pd.Series(request.test_labels)

        result = transfer_learning_service.evaluate_transfer_model(
            model_id=model_id,
            test_data=test_df,
            test_labels=test_labels
        )

        if not result.get("success", False):
            raise HTTPException(
                status_code=404 if "not found" in result.get("error", "").lower() else 400,
                detail=result.get("error", "Failed to evaluate transfer model")
            )

        return result

    except HTTPException:
        raise

    except DataValidationError as e:
        logger.error(f"Data validation error during model evaluation: {e.message}")
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
        logger.error(f"Platform error during model evaluation: {e.message}")
        raise HTTPException(status_code=500, detail=f"Platform error: {e.message}")

    except Exception as e:
        logger.error(f"Error evaluating transfer model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to evaluate transfer model: {str(e)}"
        )


@router.post("/transform", summary="Transform features using a transfer model")
async def transform_features(
    request: TransformRequest,
    transfer_learning_service: TransferLearningService = Depends(get_transfer_learning_service),
    current_user: User = Depends(get_current_user)
):
    """
    Transform features from target domain to match source domain using a transfer model.

    This endpoint applies the feature transformations learned during transfer learning
    to adapt new data to the source model's expected format.
    """
    try:
        # Convert list data to DataFrame
        features_df = pd.DataFrame(request.features)

        result = transfer_learning_service.transform_features(
            model_id=request.model_id,
            features=features_df
        )

        if not result.get("success", False):
            raise HTTPException(
                status_code=404 if "not found" in result.get("error", "").lower() else 400,
                detail=result.get("error", "Failed to transform features")
            )

        return result

    except HTTPException:
        raise

    except DataValidationError as e:
        logger.error(f"Data validation error during feature transformation: {e.message}")
        raise HTTPException(status_code=400, detail=f"Data validation error: {e.message}")

    except DataTransformationError as e:
        logger.error(f"Data transformation error: {e.message}")
        raise HTTPException(status_code=400, detail=f"Data transformation error: {e.message}")

    except ModelError as e:
        logger.error(f"Model error during feature transformation: {e.message}")
        raise HTTPException(status_code=500, detail=f"Model error: {e.message}")

    except ServiceUnavailableError as e:
        logger.error(f"Service unavailable: {e.message}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e.message}")

    except ForexTradingPlatformError as e:
        logger.error(f"Platform error during feature transformation: {e.message}")
        raise HTTPException(status_code=500, detail=f"Platform error: {e.message}")

    except Exception as e:
        logger.error(f"Error transforming features: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to transform features: {str(e)}"
        )


@router.post("/upload-data", summary="Upload data for transfer learning")
async def upload_data(
    file: UploadFile = File(...),
    transfer_learning_service: TransferLearningService = Depends(get_transfer_learning_service),
    current_user: User = Depends(get_current_user)
):
    """
    Upload data for transfer learning analysis.

    This endpoint accepts CSV or JSON files containing feature data for
    transfer learning analysis.
    """
    try:
        content = await file.read()

        # Check file type
        if file.filename.endswith('.csv'):
            import io
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))

        elif file.filename.endswith('.json'):
            data = json.loads(content.decode('utf-8'))
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])

        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload CSV or JSON files."
            )

        # Return basic info about uploaded data
        return {
            "success": True,
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns),
            "preview": df.head(5).to_dict(orient="records")
        }

    except HTTPException:
        raise

    except DataValidationError as e:
        logger.error(f"Data validation error during file upload: {e.message}")
        raise HTTPException(status_code=400, detail=f"Data validation error: {e.message}")

    except DataTransformationError as e:
        logger.error(f"Data transformation error during file upload: {e.message}")
        raise HTTPException(status_code=400, detail=f"Data transformation error: {e.message}")

    except ServiceUnavailableError as e:
        logger.error(f"Service unavailable: {e.message}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e.message}")

    except ForexTradingPlatformError as e:
        logger.error(f"Platform error during file upload: {e.message}")
        raise HTTPException(status_code=500, detail=f"Platform error: {e.message}")

    except Exception as e:
        logger.error(f"Error processing uploaded data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process uploaded data: {str(e)}"
        )
