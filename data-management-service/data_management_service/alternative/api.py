"""
Alternative Data API.

This module provides the API endpoints for the Alternative Data Integration framework.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from common_lib.exceptions import DataFetchError, DataProcessingError, DataValidationError
from data_management_service.alternative.models import (
    AlternativeDataType,
    AlternativeDataSource,
    FeatureExtractionConfig,
    DataSourceMetadata,
    AlternativeDataCorrelation,
    CorrelationMetric
)
from data_management_service.alternative.service import AlternativeDataService
from data_management_service.historical.service import HistoricalDataService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/alternative", tags=["alternative"])


# Request/Response models
class GetAlternativeDataRequest(BaseModel):
    """Request model for getting alternative data."""
    data_type: AlternativeDataType
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    adapter_id: Optional[str] = None
    use_cache: bool = True
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ExtractFeaturesRequest(BaseModel):
    """Request model for extracting features from alternative data."""
    data_type: AlternativeDataType
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    feature_config: Dict[str, Any]
    adapter_id: Optional[str] = None
    use_cache: bool = True
    parameters: Dict[str, Any] = Field(default_factory=dict)


class StoreAlternativeDataRequest(BaseModel):
    """Request model for storing alternative data."""
    data_type: AlternativeDataType
    data: List[Dict[str, Any]]
    source_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_by: Optional[str] = None


class StoreAlternativeDataResponse(BaseModel):
    """Response model for storing alternative data."""
    record_ids: List[str]


class GetAvailableDataTypesResponse(BaseModel):
    """Response model for getting available data types."""
    data_types: List[str]


class GetAvailableAdaptersResponse(BaseModel):
    """Response model for getting available adapters."""
    adapters: Dict[str, List[str]]


# Dependency
def get_alternative_data_service() -> AlternativeDataService:
    """
    Get the alternative data service.

    Returns:
        AlternativeDataService instance
    """
    # In a real implementation, this would get the service from a dependency injection container
    # or create it with the appropriate configuration
    config = {
        "adapters": {
            AlternativeDataType.NEWS: [
                {
                    "name": "MockNewsAdapter",
                    "source_id": "mock_news",
                    "use_mock": True
                }
            ],
            AlternativeDataType.ECONOMIC: [
                {
                    "name": "MockEconomicAdapter",
                    "source_id": "mock_economic",
                    "use_mock": True
                }
            ],
            AlternativeDataType.SENTIMENT: [
                {
                    "name": "MockSentimentAdapter",
                    "source_id": "mock_sentiment",
                    "use_mock": True
                }
            ],
            AlternativeDataType.SOCIAL_MEDIA: [
                {
                    "name": "MockSocialMediaAdapter",
                    "source_id": "mock_social_media",
                    "use_mock": True
                }
            ]
        }
    }
    
    # Get historical service
    historical_service = HistoricalDataService()
    
    return AlternativeDataService(config, historical_service)


@router.post("/data", response_model=Dict[str, Any])
async def get_alternative_data(
    request: GetAlternativeDataRequest,
    service: AlternativeDataService = Depends(get_alternative_data_service)
):
    """
    Get alternative data.

    Args:
        request: Request model
        service: Alternative data service

    Returns:
        Alternative data
    """
    try:
        data = await service.get_data(
            data_type=request.data_type,
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            adapter_id=request.adapter_id,
            use_cache=request.use_cache,
            **request.parameters
        )
        
        # Convert to records for JSON serialization
        return {
            "data": data.to_dict(orient="records"),
            "count": len(data)
        }
    except Exception as e:
        logger.error(f"Error getting alternative data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/features", response_model=Dict[str, Any])
async def extract_features(
    request: ExtractFeaturesRequest,
    service: AlternativeDataService = Depends(get_alternative_data_service)
):
    """
    Extract features from alternative data.

    Args:
        request: Request model
        service: Alternative data service

    Returns:
        Extracted features
    """
    try:
        features = await service.get_and_extract_features(
            data_type=request.data_type,
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            feature_config=request.feature_config,
            adapter_id=request.adapter_id,
            use_cache=request.use_cache,
            **request.parameters
        )
        
        # Convert to records for JSON serialization
        return {
            "features": features.to_dict(orient="records"),
            "count": len(features)
        }
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/store", response_model=StoreAlternativeDataResponse)
async def store_alternative_data(
    request: StoreAlternativeDataRequest,
    service: AlternativeDataService = Depends(get_alternative_data_service)
):
    """
    Store alternative data.

    Args:
        request: Request model
        service: Alternative data service

    Returns:
        Record IDs
    """
    try:
        # Convert data to DataFrame
        import pandas as pd
        data = pd.DataFrame(request.data)
        
        record_ids = await service.store_data(
            data=data,
            data_type=request.data_type,
            source_id=request.source_id,
            metadata=request.metadata,
            created_by=request.created_by
        )
        
        return StoreAlternativeDataResponse(record_ids=record_ids)
    except Exception as e:
        logger.error(f"Error storing alternative data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/types", response_model=GetAvailableDataTypesResponse)
async def get_available_data_types():
    """
    Get available alternative data types.

    Returns:
        Available data types
    """
    try:
        data_types = [data_type.value for data_type in AlternativeDataType]
        return GetAvailableDataTypesResponse(data_types=data_types)
    except Exception as e:
        logger.error(f"Error getting available data types: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/adapters", response_model=GetAvailableAdaptersResponse)
async def get_available_adapters(
    service: AlternativeDataService = Depends(get_alternative_data_service)
):
    """
    Get available alternative data adapters.

    Args:
        service: Alternative data service

    Returns:
        Available adapters
    """
    try:
        adapters = {}
        
        for data_type in AlternativeDataType:
            adapter_list = service.adapter_factory.get_adapters_by_type(data_type)
            adapters[data_type.value] = [adapter.name for adapter in adapter_list]
        
        return GetAvailableAdaptersResponse(adapters=adapters)
    except Exception as e:
        logger.error(f"Error getting available adapters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
