"""
Adapter API Module

This module provides API endpoints that use the adapter pattern.
"""
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from pydantic import BaseModel
from common_lib.interfaces.feature_store import IFeatureProvider, IFeatureStore, IFeatureGenerator
from common_lib.errors.base_exceptions import BaseError, ValidationError, DataError, ServiceError
from core.dependencies_1 import get_feature_provider, get_feature_store, get_feature_generator
from core.logging import get_logger
logger = get_logger('feature-store-service.adapter-api')
adapter_router = APIRouter(prefix='/api/v1/adapter', tags=['adapter'],
    responses={(404): {'description': 'Not found'}})


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class FeatureResponse(BaseModel):
    """Response model for feature data."""
    symbol: str
    timeframe: str
    start_time: datetime
    end_time: Optional[datetime] = None
    features: Dict[str, Any]


class FeatureMetadataResponse(BaseModel):
    """Response model for feature metadata."""
    name: str
    description: str
    parameters: Dict[str, Any]
    category: str


@adapter_router.get('/features/{symbol}/{timeframe}', response_model=
    FeatureResponse)
@async_with_exception_handling
async def get_features(symbol: str=Path(..., description=
    "The trading symbol (e.g., 'EURUSD')"), timeframe: str=Path(...,
    description="The timeframe (e.g., '1m', '5m', '1h', '1d')"), start_time:
    datetime=Query(..., description='Start time for the data'), end_time:
    Optional[datetime]=Query(None, description='End time for the data'),
    features: List[str]=Query(..., description=
    'List of features to retrieve'), feature_provider: IFeatureProvider=
    Depends(get_feature_provider)):
    """
    Get features for a symbol.
    
    Args:
        symbol: The trading symbol
        timeframe: The timeframe
        start_time: Start time for the data
        end_time: End time for the data
        features: List of features to retrieve
        feature_provider: The feature provider adapter
        
    Returns:
        FeatureResponse containing the feature data
    """
    try:
        result = await feature_provider.get_features(feature_names=features,
            symbol=symbol, timeframe=timeframe, start_time=start_time,
            end_time=end_time)
        return FeatureResponse(symbol=symbol, timeframe=timeframe,
            start_time=start_time, end_time=end_time, features={feature: (
            result[feature].tolist() if feature in result else []) for
            feature in features})
    except ServiceError as e:
        logger.error(f'Service error: {str(e)}')
        raise HTTPException(status_code=503, detail=str(e))
    except DataError as e:
        logger.error(f'Data error: {str(e)}')
        raise HTTPException(status_code=404, detail=str(e))
    except ValidationError as e:
        logger.error(f'Validation error: {str(e)}')
        raise HTTPException(status_code=400, detail=str(e))
    except BaseError as e:
        logger.error(f'Base error: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f'Unexpected error: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Unexpected error: {str(e)}')


@adapter_router.get('/metadata', response_model=List[FeatureMetadataResponse])
@async_with_exception_handling
async def get_feature_metadata(feature_provider: IFeatureProvider=Depends(
    get_feature_provider)):
    """
    Get metadata for all available features.
    
    Args:
        feature_provider: The feature provider adapter
        
    Returns:
        List of FeatureMetadataResponse containing metadata about available features
    """
    try:
        available_features = await feature_provider.get_available_features()
        metadata_list = []
        for feature_name in available_features:
            metadata = await feature_provider.get_feature_metadata(feature_name
                )
            metadata_list.append(FeatureMetadataResponse(name=feature_name,
                description=metadata.get('description', ''), parameters=
                metadata.get('parameters', {}), category=metadata.get(
                'category', '')))
        return metadata_list
    except ServiceError as e:
        logger.error(f'Service error: {str(e)}')
        raise HTTPException(status_code=503, detail=str(e))
    except BaseError as e:
        logger.error(f'Base error: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f'Unexpected error: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Unexpected error: {str(e)}')


@adapter_router.post('/store/{symbol}/{timeframe}/{feature_name}')
@async_with_exception_handling
async def store_feature(symbol: str=Path(..., description=
    "The trading symbol (e.g., 'EURUSD')"), timeframe: str=Path(...,
    description="The timeframe (e.g., '1m', '5m', '1h', '1d')"),
    feature_name: str=Path(..., description='Name of the feature to store'),
    data: Dict[str, List[float]]=..., metadata: Optional[Dict[str, Any]]=
    None, feature_store: IFeatureStore=Depends(get_feature_store)):
    """
    Store a feature in the feature store.
    
    Args:
        symbol: The trading symbol
        timeframe: The timeframe
        feature_name: Name of the feature to store
        data: Feature data to store
        metadata: Optional metadata about the feature
        feature_store: The feature store adapter
        
    Returns:
        Success message
    """
    try:
        import pandas as pd
        df = pd.DataFrame(data)
        success = await feature_store.store_feature(feature_name=
            feature_name, symbol=symbol, timeframe=timeframe, data=df,
            metadata=metadata)
        if success:
            return {'message': f'Feature {feature_name} stored successfully'}
        else:
            raise HTTPException(status_code=500, detail=
                'Failed to store feature')
    except ServiceError as e:
        logger.error(f'Service error: {str(e)}')
        raise HTTPException(status_code=503, detail=str(e))
    except ValidationError as e:
        logger.error(f'Validation error: {str(e)}')
        raise HTTPException(status_code=400, detail=str(e))
    except BaseError as e:
        logger.error(f'Base error: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f'Unexpected error: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Unexpected error: {str(e)}')
