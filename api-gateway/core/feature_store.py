"""
Feature Store Routes

This module provides routes for feature store.
"""

import logging
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException, Depends, Query, Path
from pydantic import BaseModel, Field

from services.feature_store_service import FeatureStoreService


# Create logger
logger = logging.getLogger(__name__)


# Create router
router = APIRouter()


# Create service
feature_store_service = FeatureStoreService()


# Define models
class Feature(BaseModel):
    """Feature."""
    
    name: str = Field(..., description="Feature name")
    description: str = Field(..., description="Feature description")
    type: str = Field(..., description="Feature type")
    created_at: int = Field(..., description="Creation timestamp in milliseconds")
    updated_at: int = Field(..., description="Update timestamp in milliseconds")


class FeatureValue(BaseModel):
    """Feature value."""
    
    timestamp: int = Field(..., description="Timestamp in milliseconds")
    value: Any = Field(..., description="Feature value")


class FeatureData(BaseModel):
    """Feature data."""
    
    feature: str = Field(..., description="Feature name")
    symbol: str = Field(..., description="Symbol")
    timeframe: str = Field(..., description="Timeframe")
    values: List[FeatureValue] = Field(..., description="Feature values")


class FeatureRequest(BaseModel):
    """Feature request."""
    
    feature: str = Field(..., description="Feature name")
    symbol: str = Field(..., description="Symbol")
    timeframe: str = Field(..., description="Timeframe")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Feature parameters")
    start: Optional[int] = Field(None, description="Start timestamp in milliseconds")
    end: Optional[int] = Field(None, description="End timestamp in milliseconds")
    limit: Optional[int] = Field(None, description="Limit")


class FeatureSet(BaseModel):
    """Feature set."""
    
    name: str = Field(..., description="Feature set name")
    description: str = Field(..., description="Feature set description")
    features: List[str] = Field(..., description="Features")
    created_at: int = Field(..., description="Creation timestamp in milliseconds")
    updated_at: int = Field(..., description="Update timestamp in milliseconds")


class FeatureSetRequest(BaseModel):
    """Feature set request."""
    
    name: str = Field(..., description="Feature set name")
    description: str = Field(..., description="Feature set description")
    features: List[str] = Field(..., description="Features")


# Define routes
@router.get("/features", response_model=List[Feature])
async def get_features():
    """
    Get features.
    
    Returns:
        List of features
    """
    try:
        return await feature_store_service.get_features()
    except Exception as e:
        logger.error(f"Error getting features: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/{feature}", response_model=Feature)
async def get_feature(feature: str = Path(..., description="Feature name")):
    """
    Get feature.
    
    Args:
        feature: Feature name
        
    Returns:
        Feature
    """
    try:
        return await feature_store_service.get_feature(feature)
    except Exception as e:
        logger.error(f"Error getting feature {feature}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/features/data", response_model=FeatureData)
async def get_feature_data(request: FeatureRequest):
    """
    Get feature data.
    
    Args:
        request: Feature request
        
    Returns:
        Feature data
    """
    try:
        return await feature_store_service.get_feature_data(
            request.feature,
            request.symbol,
            request.timeframe,
            request.parameters,
            request.start,
            request.end,
            request.limit
        )
    except Exception as e:
        logger.error(f"Error getting feature data for {request.feature}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/{feature}/{symbol}/{timeframe}", response_model=FeatureData)
async def get_feature_data_by_path(
    feature: str = Path(..., description="Feature name"),
    symbol: str = Path(..., description="Symbol"),
    timeframe: str = Path(..., description="Timeframe"),
    start: Optional[int] = Query(None, description="Start timestamp in milliseconds"),
    end: Optional[int] = Query(None, description="End timestamp in milliseconds"),
    limit: Optional[int] = Query(None, description="Limit")
):
    """
    Get feature data by path.
    
    Args:
        feature: Feature name
        symbol: Symbol
        timeframe: Timeframe
        start: Start timestamp in milliseconds
        end: End timestamp in milliseconds
        limit: Limit
        
    Returns:
        Feature data
    """
    try:
        return await feature_store_service.get_feature_data(
            feature,
            symbol,
            timeframe,
            {},
            start,
            end,
            limit
        )
    except Exception as e:
        logger.error(f"Error getting feature data for {feature}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature-sets", response_model=List[FeatureSet])
async def get_feature_sets():
    """
    Get feature sets.
    
    Returns:
        List of feature sets
    """
    try:
        return await feature_store_service.get_feature_sets()
    except Exception as e:
        logger.error(f"Error getting feature sets: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature-sets/{name}", response_model=FeatureSet)
async def get_feature_set(name: str = Path(..., description="Feature set name")):
    """
    Get feature set.
    
    Args:
        name: Feature set name
        
    Returns:
        Feature set
    """
    try:
        return await feature_store_service.get_feature_set(name)
    except Exception as e:
        logger.error(f"Error getting feature set {name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feature-sets", response_model=FeatureSet)
async def create_feature_set(request: FeatureSetRequest):
    """
    Create feature set.
    
    Args:
        request: Feature set request
        
    Returns:
        Feature set
    """
    try:
        return await feature_store_service.create_feature_set(
            request.name,
            request.description,
            request.features
        )
    except Exception as e:
        logger.error(f"Error creating feature set {request.name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/feature-sets/{name}", response_model=FeatureSet)
async def update_feature_set(
    name: str = Path(..., description="Feature set name"),
    request: FeatureSetRequest = None
):
    """
    Update feature set.
    
    Args:
        name: Feature set name
        request: Feature set request
        
    Returns:
        Feature set
    """
    try:
        return await feature_store_service.update_feature_set(
            name,
            request.description,
            request.features
        )
    except Exception as e:
        logger.error(f"Error updating feature set {name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/feature-sets/{name}", response_model=FeatureSet)
async def delete_feature_set(name: str = Path(..., description="Feature set name")):
    """
    Delete feature set.
    
    Args:
        name: Feature set name
        
    Returns:
        Feature set
    """
    try:
        return await feature_store_service.delete_feature_set(name)
    except Exception as e:
        logger.error(f"Error deleting feature set {name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))