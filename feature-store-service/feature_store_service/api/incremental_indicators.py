"""
Incremental Indicators API.

This module provides API endpoints for the incremental indicator functionality,
enabling efficient updates and queries for low-latency applications.
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from datetime import datetime

from core_foundations.utils.logger import get_logger
from feature_store_service.services.indicator_manager import get_indicator_manager
from feature_store_service.computation.incremental.indicator_service import IncrementalIndicatorService

logger = get_logger("feature-store-service.incremental-indicators-api")

router = APIRouter(
    prefix="/api/v1/incremental-indicators",
    tags=["incremental-indicators"],
    responses={404: {"description": "Not found"}},
)


class DataPoint(BaseModel):
    """Model for an OHLCV data point."""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe of the data point")
    timestamp: datetime = Field(..., description="Timestamp of the data point")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="Highest price")
    low: float = Field(..., description="Lowest price")
    close: float = Field(..., description="Closing price")
    volume: float = Field(..., description="Trading volume")


class IndicatorParams(BaseModel):
    """Model for indicator parameters."""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe of the indicator")
    indicator_type: str = Field(..., description="Type of indicator (e.g., 'SMA', 'EMA', 'MACD')")
    params: Optional[Dict[str, Any]] = Field(None, description="Parameters for the indicator")
    lookback_days: int = Field(60, description="Number of days of history to use for initialization")


class IndicatorResponse(BaseModel):
    """Model for indicator response."""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe of the indicator")
    indicator_type: str = Field(..., description="Type of indicator")
    values: Dict[str, float] = Field(..., description="Indicator values")
    timestamp: datetime = Field(..., description="Timestamp of the calculation")


class IndicatorInfo(BaseModel):
    """Model for indicator information."""
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe of the indicator")
    type: str = Field(..., description="Type of indicator")
    params: Dict[str, Any] = Field(..., description="Parameters for the indicator")
    is_initialized: bool = Field(..., description="Whether the indicator is initialized")
    last_updated: Optional[datetime] = Field(None, description="Timestamp of the last update")


@router.post("/initialize", response_model=Dict[str, bool], summary="Initialize an incremental indicator")
async def initialize_indicator(
    params: IndicatorParams,
    indicator_service: IncrementalIndicatorService = Depends(get_indicator_manager)
):
    """
    Initialize an incremental indicator with historical data.
    
    This endpoint creates and initializes a stateful indicator that can be
    efficiently updated with new data points.
    """
    try:
        indicator = await indicator_service.get_or_initialize_indicator(
            symbol=params.symbol,
            timeframe=params.timeframe,
            indicator_type=params.indicator_type,
            params=params.params,
            lookback_days=params.lookback_days
        )
        
        if indicator and indicator.is_initialized:
            return {"success": True, "message": f"Initialized {params.indicator_type} for {params.symbol} {params.timeframe}"}
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize {params.indicator_type} for {params.symbol} {params.timeframe}"
            )
    except Exception as e:
        logger.error(f"Error initializing indicator: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error initializing indicator: {str(e)}"
        )


@router.post("/update", response_model=IndicatorResponse, summary="Update an incremental indicator")
async def update_indicator(
    data_point: DataPoint,
    indicator_type: str = Query(..., description="Type of indicator to update"),
    params: Optional[Dict[str, Any]] = None,
    indicator_service: IncrementalIndicatorService = Depends(get_indicator_manager)
):
    """
    Update an incremental indicator with a new data point.
    
    This endpoint efficiently updates a stateful indicator with a new data point
    without recalculating the entire time series.
    """
    try:
        # Convert data point to dictionary
        data_dict = {
            "timestamp": data_point.timestamp,
            "open": data_point.open,
            "high": data_point.high,
            "low": data_point.low,
            "close": data_point.close,
            "volume": data_point.volume
        }
        
        # Update the indicator
        result = await indicator_service.update_indicator(
            symbol=data_point.symbol,
            timeframe=data_point.timeframe,
            indicator_type=indicator_type,
            new_data_point=data_dict,
            params=params
        )
        
        if result:
            return IndicatorResponse(
                symbol=data_point.symbol,
                timeframe=data_point.timeframe,
                indicator_type=indicator_type,
                values=result,
                timestamp=data_point.timestamp
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to update {indicator_type} for {data_point.symbol} {data_point.timeframe}"
            )
    except Exception as e:
        logger.error(f"Error updating indicator: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating indicator: {str(e)}"
        )


@router.post("/update-all", response_model=Dict[str, Dict[str, float]], summary="Update all indicators for a symbol")
async def update_all_indicators(
    data_point: DataPoint,
    indicator_service: IncrementalIndicatorService = Depends(get_indicator_manager)
):
    """
    Update all active indicators for a specific symbol and timeframe.
    
    This endpoint efficiently updates all stateful indicators for a symbol and timeframe
    with a new data point.
    """
    try:
        # Convert data point to dictionary
        data_dict = {
            "timestamp": data_point.timestamp,
            "open": data_point.open,
            "high": data_point.high,
            "low": data_point.low,
            "close": data_point.close,
            "volume": data_point.volume
        }
        
        # Update all indicators
        result = await indicator_service.update_all_indicators_for_symbol(
            symbol=data_point.symbol,
            timeframe=data_point.timeframe,
            new_data_point=data_dict
        )
        
        return result
    except Exception as e:
        logger.error(f"Error updating indicators: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating indicators: {str(e)}"
        )


@router.get("/active", response_model=Dict[str, IndicatorInfo], summary="Get all active indicators")
async def get_active_indicators(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    timeframe: Optional[str] = Query(None, description="Filter by timeframe"),
    indicator_type: Optional[str] = Query(None, description="Filter by indicator type"),
    indicator_service: IncrementalIndicatorService = Depends(get_indicator_manager)
):
    """
    Get information about all active indicators.
    
    This endpoint returns information about all active stateful indicators,
    optionally filtered by symbol, timeframe, or indicator type.
    """
    try:
        # Get all active indicators
        all_indicators = indicator_service.get_all_active_indicators()
        
        # Apply filters if provided
        if symbol or timeframe or indicator_type:
            filtered_indicators = {}
            for key, info in all_indicators.items():
                if symbol and info["symbol"] != symbol:
                    continue
                if timeframe and info["timeframe"] != timeframe:
                    continue
                if indicator_type and info["type"] != indicator_type:
                    continue
                    
                filtered_indicators[key] = info
                
            return filtered_indicators
        else:
            return all_indicators
    except Exception as e:
        logger.error(f"Error getting active indicators: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting active indicators: {str(e)}"
        )


@router.post("/load-states", response_model=Dict[str, int], summary="Load saved indicator states")
async def load_saved_states(
    indicator_service: IncrementalIndicatorService = Depends(get_indicator_manager)
):
    """
    Load all saved indicator states from persistent storage.
    
    This endpoint loads the saved states of all indicators from persistent storage.
    """
    try:
        count = await indicator_service.load_all_saved_states()
        return {"loaded_indicators": count}
    except Exception as e:
        logger.error(f"Error loading indicator states: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error loading indicator states: {str(e)}"
        )