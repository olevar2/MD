"""
API endpoints for data cleaning operations.
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field

from core_foundations.utils.logger import get_logger
from data_pipeline_service.cleaning.cleaning_engine import DataCleaningEngine, DataType
from data_pipeline_service.models.schemas import OHLCVData, TickData
from data_pipeline_service.repositories.ohlcv_repository import OHLCVRepository
from data_pipeline_service.repositories.tick_data_repository import TickDataRepository

# Initialize logger
logger = get_logger("cleaning-api")

# Create router
router = APIRouter()

# Initialize cleaning engine
cleaning_engine = DataCleaningEngine()

# Initialize repositories
ohlcv_repository = OHLCVRepository()
tick_repository = TickDataRepository()


# Models for API
class CleaningReport(BaseModel):
    """Report of a data cleaning operation."""
    
    data_type: str = Field(..., description="Type of data cleaned")
    rows_processed: int = Field(..., description="Number of rows processed")
    missing_values_imputed: int = Field(..., description="Number of missing values imputed")
    outliers_detected: int = Field(..., description="Number of outliers detected")
    outliers_treated: int = Field(..., description="Number of outliers treated")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")


class CleaningRequest(BaseModel):
    """Request for a data cleaning operation."""
    
    data: List[Dict] = Field(..., description="Data to clean")
    data_type: str = Field(..., description="Type of data (ohlcv, tick, or generic)")
    columns: Optional[List[str]] = Field(None, description="Columns to clean (optional)")


class CleaningResponse(BaseModel):
    """Response from a data cleaning operation."""
    
    cleaned_data: List[Dict] = Field(..., description="Cleaned data")
    report: CleaningReport = Field(..., description="Cleaning report")


@router.post(
    "/clean",
    response_model=CleaningResponse,
    summary="Clean data",
    description="Clean data by imputing missing values and handling outliers.",
)
async def clean_data(request: CleaningRequest = Body(...)):
    """
    Clean data by imputing missing values and handling outliers.
    """
    try:
        import time
        start_time = time.time()
        
        # Convert data_type string to enum
        data_type_str = request.data_type.lower()
        if data_type_str == "ohlcv":
            data_type = DataType.OHLCV
        elif data_type_str == "tick":
            data_type = DataType.TICK
        else:
            data_type = DataType.GENERIC
        
        # Count initial missing values
        import pandas as pd
        df = pd.DataFrame(request.data)
        columns_to_check = request.columns or df.columns.tolist()
        missing_values_before = df[columns_to_check].isna().sum().sum()
        
        # Clean the data based on type
        if data_type == DataType.OHLCV:
            cleaned_data = cleaning_engine.clean_ohlcv_data(request.data)
        elif data_type == DataType.TICK:
            cleaned_data = cleaning_engine.clean_tick_data(request.data)
        else:
            cleaned_data = cleaning_engine.clean_data(
                request.data, data_type, request.columns
            )
        
        # Count missing values after cleaning
        df_after = pd.DataFrame(cleaned_data)
        missing_values_after = df_after[columns_to_check].isna().sum().sum()
        missing_values_imputed = missing_values_before - missing_values_after
        
        # Create report
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000
        
        report = CleaningReport(
            data_type=request.data_type,
            rows_processed=len(request.data),
            missing_values_imputed=int(missing_values_imputed),
            outliers_detected=0,  # Placeholder, actual value difficult to track
            outliers_treated=0,   # Placeholder, actual value difficult to track
            execution_time_ms=execution_time_ms
        )
        
        return CleaningResponse(cleaned_data=cleaned_data, report=report)
    
    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/clean/ohlcv",
    response_model=List[OHLCVData],
    summary="Clean OHLCV data",
    description="Clean OHLCV data by imputing missing values and handling outliers.",
)
async def clean_ohlcv_data(
    symbol: str = Query(..., description="Trading instrument symbol"),
    timeframe: str = Query(..., description="Candle timeframe"),
    from_time: datetime = Query(..., description="Start time for data query (ISO format)"),
    to_time: datetime = Query(..., description="End time for data query (ISO format)"),
    limit: Optional[int] = Query(1000, description="Maximum number of candles to return"),
):
    """
    Clean OHLCV data by imputing missing values and handling outliers.
    """
    try:
        # Ensure timezone-aware datetimes
        if from_time.tzinfo is None:
            from_time = from_time.replace(tzinfo=timezone.utc)
        if to_time.tzinfo is None:
            to_time = to_time.replace(tzinfo=timezone.utc)
        
        # Fetch data from repository
        candles = await ohlcv_repository.get_ohlcv_data(
            symbol, timeframe, from_time, to_time, limit
        )
        
        if not candles:
            return []
        
        # Clean the data
        cleaned_candles = cleaning_engine.clean_ohlcv_data(candles)
        
        return cleaned_candles
    
    except Exception as e:
        logger.error(f"Error cleaning OHLCV data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/clean/tick",
    response_model=List[TickData],
    summary="Clean tick data",
    description="Clean tick data by imputing missing values and handling outliers.",
)
async def clean_tick_data(
    symbol: str = Query(..., description="Trading instrument symbol"),
    from_time: datetime = Query(..., description="Start time for data query (ISO format)"),
    to_time: datetime = Query(..., description="End time for data query (ISO format)"),
    limit: Optional[int] = Query(10000, description="Maximum number of ticks to return"),
):
    """
    Clean tick data by imputing missing values and handling outliers.
    """
    try:
        # Ensure timezone-aware datetimes
        if from_time.tzinfo is None:
            from_time = from_time.replace(tzinfo=timezone.utc)
        if to_time.tzinfo is None:
            to_time = to_time.replace(tzinfo=timezone.utc)
        
        # Fetch data from repository
        ticks = await tick_repository.get_tick_data(
            symbol, from_time, to_time, limit
        )
        
        if not ticks:
            return []
        
        # Clean the data
        cleaned_ticks = cleaning_engine.clean_tick_data(ticks)
        
        return cleaned_ticks
    
    except Exception as e:
        logger.error(f"Error cleaning tick data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))