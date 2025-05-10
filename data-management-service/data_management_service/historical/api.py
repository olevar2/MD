"""
Historical Data Management API.

This module provides the API endpoints for the Historical Data Management system.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from data_management_service.historical.models import (
    HistoricalDataRecord,
    HistoricalOHLCVRecord,
    HistoricalTickRecord,
    HistoricalAlternativeRecord,
    DataCorrectionRecord,
    DataQualityReport,
    DataSourceType,
    CorrectionType,
    MLDatasetConfig,
    HistoricalDataQuery
)
from data_management_service.historical.repository import HistoricalDataRepository
from data_management_service.historical.service import HistoricalDataService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/historical", tags=["historical"])


# Dependency for database connection
async def get_db_engine() -> AsyncEngine:
    """Get database engine."""
    # This should be configured from environment variables or config file
    connection_string = "postgresql+asyncpg://postgres:postgres@localhost:5432/forex_platform"
    return create_async_engine(connection_string)


# Dependency for repository
async def get_repository(engine: AsyncEngine = Depends(get_db_engine)) -> HistoricalDataRepository:
    """Get repository."""
    repository = HistoricalDataRepository(engine)
    await repository.initialize()
    return repository


# Dependency for service
async def get_service(repository: HistoricalDataRepository = Depends(get_repository)) -> HistoricalDataService:
    """Get service."""
    service = HistoricalDataService(repository)
    await service.initialize()
    return service


# Request/Response models
class OHLCVDataRequest(BaseModel):
    """Request model for storing OHLCV data."""
    symbol: str
    timeframe: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    source_id: str
    metadata: Optional[Dict[str, Any]] = None
    created_by: Optional[str] = None


class TickDataRequest(BaseModel):
    """Request model for storing tick data."""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    bid_volume: Optional[float] = None
    ask_volume: Optional[float] = None
    source_id: str = "default"
    metadata: Optional[Dict[str, Any]] = None
    created_by: Optional[str] = None


class AlternativeDataRequest(BaseModel):
    """Request model for storing alternative data."""
    symbol: str
    timestamp: datetime
    data_type: str
    data: Dict[str, Any]
    source_id: str = "default"
    metadata: Optional[Dict[str, Any]] = None
    created_by: Optional[str] = None


class CorrectionRequest(BaseModel):
    """Request model for creating a correction."""
    original_record_id: str
    correction_data: Dict[str, Any]
    correction_type: CorrectionType
    correction_reason: str
    corrected_by: str
    source_type: DataSourceType


class RecordResponse(BaseModel):
    """Response model for record ID."""
    record_id: str


class CorrectionResponse(BaseModel):
    """Response model for correction."""
    corrected_record_id: str
    correction_id: str


class QualityReportRequest(BaseModel):
    """Request model for generating a quality report."""
    symbol: str
    source_type: DataSourceType
    timeframe: Optional[str] = None
    start_timestamp: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None


class QualityReportResponse(BaseModel):
    """Response model for quality report."""
    report_id: str


# API endpoints
@router.post("/ohlcv", response_model=RecordResponse)
async def store_ohlcv_data(
    request: OHLCVDataRequest,
    service: HistoricalDataService = Depends(get_service)
) -> RecordResponse:
    """Store OHLCV data."""
    try:
        record_id = await service.store_ohlcv_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            timestamp=request.timestamp,
            open_price=request.open_price,
            high_price=request.high_price,
            low_price=request.low_price,
            close_price=request.close_price,
            volume=request.volume,
            source_id=request.source_id,
            metadata=request.metadata,
            created_by=request.created_by
        )

        return RecordResponse(record_id=record_id)
    except Exception as e:
        logger.error(f"Failed to store OHLCV data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tick", response_model=RecordResponse)
async def store_tick_data(
    request: TickDataRequest,
    service: HistoricalDataService = Depends(get_service)
) -> RecordResponse:
    """Store tick data."""
    try:
        record_id = await service.store_tick_data(
            symbol=request.symbol,
            timestamp=request.timestamp,
            bid=request.bid,
            ask=request.ask,
            bid_volume=request.bid_volume,
            ask_volume=request.ask_volume,
            source_id=request.source_id,
            metadata=request.metadata,
            created_by=request.created_by
        )

        return RecordResponse(record_id=record_id)
    except Exception as e:
        logger.error(f"Failed to store tick data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alternative", response_model=RecordResponse)
async def store_alternative_data(
    request: AlternativeDataRequest,
    service: HistoricalDataService = Depends(get_service)
) -> RecordResponse:
    """Store alternative data."""
    try:
        record_id = await service.store_alternative_data(
            symbol=request.symbol,
            timestamp=request.timestamp,
            data_type=request.data_type,
            data=request.data,
            source_id=request.source_id,
            metadata=request.metadata,
            created_by=request.created_by
        )

        return RecordResponse(record_id=record_id)
    except Exception as e:
        logger.error(f"Failed to store alternative data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/correction", response_model=CorrectionResponse)
async def create_correction(
    request: CorrectionRequest,
    service: HistoricalDataService = Depends(get_service)
) -> CorrectionResponse:
    """Create a correction for an existing record."""
    try:
        corrected_record_id, correction_id = await service.create_correction(
            original_record_id=request.original_record_id,
            correction_data=request.correction_data,
            correction_type=request.correction_type,
            correction_reason=request.correction_reason,
            corrected_by=request.corrected_by,
            source_type=request.source_type
        )

        return CorrectionResponse(
            corrected_record_id=corrected_record_id,
            correction_id=correction_id
        )
    except Exception as e:
        logger.error(f"Failed to create correction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quality-report", response_model=QualityReportResponse)
async def generate_quality_report(
    request: QualityReportRequest,
    service: HistoricalDataService = Depends(get_service)
) -> QualityReportResponse:
    """Generate a data quality report."""
    try:
        report_id = await service.generate_quality_report(
            symbol=request.symbol,
            source_type=request.source_type,
            timeframe=request.timeframe,
            start_timestamp=request.start_timestamp,
            end_timestamp=request.end_timestamp
        )

        return QualityReportResponse(report_id=report_id)
    except Exception as e:
        logger.error(f"Failed to generate quality report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ohlcv")
async def get_ohlcv_data(
    symbols: str = Query(..., description="Comma-separated list of symbols"),
    timeframe: str = Query(..., description="Timeframe"),
    start_timestamp: datetime = Query(..., description="Start timestamp"),
    end_timestamp: datetime = Query(..., description="End timestamp"),
    version: Optional[int] = Query(None, description="Specific version to retrieve"),
    point_in_time: Optional[datetime] = Query(None, description="Point-in-time for historical accuracy"),
    include_corrections: bool = Query(True, description="Whether to include corrections"),
    format: str = Query("json", description="Response format (json or csv)"),
    service: HistoricalDataService = Depends(get_service)
):
    """Get OHLCV data."""
    try:
        # Parse symbols
        symbol_list = [s.strip() for s in symbols.split(",")]

        # Get data
        df = await service.get_ohlcv_data(
            symbols=symbol_list,
            timeframe=timeframe,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            version=version,
            point_in_time=point_in_time,
            include_corrections=include_corrections
        )

        # Convert to desired format
        if format.lower() == "csv":
            return df.reset_index().to_csv(index=False)
        else:
            return df.reset_index().to_dict(orient="records")
    except Exception as e:
        logger.error(f"Failed to get OHLCV data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tick")
async def get_tick_data(
    symbols: str = Query(..., description="Comma-separated list of symbols"),
    start_timestamp: datetime = Query(..., description="Start timestamp"),
    end_timestamp: datetime = Query(..., description="End timestamp"),
    version: Optional[int] = Query(None, description="Specific version to retrieve"),
    point_in_time: Optional[datetime] = Query(None, description="Point-in-time for historical accuracy"),
    include_corrections: bool = Query(True, description="Whether to include corrections"),
    format: str = Query("json", description="Response format (json or csv)"),
    service: HistoricalDataService = Depends(get_service)
):
    """Get tick data."""
    try:
        # Parse symbols
        symbol_list = [s.strip() for s in symbols.split(",")]

        # Get data
        df = await service.get_tick_data(
            symbols=symbol_list,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            version=version,
            point_in_time=point_in_time,
            include_corrections=include_corrections
        )

        # Convert to desired format
        if format.lower() == "csv":
            return df.reset_index().to_csv(index=False)
        else:
            return df.reset_index().to_dict(orient="records")
    except Exception as e:
        logger.error(f"Failed to get tick data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alternative")
async def get_alternative_data(
    symbols: str = Query(..., description="Comma-separated list of symbols"),
    data_type: str = Query(..., description="Type of alternative data"),
    start_timestamp: datetime = Query(..., description="Start timestamp"),
    end_timestamp: datetime = Query(..., description="End timestamp"),
    version: Optional[int] = Query(None, description="Specific version to retrieve"),
    point_in_time: Optional[datetime] = Query(None, description="Point-in-time for historical accuracy"),
    include_corrections: bool = Query(True, description="Whether to include corrections"),
    format: str = Query("json", description="Response format (json or csv)"),
    service: HistoricalDataService = Depends(get_service)
):
    """Get alternative data."""
    try:
        # Parse symbols
        symbol_list = [s.strip() for s in symbols.split(",")]

        # Get data
        df = await service.get_alternative_data(
            symbols=symbol_list,
            data_type=data_type,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            version=version,
            point_in_time=point_in_time,
            include_corrections=include_corrections
        )

        # Convert to desired format
        if format.lower() == "csv":
            return df.reset_index().to_csv(index=False)
        else:
            return df.reset_index().to_dict(orient="records")
    except Exception as e:
        logger.error(f"Failed to get alternative data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/record-history/{record_id}")
async def get_record_history(
    record_id: str,
    source_type: DataSourceType,
    service: HistoricalDataService = Depends(get_service)
):
    """Get the history of a record, including all corrections."""
    try:
        history = await service.get_record_history(
            record_id=record_id,
            source_type=source_type
        )

        return history
    except Exception as e:
        logger.error(f"Failed to get record history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ml-dataset")
async def create_ml_dataset(
    config: MLDatasetConfig,
    format: str = Query("json", description="Response format (json or csv)"),
    service: HistoricalDataService = Depends(get_service)
):
    """Create a dataset for machine learning."""
    try:
        df = await service.create_ml_dataset(config=config)

        # Convert to desired format
        if format.lower() == "csv":
            return df.to_csv(index=False)
        else:
            return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Failed to create ML dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/point-in-time")
async def get_point_in_time_data(
    query: HistoricalDataQuery,
    format: str = Query("json", description="Response format (json or csv)"),
    service: HistoricalDataService = Depends(get_service)
):
    """Get point-in-time accurate data."""
    try:
        df = await service.get_point_in_time_data(query=query)

        # Convert to desired format
        if format.lower() == "csv":
            return df.reset_index().to_csv(index=False)
        else:
            return df.reset_index().to_dict(orient="records")
    except Exception as e:
        logger.error(f"Failed to get point-in-time data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
