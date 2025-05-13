"""
API endpoints for OHLCV data access.
"""
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging_setup_standardized import get_service_logger
from core.database_standardized import database
from models.schemas import (
    OHLCVData,
    OHLCVRequest,
    PaginatedResponse,
    TimeFrame
)
from repositories.ohlcv_repository import OHLCVRepository
from services.ohlcv_service import OHLCVService
from api.error_handling_standardized import handle_async_exception

# Initialize logger
logger = get_service_logger("ohlcv-api")

# Create API router
router = APIRouter()


# Dependencies
async def get_db_session() -> AsyncSession:
    """Get database session."""
    return await database.get_session()


# Endpoints
@router.get(
    "",
    response_model=PaginatedResponse,
    summary="Get historical OHLCV data",
    description="Retrieve historical OHLCV (candlestick) data for a specific instrument and timeframe.",
)
@handle_async_exception(operation="get_ohlcv_data")
async def get_ohlcv_data(
    symbol: str = Query(..., description="Instrument symbol (e.g., 'EUR/USD')"),
    timeframe: TimeFrame = Query(..., description="Candle timeframe"),
    from_time: Optional[datetime] = Query(None, description="Start time in UTC (ISO format)"),
    to_time: Optional[datetime] = Query(None, description="End time in UTC (ISO format)"),
    limit: int = Query(1000, description="Maximum number of candles to return", le=10000),
    page: int = Query(1, description="Page number", ge=1),
    page_size: int = Query(100, description="Page size", ge=1, le=1000),
    db_session: AsyncSession = Depends(get_db_session),
):
    """
    Get historical OHLCV (Open, High, Low, Close, Volume) data for a specific
    instrument and timeframe.
    """
    # Set default times if not provided
    if not to_time:
        to_time = datetime.now(timezone.utc)
    if not from_time:
        # Default to 1000 candles based on timeframe
        from_time = to_time - OHLCVService.get_timeframe_delta(timeframe.value, limit)

    repository = OHLCVRepository(db_session)
    service = OHLCVService(repository)

    result = await service.get_ohlcv_data(
        symbol=symbol,
        timeframe=timeframe.value,
        from_time=from_time,
        to_time=to_time,
        limit=limit,
        page=page,
        page_size=page_size,
    )

    return result


@router.post(
    "/validate",
    response_model=bool,
    summary="Validate OHLCV data",
    description="Validate a batch of OHLCV data against data quality rules.",
)
@handle_async_exception(operation="validate_ohlcv_data")
async def validate_ohlcv_data(
    data: List[OHLCVData],
    db_session: AsyncSession = Depends(get_db_session),
):
    """
    Validate a batch of OHLCV data against data quality rules.
    Returns True if all data passes validation, False otherwise.
    """
    repository = OHLCVRepository(db_session)
    service = OHLCVService(repository)

    result = await service.validate_ohlcv_data(data)
    return result


@router.post(
    "",
    response_model=int,
    summary="Store OHLCV data",
    description="Store a batch of OHLCV data in the database.",
)
@handle_async_exception(operation="store_ohlcv_data")
async def store_ohlcv_data(
    data: List[OHLCVData],
    db_session: AsyncSession = Depends(get_db_session),
):
    """
    Store a batch of OHLCV data in the database.
    Returns the number of records stored.
    """
    repository = OHLCVRepository(db_session)
    service = OHLCVService(repository)

    count = await service.store_ohlcv_data(data)
    return count