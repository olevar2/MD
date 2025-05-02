"""
API endpoints for market instruments.
"""
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from core_foundations.utils.logger import get_logger
from data_pipeline_service.models.schemas import (
    Instrument,
    InstrumentType,
    PaginatedResponse,
    TradingHours,
)
from data_pipeline_service.repositories.instrument_repository import InstrumentRepository
from data_pipeline_service.services.instrument_service import InstrumentService

# Initialize logger
logger = get_logger("data-pipeline-service")

# Create API router
router = APIRouter()


# Dependencies
async def get_db_session() -> AsyncSession:
    """Get database session."""
    # This is a placeholder. In production, this would use app state
    # to access the engine and create a session.
    raise NotImplementedError("Database session dependency not implemented")


# Endpoints
@router.get(
    "",
    response_model=PaginatedResponse,
    summary="Get available instruments",
    description="Retrieve a list of available trading instruments.",
)
async def get_instruments(
    type: Optional[InstrumentType] = Query(None, description="Filter by instrument type"),
    search: Optional[str] = Query(None, description="Search in symbol or name"),
    page: int = Query(1, description="Page number", ge=1),
    page_size: int = Query(100, description="Page size", ge=1, le=1000),
    db_session: AsyncSession = Depends(get_db_session),
):
    """
    Get a paginated list of available trading instruments.
    Optionally filter by type and search term.
    """
    try:
        repository = InstrumentRepository(db_session)
        service = InstrumentService(repository)
        
        result = await service.get_instruments(
            type=type.value if type else None,
            search=search,
            page=page,
            page_size=page_size,
        )
        
        return result
    except Exception as e:
        logger.error(f"Error retrieving instruments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{symbol}",
    response_model=Instrument,
    summary="Get instrument details",
    description="Retrieve details for a specific trading instrument.",
)
async def get_instrument(
    symbol: str,
    db_session: AsyncSession = Depends(get_db_session),
):
    """
    Get detailed information about a specific trading instrument.
    """
    try:
        repository = InstrumentRepository(db_session)
        service = InstrumentService(repository)
        
        result = await service.get_instrument_by_symbol(symbol)
        if not result:
            raise HTTPException(status_code=404, detail=f"Instrument {symbol} not found")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving instrument {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{symbol}/trading-hours",
    response_model=List[TradingHours],
    summary="Get instrument trading hours",
    description="Retrieve trading hours for a specific instrument.",
)
async def get_trading_hours(
    symbol: str,
    db_session: AsyncSession = Depends(get_db_session),
):
    """
    Get trading hours for a specific instrument by day of week.
    """
    try:
        repository = InstrumentRepository(db_session)
        service = InstrumentService(repository)
        
        result = await service.get_trading_hours(symbol)
        if not result:
            raise HTTPException(
                status_code=404, detail=f"Trading hours for {symbol} not found"
            )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving trading hours for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "",
    response_model=Instrument,
    summary="Create or update instrument",
    description="Create a new instrument or update an existing one.",
    status_code=201,
)
async def create_or_update_instrument(
    instrument: Instrument,
    db_session: AsyncSession = Depends(get_db_session),
):
    """
    Create a new instrument or update an existing one if the symbol already exists.
    """
    try:
        repository = InstrumentRepository(db_session)
        service = InstrumentService(repository)
        
        result = await service.create_or_update_instrument(instrument)
        return result
    except Exception as e:
        logger.error(f"Error creating/updating instrument: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/{symbol}/trading-hours",
    response_model=TradingHours,
    summary="Add trading hours",
    description="Add trading hours for a specific instrument.",
    status_code=201,
)
async def add_trading_hours(
    symbol: str,
    trading_hours: TradingHours,
    db_session: AsyncSession = Depends(get_db_session),
):
    """
    Add trading hours for a specific instrument.
    """
    if trading_hours.symbol != symbol:
        raise HTTPException(
            status_code=400, detail="Symbol in path must match symbol in body"
        )
    
    try:
        repository = InstrumentRepository(db_session)
        service = InstrumentService(repository)
        
        # First check if instrument exists
        instrument = await service.get_instrument_by_symbol(symbol)
        if not instrument:
            raise HTTPException(status_code=404, detail=f"Instrument {symbol} not found")
        
        result = await service.add_trading_hours(trading_hours)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding trading hours for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))