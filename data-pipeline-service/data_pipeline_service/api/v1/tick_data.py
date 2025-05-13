"""
API endpoints for tick data operations.
"""
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from core_foundations.utils.logger import get_logger
from data_pipeline_service.db.engine import get_db_session
from data_pipeline_service.models.schemas import PaginatedResponse, TickData
from data_pipeline_service.repositories.tick_data_repository import TickDataRepository
from data_pipeline_service.services.tick_data_service import TickDataService
logger = get_logger('tick-data-api')
router = APIRouter(prefix='/tick', tags=['Tick Data'])


from data_pipeline_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@router.get('', response_model=PaginatedResponse, summary='Get tick data',
    description=
    'Retrieve tick data for a specific instrument within a time range.')
@async_with_exception_handling
async def get_tick_data(symbol: str=Query(..., description=
    'Trading instrument symbol'), from_time: datetime=Query(...,
    description='Start time for data query (ISO format)'), to_time:
    datetime=Query(..., description='End time for data query (ISO format)'),
    limit: int=Query(10000, description='Maximum number of ticks to return'
    ), page: int=Query(1, description='Page number for pagination'),
    page_size: int=Query(1000, description='Number of items per page'),
    db_session: AsyncSession=Depends(get_db_session)):
    """
    Retrieve tick data for a specific instrument within a time range.
    Data is returned in pages for efficient transmission of large datasets.
    """
    try:
        if from_time.tzinfo is None:
            from_time = from_time.replace(tzinfo=timezone.utc)
        if to_time.tzinfo is None:
            to_time = to_time.replace(tzinfo=timezone.utc)
        repository = TickDataRepository(db_session)
        service = TickDataService(repository)
        result = await service.get_tick_data(symbol=symbol, from_time=
            from_time, to_time=to_time, limit=limit, page=page, page_size=
            page_size)
        return result
    except Exception as e:
        logger.error(f'Error retrieving tick data: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/validate', response_model=bool, summary='Validate tick data',
    description='Validate a batch of tick data against data quality rules.')
@async_with_exception_handling
async def validate_tick_data(data: List[TickData], db_session: AsyncSession
    =Depends(get_db_session)):
    """
    Validate a batch of tick data against data quality rules.
    Returns True if all data passes validation, False otherwise.
    """
    try:
        repository = TickDataRepository(db_session)
        service = TickDataService(repository)
        result = service.validate_tick_data(data)
        return result
    except Exception as e:
        logger.error(f'Error validating tick data: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


@router.post('', response_model=int, summary='Store tick data', description
    ='Store a batch of tick data in the database.')
@async_with_exception_handling
async def store_tick_data(data: List[TickData], db_session: AsyncSession=
    Depends(get_db_session)):
    """
    Store a batch of tick data in the database.
    Returns the number of records stored.
    """
    try:
        repository = TickDataRepository(db_session)
        service = TickDataService(repository)
        count = await service.store_tick_data(data)
        return count
    except Exception as e:
        logger.error(f'Error storing tick data: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/window', response_model=PaginatedResponse, summary=
    'Get tick data for a time window', description=
    'Retrieve tick data for a specific instrument for a time window from now.')
@async_with_exception_handling
async def get_tick_data_window(symbol: str=Query(..., description=
    'Trading instrument symbol'), window: str=Query('1h', description=
    "Time window (e.g., '1h', '1d')"), limit: int=Query(10000, description=
    'Maximum number of ticks to return'), page: int=Query(1, description=
    'Page number for pagination'), page_size: int=Query(1000, description=
    'Number of items per page'), db_session: AsyncSession=Depends(
    get_db_session)):
    """
    Retrieve tick data for a specific instrument for a time window from now.
    For example, get the last 1 hour or 1 day of tick data.
    """
    try:
        repository = TickDataRepository(db_session)
        service = TickDataService(repository)
        now = datetime.now(timezone.utc)
        time_delta = service.get_time_window_delta(window)
        from_time = now - time_delta
        result = await service.get_tick_data(symbol=symbol, from_time=
            from_time, to_time=now, limit=limit, page=page, page_size=page_size
            )
        return result
    except Exception as e:
        logger.error(f'Error retrieving tick data for window: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))
