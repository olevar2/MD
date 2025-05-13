"""
API endpoints for data source adapters.
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from core_foundations.utils.logger import get_logger
from models.schemas import OHLCVData, PaginatedResponse, TickData
from core.data_fetcher_manager import DataFetcherManager
from adapters.dukascopy_adapter import DukascopyAdapter
DESC_TRADING_SYMBOL = 'Trading instrument symbol'
DESC_ADAPTER_ID = 'Specific adapter to use, or None for default'
logger = get_logger('adapters-api')
router = APIRouter(prefix='/adapters', tags=['Data Source Adapters'])
data_fetcher_manager = DataFetcherManager()
dukascopy_adapter = DukascopyAdapter()
data_fetcher_manager.register_adapter('dukascopy', dukascopy_adapter,
    set_as_default=True)


from data_pipeline_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class AdapterStatus(BaseModel):
    """Status information for a data source adapter."""
    adapter_id: str = Field(..., description=
        'Unique identifier for the adapter')
    status: str = Field(..., description=
        'Connection status (connected/disconnected/error)')
    type: str = Field(..., description='Adapter type')
    is_default_ohlcv: bool = Field(False, description=
        'Whether this is the default OHLCV adapter')
    is_default_tick: bool = Field(False, description=
        'Whether this is the default tick data adapter')
    message: Optional[str] = Field(None, description=
        'Error message if status is error')


class InstrumentInfo(BaseModel):
    """Information about a trading instrument."""
    symbol: str = Field(..., description='Trading instrument symbol')
    name: str = Field(..., description='Instrument name')
    type: str = Field(..., description='Instrument type (forex, crypto, etc.)')
    pip_size: float = Field(..., description='Size of one pip')
    source_symbol: Optional[str] = Field(None, description=
        'Symbol used by the data source')


@router.get('', response_model=Dict[str, AdapterStatus], summary=
    'Get adapter status', description=
    'Get connection status for all registered data adapters.')
@async_with_exception_handling
async def get_adapters_status():
    """
    Get connection status for all registered data adapters.
    """
    try:
        status_dict = await data_fetcher_manager.health_check()
        return {adapter_id: AdapterStatus(adapter_id=adapter_id, **status) for
            adapter_id, status in status_dict.items()}
    except Exception as e:
        logger.error(f'Error checking adapter health: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/connect', response_model=Dict[str, bool], summary=
    'Connect to all adapters', description=
    'Connect to all registered data source adapters.')
@async_with_exception_handling
async def connect_adapters():
    """
    Connect to all registered data source adapters.
    """
    try:
        results = await data_fetcher_manager.connect_all()
        return results
    except Exception as e:
        logger.error(f'Error connecting to adapters: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/disconnect', response_model=Dict[str, str], summary=
    'Disconnect from all adapters', description=
    'Disconnect from all registered data source adapters.')
@async_with_exception_handling
async def disconnect_adapters():
    """
    Disconnect from all registered data source adapters.
    """
    try:
        await data_fetcher_manager.disconnect_all()
        return {'status': 'Disconnected from all adapters'}
    except Exception as e:
        logger.error(f'Error disconnecting from adapters: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/instruments', response_model=List[InstrumentInfo], summary=
    'Get available instruments', description=
    'Get list of available instruments from a data source.')
@async_with_exception_handling
async def get_instruments(adapter_id: Optional[str]=Query(None, description
    =DESC_ADAPTER_ID)):
    """
    Get list of available instruments from a data source.
    """
    try:
        instruments = await data_fetcher_manager.get_available_instruments(
            adapter_id)
        return [InstrumentInfo(**instrument) for instrument in instruments]
    except Exception as e:
        logger.error(f'Error fetching instruments: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/tick', response_model=List[TickData], summary='Get tick data',
    description='Get tick data from a data source.')
@async_with_exception_handling
async def get_tick_data(symbol: str=Query(..., description=
    DESC_TRADING_SYMBOL), from_time: datetime=Query(..., description=
    'Start time for data query (ISO format)'), to_time: datetime=Query(...,
    description='End time for data query (ISO format)'), limit: Optional[
    int]=Query(10000, description='Maximum number of ticks to return'),
    adapter_id: Optional[str]=Query(None, description=DESC_ADAPTER_ID)):
    """
    Get tick data from a data source.
    """
    try:
        if from_time.tzinfo is None:
            from_time = from_time.replace(tzinfo=timezone.utc)
        if to_time.tzinfo is None:
            to_time = to_time.replace(tzinfo=timezone.utc)
        ticks = await data_fetcher_manager.get_tick_data(symbol, from_time,
            to_time, limit, adapter_id)
        return ticks
    except Exception as e:
        logger.error(f'Error fetching tick data: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/ohlcv', response_model=List[OHLCVData], summary=
    'Get OHLCV data', description='Get OHLCV data from a data source.')
@async_with_exception_handling
async def get_ohlcv_data(symbol: str=Query(..., description=
    DESC_TRADING_SYMBOL), timeframe: str=Query(..., description=
    'Candle timeframe'), from_time: datetime=Query(..., description=
    'Start time for data query (ISO format)'), to_time: datetime=Query(...,
    description='End time for data query (ISO format)'), limit: Optional[
    int]=Query(1000, description='Maximum number of candles to return'),
    adapter_id: Optional[str]=Query(None, description=DESC_ADAPTER_ID)):
    """
    Get OHLCV data from a data source.
    """
    try:
        if from_time.tzinfo is None:
            from_time = from_time.replace(tzinfo=timezone.utc)
        if to_time.tzinfo is None:
            to_time = to_time.replace(tzinfo=timezone.utc)
        candles = await data_fetcher_manager.get_ohlcv_data(symbol,
            timeframe, from_time, to_time, limit, adapter_id)
        return candles
    except Exception as e:
        logger.error(f'Error fetching OHLCV data: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))
