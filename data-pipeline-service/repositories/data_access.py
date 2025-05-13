"""
DataAccessAPI module for retrieving historical and real-time data.

This module provides a unified interface for accessing financial data
across different instruments and timeframes with various formatting options.
"""
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Union
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Body, Depends, Path
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from core_foundations.utils.logger import get_logger
from models.schemas import OHLCVData, OHLCVResponse, OHLCVBatchRequest, TimeframeEnum, AggregationMethodEnum, ExportFormatEnum, ServiceAuth
from repositories.ohlcv_repository import OHLCVRepository
from services.ohlcv_service import OHLCVService
from core.timeseries_aggregator import TimeseriesAggregator
from services.export_service import convert_to_csv, convert_to_parquet, format_ohlcv_for_json
from core.ohlcv_validators import validate_instrument_format, validate_timeframe, validate_date_range, is_valid_timeframe_conversion
from core.validation_exceptions import ValidationError
from core.engine import get_db_pool
from core.auth import verify_service_auth
from config.settings import get_redis_client
DESC_SYMBOL = 'Trading instrument symbol'
DESC_TIMEFRAME = 'Candle timeframe'
DESC_START_TIME = 'Start time for data query (ISO format)'
DESC_END_TIME = 'End time for data query (ISO format)'
DESC_LIMIT = 'Maximum number of candles to return'
DESC_INCLUDE_CURRENT = 'Include current incomplete candle'
DESC_SORT_ORDER = "Sorting order: 'asc' or 'desc'"
DESC_INSTRUMENT_ID = 'Trading instrument identifier (e.g., EUR_USD)'
logger = get_logger('data-access-api')
router = APIRouter()
ohlcv_repository = OHLCVRepository()
ohlcv_service = OHLCVService()


from data_pipeline_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class DataFormat(str, Enum):
    """Enum for data output formats."""
    JSON = 'json'
    DATAFRAME = 'dataframe'
    CSV = 'csv'
    NUMPY = 'numpy'


class Granularity(str, Enum):
    """Enum for data granularity/timeframes."""
    S1 = '1s'
    S5 = '5s'
    S15 = '15s'
    S30 = '30s'
    M1 = '1m'
    M5 = '5m'
    M15 = '15m'
    M30 = '30m'
    H1 = '1h'
    H4 = '4h'
    H8 = '8h'
    D1 = '1d'
    W1 = '1w'
    MN1 = '1mn'


def calculate_from_time(to_time: datetime, timeframe: str, limit: int
    ) ->datetime:
    """
    Calculate the start time based on timeframe, end time, and limit.
    
    Args:
        to_time: The end time to calculate back from
        timeframe: The timeframe as a string (e.g., "1m", "1h", "1d")
        limit: Number of candles to go back
    
    Returns:
        The calculated start time
    """
    if len(timeframe) < 2:
        raise ValueError(f'Invalid timeframe format: {timeframe}')
    if timeframe.endswith('mn'):
        value = int(timeframe[:-2])
        delta = timedelta(days=30 * value * limit)
        return to_time - delta
    value = int(timeframe[:-1])
    unit = timeframe[-1]
    if unit == 's':
        delta = timedelta(seconds=value * limit)
    elif unit == 'm':
        delta = timedelta(minutes=value * limit)
    elif unit == 'h':
        delta = timedelta(hours=value * limit)
    elif unit == 'd':
        delta = timedelta(days=value * limit)
    elif unit == 'w':
        delta = timedelta(weeks=value * limit)
    else:
        raise ValueError(f'Unsupported timeframe unit: {unit}')
    return to_time - delta


def get_period_start(time: datetime, timeframe: str) ->datetime:
    """
    Get the start of the current period based on the timeframe.
    
    Args:
        time: The reference time
        timeframe: The timeframe as a string (e.g., "1m", "1h", "1d")
    
    Returns:
        The start of the current period
    """
    if len(timeframe) < 2:
        raise ValueError(f'Invalid timeframe format: {timeframe}')
    if timeframe.endswith('mn'):
        return datetime(time.year, time.month, 1, tzinfo=time.tzinfo)
    unit = timeframe[-1]
    if unit == 's':
        value = int(timeframe[:-1])
        if value <= 0:
            raise ValueError(f'Invalid timeframe value: {timeframe}')
        seconds_since_minute = time.second
        period_start_seconds = seconds_since_minute // value * value
        return time.replace(second=period_start_seconds, microsecond=0)
    elif unit == 'm':
        value = int(timeframe[:-1])
        if value <= 0:
            raise ValueError(f'Invalid timeframe value: {timeframe}')
        minutes_since_hour = time.minute
        period_start_minutes = minutes_since_hour // value * value
        return time.replace(minute=period_start_minutes, second=0,
            microsecond=0)
    elif unit == 'h':
        value = int(timeframe[:-1])
        if value <= 0:
            raise ValueError(f'Invalid timeframe value: {timeframe}')
        hours_since_day = time.hour
        period_start_hours = hours_since_day // value * value
        return time.replace(hour=period_start_hours, minute=0, second=0,
            microsecond=0)
    elif unit == 'd':
        return time.replace(hour=0, minute=0, second=0, microsecond=0)
    elif unit == 'w':
        days_since_monday = time.weekday()
        return (time - timedelta(days=days_since_monday)).replace(hour=0,
            minute=0, second=0, microsecond=0)
    else:
        raise ValueError(f'Unsupported timeframe unit: {unit}')


class TimeSeriesMetrics(BaseModel):
    """Metadata and metrics about a time series dataset."""
    symbol: str = Field(..., description='Instrument symbol')
    timeframe: str = Field(..., description='Timeframe')
    start_time: datetime = Field(..., description='Start time of the dataset')
    end_time: datetime = Field(..., description='End time of the dataset')
    count: int = Field(..., description='Number of data points')
    earliest_timestamp: datetime = Field(..., description=
        'Earliest timestamp in the dataset')
    latest_timestamp: datetime = Field(..., description=
        'Latest timestamp in the dataset')
    missing_points: int = Field(..., description=
        'Number of missing data points')
    missing_percentage: float = Field(..., description=
        'Percentage of missing data points')
    average_gap: float = Field(..., description=
        'Average gap between data points in seconds')
    max_gap: float = Field(..., description=
        'Maximum gap between data points in seconds')
    source: str = Field(..., description='Data source')


@router.get('/ohlcv', response_model=List[OHLCVData], summary=
    'Retrieve historical OHLCV data', description=
    'Get historical OHLCV data for a specified symbol and timeframe.',
    responses={(200): {'description': 'OHLCV data retrieved successfully'},
    (204): {'description': 'No data available for the requested parameters'
    }, (400): {'description': 'Invalid request parameters'}, (500): {
    'description': 'Internal server error'}})
@async_with_exception_handling
async def get_historical_ohlcv(symbol: str=Query(..., description=
    DESC_SYMBOL), timeframe: Granularity=Query(..., description=
    DESC_TIMEFRAME), from_time: datetime=Query(None, description=
    DESC_START_TIME), to_time: datetime=Query(None, description=
    DESC_END_TIME), limit: Optional[int]=Query(1000, description=DESC_LIMIT
    ), include_current: bool=Query(False, description=DESC_INCLUDE_CURRENT),
    sort_order: str=Query('asc', description=DESC_SORT_ORDER), fill_missing:
    bool=Query(False, description=
    'Fill missing candles with interpolated data')):
    """
    Retrieve historical OHLCV (Open, High, Low, Close, Volume) data.
    
    Parameters:
    - **symbol**: Trading pair or instrument (e.g., 'EURUSD')
    - **timeframe**: Time interval for each candle
    - **from_time**: Start datetime (defaults to limit candles before to_time)
    - **to_time**: End datetime (defaults to current time)
    - **limit**: Maximum number of candles to return
    - **include_current**: Whether to include the current forming candle
    - **sort_order**: 'asc' for oldest first, 'desc' for newest first
    - **fill_missing**: Whether to fill gaps with interpolated data
    
    Returns:
    - List of OHLCV data objects
    """
    try:
        if to_time is None:
            to_time = datetime.now(timezone.utc)
        if to_time.tzinfo is None:
            to_time = to_time.replace(tzinfo=timezone.utc)
        if from_time is None:
            from_time = calculate_from_time(to_time, timeframe.value, limit)
        elif from_time.tzinfo is None:
            from_time = from_time.replace(tzinfo=timezone.utc)
        if from_time >= to_time:
            raise HTTPException(status_code=400, detail=
                'from_time must be earlier than to_time')
        logger.info(
            f'Retrieving OHLCV data for {symbol} {timeframe.value} from {from_time} to {to_time}'
            )
        candles = await ohlcv_repository.get_ohlcv_data(symbol, timeframe.
            value, from_time, to_time, limit)
        if not candles:
            logger.warning(
                f'No OHLCV data found for {symbol} {timeframe.value}')
            return []
        if fill_missing and len(candles) > 0:
            candles = await ohlcv_service.fill_missing_candles(candles,
                timeframe.value, from_time, to_time)
        if not include_current and len(candles) > 0:
            current_period_start = get_period_start(datetime.now(timezone.
                utc), timeframe.value)
            candles = [candle for candle in candles if candle.timestamp.
                replace(tzinfo=timezone.utc) < current_period_start]
        if sort_order.lower() == 'desc':
            candles.sort(key=lambda x: x.timestamp, reverse=True)
        else:
            candles.sort(key=lambda x: x.timestamp)
        return candles
    except Exception as e:
        logger.error(f'Error retrieving OHLCV data: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/ohlcv-dataframe', summary='Retrieve OHLCV data as DataFrame',
    description=
    'Get historical OHLCV data as pandas DataFrame, CSV, or NumPy array.',
    responses={(200): {'description': 'OHLCV data retrieved successfully'},
    (204): {'description': 'No data available for the requested parameters'
    }, (400): {'description': 'Invalid request parameters'}, (500): {
    'description': 'Internal server error'}})
@async_with_exception_handling
async def get_ohlcv_dataframe(symbol: str=Query(..., description=
    DESC_SYMBOL), timeframe: Granularity=Query(..., description=
    DESC_TIMEFRAME), from_time: datetime=Query(None, description=
    DESC_START_TIME), to_time: datetime=Query(None, description=
    DESC_END_TIME), limit: Optional[int]=Query(1000, description=DESC_LIMIT
    ), include_current: bool=Query(False, description=DESC_INCLUDE_CURRENT),
    sort_order: str=Query('asc', description=DESC_SORT_ORDER), fill_missing:
    bool=Query(False, description=
    'Fill missing candles with interpolated data'), format: DataFormat=
    Query(DataFormat.JSON, description='Output data format')):
    """
    Retrieve historical OHLCV data in various formats (DataFrame, CSV, NumPy).
    
    This endpoint is particularly useful for data analysis and machine learning workflows,
    offering formats suitable for direct integration with data science libraries.
    
    Parameters:
    - **symbol**: Trading pair or instrument (e.g., 'EURUSD')
    - **timeframe**: Time interval for each candle
    - **from_time**: Start datetime (defaults to limit candles before to_time)
    - **to_time**: End datetime (defaults to current time)
    - **limit**: Maximum number of candles to return
    - **include_current**: Whether to include the current forming candle
    - **sort_order**: 'asc' for oldest first, 'desc' for newest first
    - **fill_missing**: Whether to fill gaps with interpolated data
    - **format**: Output format (json, dataframe, csv, numpy)
    
    Returns:
    - OHLCV data in the requested format
    """
    try:
        candles = await get_historical_ohlcv(symbol, timeframe, from_time,
            to_time, limit, include_current, sort_order, fill_missing)
        if not candles:
            if format == DataFormat.JSON:
                return []
            raise HTTPException(status_code=204, detail=
                'No data available for the requested parameters')
        if format in [DataFormat.DATAFRAME, DataFormat.CSV, DataFormat.NUMPY]:
            data = [{'timestamp': candle.timestamp, 'open': candle.open,
                'high': candle.high, 'low': candle.low, 'close': candle.
                close, 'volume': candle.volume, 'symbol': candle.symbol,
                'timeframe': candle.timeframe} for candle in candles]
            df = pd.DataFrame(data)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
            if format == DataFormat.DATAFRAME:
                return df
            elif format == DataFormat.CSV:
                from fastapi.responses import Response
                csv_content = df.to_csv()
                return Response(content=csv_content, media_type='text/csv',
                    headers={'Content-Disposition':
                    f'attachment;filename={symbol}_{timeframe.value}.csv'})
            elif format == DataFormat.NUMPY:
                import numpy as np
                ohlcv_array = df[['open', 'high', 'low', 'close', 'volume']
                    ].to_numpy()
                return {'data': ohlcv_array.tolist(), 'index': df.index.
                    strftime('%Y-%m-%d %H:%M:%S').tolist()}
        return candles
    except Exception as e:
        logger.error(f'Error retrieving OHLCV DataFrame: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


def calculate_time_series_metrics(candles: List[OHLCVData], symbol: str,
    timeframe: str, from_time: datetime, to_time: datetime, source: str
    ) ->TimeSeriesMetrics:
    """Helper function to calculate metrics from candle data."""
    if not candles:
        return TimeSeriesMetrics(symbol=symbol, timeframe=timeframe,
            start_time=from_time, end_time=to_time, count=0,
            earliest_timestamp=from_time, latest_timestamp=to_time,
            missing_points=calculate_expected_candles(from_time, to_time,
            timeframe), missing_percentage=100.0, average_gap=0, max_gap=0,
            source=source)
    expected_count = calculate_expected_candles(from_time, to_time, timeframe)
    timestamps = [c.timestamp.replace(tzinfo=timezone.utc) for c in candles]
    timestamps.sort()
    gaps = []
    expected_gap_seconds = get_timeframe_seconds(timeframe)
    for i in range(1, len(timestamps)):
        gap = (timestamps[i] - timestamps[i - 1]).total_seconds()
        if gap > expected_gap_seconds * 1.5:
            gaps.append(gap)
    missing_points = max(0, expected_count - len(candles))
    missing_percentage = (missing_points / expected_count * 100 if 
        expected_count > 0 else 0)
    avg_gap = sum(gaps) / len(gaps) if gaps else 0
    max_gap = max(gaps) if gaps else 0
    return TimeSeriesMetrics(symbol=symbol, timeframe=timeframe, start_time
        =from_time, end_time=to_time, count=len(candles),
        earliest_timestamp=min(timestamps), latest_timestamp=max(timestamps
        ), missing_points=missing_points, missing_percentage=
        missing_percentage, average_gap=avg_gap, max_gap=max_gap, source=source
        )


@router.get('/ohlcv/metrics', response_model=TimeSeriesMetrics, summary=
    'Get OHLCV data metrics', description=
    'Retrieve metadata and quality metrics about available OHLCV data.')
@async_with_exception_handling
async def get_ohlcv_metrics(symbol: str=Query(..., description=DESC_SYMBOL),
    timeframe: Granularity=Query(..., description=DESC_TIMEFRAME),
    from_time: Optional[datetime]=Query(None, description=
    'Start time for analysis'), to_time: Optional[datetime]=Query(None,
    description='End time for analysis')):
    """
    Retrieve metadata and quality metrics about OHLCV data.
    
    This endpoint provides information about data availability, gaps,
    and overall data quality for a specified symbol and timeframe.
    
    Parameters:
    - **symbol**: Trading pair or instrument (e.g., 'EURUSD')
    - **timeframe**: Time interval for each candle
    - **from_time**: Start time for analysis (optional)
    - **to_time**: End time for analysis (optional)
    
    Returns:
    - TimeSeriesMetrics object with data quality information
    """
    try:
        to_time = to_time or datetime.now(timezone.utc)
        if to_time.tzinfo is None:
            to_time = to_time.replace(tzinfo=timezone.utc)
        from_time = from_time or to_time - timedelta(days=30)
        if from_time.tzinfo is None:
            from_time = from_time.replace(tzinfo=timezone.utc)
        if from_time >= to_time:
            raise HTTPException(status_code=400, detail=
                'from_time must be earlier than to_time')
        candles = await ohlcv_repository.get_ohlcv_data(symbol, timeframe.
            value, from_time, to_time, limit=None)
        source = await ohlcv_repository.get_data_source(symbol, timeframe.value
            ) or 'unknown'
        metrics = calculate_time_series_metrics(candles=candles, symbol=
            symbol, timeframe=timeframe.value, from_time=from_time, to_time
            =to_time, source=source)
        if metrics.count == 0 and not candles:
            raise HTTPException(status_code=204, detail=
                'No data available for the requested parameters to calculate metrics'
                )
        return metrics
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error calculating OHLCV metrics: {str(e)}', exc_info
            =True)
        raise HTTPException(status_code=500, detail=
            f'Internal server error calculating metrics: {str(e)}')


@router.get('/ohlcv/available-ranges', response_model=Dict[str, Dict[str,
    Dict[str, Union[str, datetime]]]], summary=
    'Get available data date ranges', description=
    'Retrieve information about available data date ranges per symbol and timeframe.'
    )
@async_with_exception_handling
async def get_available_ranges():
    """
    Get information about available data date ranges by symbol and timeframe.
    
    Returns a nested dictionary structure with information about earliest and
    latest data points available for each symbol and timeframe combination.
    
    Returns:
    - Dictionary mapping symbols to timeframes to date range information
    """
    try:
        ranges = await ohlcv_repository.get_available_data_ranges()
        result = {}
        for item in ranges:
            symbol = item['symbol']
            timeframe = item['timeframe']
            start_date = item['earliest_timestamp']
            end_date = item['latest_timestamp']
            source = item.get('source', 'unknown')
            if symbol not in result:
                result[symbol] = {}
            result[symbol][timeframe] = {'earliest': start_date, 'latest':
                end_date, 'source': source}
        return result
    except Exception as e:
        logger.error(f'Error retrieving available data ranges: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/ohlcv/available-symbols', response_model=Dict[str, List[str]],
    summary='Get available symbols by timeframe', description=
    'Retrieve information about available symbols for each timeframe.')
@async_with_exception_handling
async def get_available_symbols():
    """
    Get information about which symbols are available for each timeframe.
    
    Returns a dictionary mapping timeframes to lists of available symbols.
    
    Returns:
    - Dictionary mapping timeframes to lists of symbols
    """
    try:
        ranges = await ohlcv_repository.get_available_data_ranges()
        result = {}
        for item in ranges:
            symbol = item['symbol']
            timeframe = item['timeframe']
            if timeframe not in result:
                result[timeframe] = []
            if symbol not in result[timeframe]:
                result[timeframe].append(symbol)
        return result
    except Exception as e:
        logger.error(f'Error retrieving available symbols: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


def calculate_from_time(to_time: datetime, timeframe: str, limit: int
    ) ->datetime:
    """
    Calculate the from_time based on to_time, timeframe, and limit.
    
    Args:
        to_time: End time
        timeframe: Candle timeframe
        limit: Maximum number of candles
        
    Returns:
        Calculated from_time
    """
    seconds = get_timeframe_seconds(timeframe)
    from_time = to_time - timedelta(seconds=seconds * limit)
    return from_time


def get_timeframe_seconds(timeframe: str) ->int:
    """
    Get the number of seconds for a given timeframe.
    
    Args:
        timeframe: Candle timeframe
        
    Returns:
        Number of seconds
    """
    if timeframe.endswith('s'):
        return int(timeframe[:-1])
    elif timeframe.endswith('m'):
        return int(timeframe[:-1]) * 60
    elif timeframe.endswith('h'):
        return int(timeframe[:-1]) * 3600
    elif timeframe.endswith('d'):
        return int(timeframe[:-1]) * 86400
    elif timeframe.endswith('w'):
        return int(timeframe[:-1]) * 604800
    elif timeframe.endswith('mn'):
        return int(timeframe[:-2]) * 2592000
    else:
        raise ValueError(f'Invalid timeframe format: {timeframe}')


def get_period_start(dt: datetime, timeframe: str) ->datetime:
    """
    Get the start of the period for a given datetime and timeframe.
    
    Args:
        dt: Datetime
        timeframe: Candle timeframe
        
    Returns:
        Start of period
    """
    if timeframe.endswith('s'):
        seconds = int(timeframe[:-1])
        return dt.replace(microsecond=0, second=dt.second - dt.second % seconds
            )
    elif timeframe.endswith('m'):
        minutes = int(timeframe[:-1])
        return dt.replace(microsecond=0, second=0, minute=dt.minute - dt.
            minute % minutes)
    elif timeframe.endswith('h'):
        hours = int(timeframe[:-1])
        return dt.replace(microsecond=0, second=0, minute=0, hour=dt.hour -
            dt.hour % hours)
    elif timeframe.endswith('d'):
        return dt.replace(microsecond=0, second=0, minute=0, hour=0)
    elif timeframe.endswith('w'):
        days_from_monday = dt.weekday()
        return (dt - timedelta(days=days_from_monday)).replace(microsecond=
            0, second=0, minute=0, hour=0)
    elif timeframe.endswith('mn'):
        return dt.replace(microsecond=0, second=0, minute=0, hour=0, day=1)
    else:
        raise ValueError(f'Invalid timeframe format: {timeframe}')


def calculate_expected_candles(from_time: datetime, to_time: datetime,
    timeframe: str) ->int:
    """
    Calculate the expected number of candles between two datetimes.
    
    Args:
        from_time: Start time
        to_time: End time
        timeframe: Candle timeframe
        
    Returns:
        Expected number of candles
    """
    seconds = get_timeframe_seconds(timeframe)
    total_seconds = (to_time - from_time).total_seconds()
    return int((total_seconds + seconds - 1) // seconds)


@router.get('/ohlcv/filtered', response_model=List[OHLCVData], summary=
    'Retrieve filtered OHLCV data', description=
    'Get OHLCV data with advanced filtering options beyond basic time and symbol filters.'
    , responses={(200): {'description':
    'Filtered OHLCV data retrieved successfully'}, (204): {'description':
    'No data matches the filter criteria'}, (400): {'description':
    'Invalid filter parameters'}, (500): {'description':
    'Internal server error'}})
@async_with_exception_handling
async def get_filtered_ohlcv(symbol: str=Query(..., description=DESC_SYMBOL
    ), timeframe: Granularity=Query(..., description=DESC_TIMEFRAME),
    from_time: Optional[datetime]=Query(None, description=DESC_START_TIME),
    to_time: Optional[datetime]=Query(None, description=DESC_END_TIME),
    limit: Optional[int]=Query(1000, description=DESC_LIMIT),
    include_current: bool=Query(False, description=DESC_INCLUDE_CURRENT),
    sort_order: str=Query('asc', description=DESC_SORT_ORDER), filters:
    FilterParams=Depends()):
    """
    Retrieve historical OHLCV data with advanced filtering capabilities.
    
    This endpoint extends the basic OHLCV retrieval with additional filtering options
    like price thresholds, volume requirements, and market session filters.
    
    Parameters:
    - **symbol**: Trading pair or instrument (e.g., 'EURUSD')
    - **timeframe**: Time interval for each candle
    - **from_time/to_time**: Time range filters
    - **limit**: Maximum number of candles to return
    - **include_current**: Whether to include the current forming candle
    - **sort_order**: 'asc' for oldest first, 'desc' for newest first
    - **fill_missing**: Whether to fill gaps with interpolated data
    - **format**: Output format (json, dataframe, csv, numpy)
    
    Returns:
    - OHLCV data in the requested format
    """
    try:
        candles = await get_historical_ohlcv(symbol, timeframe, from_time,
            to_time, limit, include_current, sort_order, False)
        if not candles:
            return []
        filtered_candles = candles.copy()
        if filters.min_price is not None:
            filtered_candles = [c for c in filtered_candles if c.low >=
                filters.min_price]
        if filters.max_price is not None:
            filtered_candles = [c for c in filtered_candles if c.high <=
                filters.max_price]
        if filters.price_range is not None and len(filters.price_range) == 2:
            min_p, max_p = filters.price_range
            filtered_candles = [c for c in filtered_candles if c.low >=
                min_p and c.high <= max_p]
        if filters.min_volume is not None:
            filtered_candles = [c for c in filtered_candles if c.volume >=
                filters.min_volume]
        if filters.max_volume is not None:
            filtered_candles = [c for c in filtered_candles if c.volume <=
                filters.max_volume]
        if filters.session is not None:
            filtered_candles = filter_by_session(filtered_candles, filters.
                session)
        pip_factor = get_pip_factor(symbol)
        if filters.min_range_pips is not None:
            min_range = filters.min_range_pips / pip_factor
            filtered_candles = [c for c in filtered_candles if c.high - c.
                low >= min_range]
        if filters.min_body_pips is not None:
            min_body = filters.min_body_pips / pip_factor
            filtered_candles = [c for c in filtered_candles if abs(c.close -
                c.open) >= min_body]
        return filtered_candles
    except Exception as e:
        logger.error(f'Error retrieving filtered OHLCV data: {str(e)}',
            exc_info=True)
        raise HTTPException(status_code=500, detail=
            f'Internal server error during filtering: {str(e)}')


@router.post('/ohlcv/batch', response_model=Dict[str, Dict[str, List[
    OHLCVData]]], summary='Batch retrieve OHLCV data', description=
    'Get data for multiple symbols and timeframes in one request.',
    responses={(200): {'description': 'Batch data retrieval successful'}, (
    400): {'description': 'Invalid request parameters'}, (500): {
    'description': 'Internal server error'}})
@async_with_exception_handling
async def get_batch_ohlcv(request: Dict[str, List[str]]=Body(...,
    description='Dictionary mapping symbols to list of timeframes', example
    ={'EURUSD': ['1m', '5m'], 'GBPUSD': ['1h']}), from_time: Optional[
    datetime]=Query(None, description=DESC_START_TIME), to_time: Optional[
    datetime]=Query(None, description=DESC_END_TIME), limit: Optional[int]=
    Query(1000, description=
    'Maximum number of candles per symbol/timeframe'), include_current:
    bool=Query(False, description='Include current incomplete candle'),
    sort_order: str=Query('asc', description="Sorting order: 'asc' or 'desc'")
    ):
    """
    Retrieve historical OHLCV data for multiple symbols and timeframes in a single request.
    
    This endpoint is useful for efficiently retrieving data across multiple instruments
    and timeframes in a single API call, reducing the number of requests needed.
    
    Parameters:
    - **request**: Dictionary mapping symbols to list of timeframes
    - **from_time/to_time**: Time range for all data (applied to all symbols)
    - **limit**: Maximum candles per symbol/timeframe
    - **include_current**: Whether to include the current forming candle
    - **sort_order**: 'asc' for oldest first, 'desc' for newest first
    
    Returns:
    - Nested dictionary of symbol -> timeframe -> OHLCV data list
    """
    try:
        result = {}
        for symbol, timeframes in request.items():
            result[symbol] = {}
            for timeframe_str in timeframes:
                try:
                    timeframe = Granularity(timeframe_str)
                    candles = await get_historical_ohlcv(symbol, timeframe,
                        from_time, to_time, limit, include_current,
                        sort_order, False)
                    result[symbol][timeframe_str] = candles
                except ValueError as e:
                    logger.warning(f'Invalid timeframe: {timeframe_str}')
                    result[symbol][timeframe_str] = []
        return result
    except Exception as e:
        logger.error(f'Error processing batch OHLCV request: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


def filter_by_session(candles: List[OHLCVData], session: str) ->List[OHLCVData
    ]:
    """
    Filter candles by market session.
    
    Args:
        candles: List of OHLCV candles
        session: Session name ('asian', 'london', 'new_york', 'overlap')
        
    Returns:
        Filtered list of candles
    """
    session_hours = {'asian': (0, 9), 'london': (8, 16), 'new_york': (13, 
        21), 'overlap': (13, 16)}
    if session not in session_hours:
        raise ValueError(f'Invalid session: {session}')
    start_hour, end_hour = session_hours[session]
    return [candle for candle in candles if start_hour <= candle.timestamp.
        hour < end_hour]


def get_pip_factor(symbol: str) ->float:
    """
    Get the pip factor for a currency pair.
    
    Args:
        symbol: Currency pair symbol
        
    Returns:
        Pip factor (usually 0.0001 for most pairs, 0.01 for JPY pairs)
    """
    if symbol.endswith('JPY'):
        return 0.01
    return 0.0001


async def get_ohlcv_service():
    """Dependency for getting OHLCV service instance."""
    pool = await get_db_pool()
    redis_client = await get_redis_client()
    return OHLCVService(pool=pool, redis_client=redis_client)


@router.get('/historical/{instrument}/ohlcv', response_model=OHLCVResponse)
@async_with_exception_handling
async def get_historical_ohlcv_by_path(instrument: str=Path(...,
    description=DESC_INSTRUMENT_ID), start_time: datetime=Query(...,
    description='Start time for data retrieval (ISO format)'), end_time:
    datetime=Query(..., description=
    'End time for data retrieval (ISO format)'), timeframe: TimeframeEnum=
    Query(..., description='Candle timeframe (e.g., 1m, 5m, 1h, 1d)'),
    include_incomplete: bool=Query(False, description=
    'Include incomplete candles'), ohlcv_service: OHLCVService=Depends(
    get_ohlcv_service)):
    """
    Retrieve historical OHLCV data for specified instrument and timeframe using path parameter.
    
    - **instrument**: Trading pair or instrument (e.g., EUR_USD)
    - **start_time**: Start time in ISO format
    - **end_time**: End time in ISO format
    - **timeframe**: Candle timeframe (e.g., 1m, 5m, 1h, 1d)
    - **include_incomplete**: Whether to include the most recent candle if incomplete
    
    Returns OHLCV data as a list of candles with timestamp, open, high, low, close and volume.
    """
    try:
        validate_instrument_format(instrument)
        validate_timeframe(timeframe)
        validate_date_range(start_time, end_time)
        data = await ohlcv_service.get_historical_ohlcv(instrument=
            instrument, start_time=start_time, end_time=end_time, timeframe
            =timeframe, include_incomplete=include_incomplete)
        return OHLCVResponse(instrument=instrument, timeframe=timeframe,
            start_time=start_time, end_time=end_time, candle_count=len(data
            ), data=data)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as db_error:
        logger.error(
            f'Database error retrieving OHLCV for {instrument}: {db_error}',
            exc_info=True)
        raise HTTPException(status_code=500, detail='Database error occurred')


@router.get('/historical/{instrument}/aggregated', response_model=OHLCVResponse
    )
@async_with_exception_handling
async def get_aggregated_data(instrument: str=Path(..., description=
    DESC_INSTRUMENT_ID), start_time: datetime=Query(..., description=
    'Start time for data retrieval (ISO format)'), end_time: datetime=Query
    (..., description='End time for data retrieval (ISO format)'),
    source_timeframe: TimeframeEnum=Query(..., description=
    'Source timeframe (e.g., 1m, 5m)'), target_timeframe: TimeframeEnum=
    Query(..., description='Target timeframe (e.g., 1h, 1d)'),
    aggregation_method: AggregationMethodEnum=Query(AggregationMethodEnum.
    OHLCV, description='Aggregation method to use'), ohlcv_service:
    OHLCVService=Depends(get_ohlcv_service)):
    """
    Retrieve historical data and aggregate from source to target timeframe.
    Useful for converting smaller timeframes to larger ones.
    
    - **instrument**: Trading pair or instrument (e.g., EUR_USD)
    - **start_time**: Start time in ISO format
    - **end_time**: End time in ISO format
    - **source_timeframe**: Original timeframe of the data
    - **target_timeframe**: Target timeframe to convert to
    - **aggregation_method**: Method to use for aggregation
    """
    try:
        validate_instrument_format(instrument)
        validate_timeframe(source_timeframe)
        validate_timeframe(target_timeframe)
        validate_date_range(start_time, end_time)
        if not is_valid_timeframe_conversion(source_timeframe, target_timeframe
            ):
            raise ValidationError(
                'Target timeframe must be larger than source timeframe')
        raw_data = await ohlcv_service.get_historical_ohlcv(instrument=
            instrument, start_time=start_time, end_time=end_time, timeframe
            =source_timeframe)
        aggregator = TimeseriesAggregator()
        aggregated_data = aggregator.aggregate(data=raw_data,
            source_timeframe=source_timeframe, target_timeframe=
            target_timeframe, method=aggregation_method)
        return OHLCVResponse(instrument=instrument, timeframe=
            target_timeframe, start_time=start_time, end_time=end_time,
            candle_count=len(aggregated_data), data=aggregated_data)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=
            'Error during data aggregation')


@router.get('/export/{instrument}/ohlcv')
@async_with_exception_handling
async def export_historical_ohlcv(instrument: str=Path(..., description=
    DESC_INSTRUMENT_ID), start_time: datetime=Query(..., description=
    'Start time for data export (ISO format)'), end_time: datetime=Query(
    ..., description='End time for data export (ISO format)'), timeframe:
    TimeframeEnum=Query(..., description=
    'Candle timeframe (e.g., 1m, 5m, 1h, 1d)'), format: ExportFormatEnum=
    Query(ExportFormatEnum.CSV, description=
    'Export format (csv, json, parquet)'), ohlcv_service: OHLCVService=
    Depends(get_ohlcv_service)):
    """
    Export historical OHLCV data in specified format.
    
    - **instrument**: Trading pair or instrument (e.g., EUR_USD)
    - **start_time**: Start time in ISO format
    - **end_time**: End time in ISO format
    - **timeframe**: Candle timeframe (e.g., 1m, 5m, 1h, 1d)
    - **format**: Export format (csv, json, parquet)
    """
    try:
        validate_instrument_format(instrument)
        validate_timeframe(timeframe)
        validate_date_range(start_time, end_time)
        data = await ohlcv_service.get_historical_ohlcv(instrument=
            instrument, start_time=start_time, end_time=end_time, timeframe
            =timeframe)
        if not data:
            return JSONResponse(status_code=404, content={'message':
                'No data found for specified parameters'})
        if format == ExportFormatEnum.CSV:
            csv_content = convert_to_csv(data)
            filename = (
                f'{instrument}_{timeframe.value}_{start_time.date()}_{end_time.date()}.csv'
                )
            return StreamingResponse(io.StringIO(csv_content), media_type=
                'text/csv', headers={'Content-Disposition':
                f'attachment; filename={filename}'})
        elif format == ExportFormatEnum.JSON:
            return {'metadata': {'instrument': instrument, 'timeframe':
                timeframe.value, 'start_time': start_time, 'end_time':
                end_time, 'count': len(data)}, 'data':
                format_ohlcv_for_json(data)}
        elif format == ExportFormatEnum.PARQUET:
            parquet_bytes = convert_to_parquet(data)
            filename = (
                f'{instrument}_{timeframe.value}_{start_time.date()}_{end_time.date()}.parquet'
                )
            return StreamingResponse(io.BytesIO(parquet_bytes), media_type=
                'application/octet-stream', headers={'Content-Disposition':
                f'attachment; filename={filename}'})
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail='Error during data export')


@router.post('/internal/ohlcv_batch')
@async_with_exception_handling
async def get_ohlcv_batch_for_feature_calculation(request_data:
    OHLCVBatchRequest, ohlcv_service: OHLCVService=Depends(
    get_ohlcv_service), auth: ServiceAuth=Depends(verify_service_auth)):
    """
    Internal API endpoint for feature-store-service to retrieve batch OHLCV data.
    
    This endpoint requires service-to-service authentication.
    """
    if auth.service_name != 'feature-store-service':
        raise HTTPException(status_code=403, detail=
            'Unauthorized service access')
    try:
        results = await ohlcv_service.get_multi_instrument_ohlcv(instruments
            =request_data.instruments, start_time=request_data.start_time,
            end_time=request_data.end_time, timeframe=request_data.timeframe)
        return {'data': {instrument: format_ohlcv_for_json(data) for 
            instrument, data in results.items()}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=
            'Error processing batch request')
