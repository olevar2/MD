"""
Ohlcv router module.

This module provides functionality for...
"""

import io
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import JSONResponse, StreamingResponse
from data_pipeline_service.services.ohlcv_service import OHLCVService, get_ohlcv_service
from data_pipeline_service.models.schemas import OHLCVResponse, TimeframeEnum
from data_pipeline_service.exceptions.validation_exceptions import ValidationError
from ...models.schemas import OHLCVData, OHLCVBatchRequest, AggregationMethodEnum, ExportFormatEnum, ServiceAuth
from ...services.timeseries_aggregator import TimeseriesAggregator
from ...services.export_service import convert_to_csv, convert_to_parquet, format_ohlcv_for_json
from ...validation.ohlcv_validators import validate_instrument_format, validate_timeframe, validate_date_range, is_valid_timeframe_conversion
from ...db.engine import get_db_pool
from ...api.auth import verify_service_auth
from ...config.settings import get_redis_client
router = APIRouter(prefix='/api/v1', tags=['OHLCV Data'])


from data_pipeline_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@router.get('/historical/{instrument}/ohlcv', response_model=OHLCVResponse,
    tags=['Historical Data'])
@async_with_exception_handling
async def get_historical_ohlcv(instrument: str=Path(..., description=
    'Trading instrument identifier (e.g., EUR_USD)'), start_time: datetime=
    Query(..., description='Start time for data retrieval (ISO format)'),
    end_time: datetime=Query(..., description=
    'End time for data retrieval (ISO format)'), timeframe: TimeframeEnum=
    Query(..., description='Candle timeframe (e.g., 1m, 5m, 1h, 1d)'),
    include_incomplete: bool=Query(False, description=
    'Include incomplete candles'), ohlcv_service: OHLCVService=Depends(
    get_ohlcv_service)):
    """
    Retrieve historical OHLCV data for specified instrument and timeframe.
    
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
    except Exception as e:
        raise HTTPException(status_code=500, detail='Database error occurred')


@router.get('/historical/{instrument}/aggregated', response_model=
    OHLCVResponse, tags=['Data Aggregation'])
@async_with_exception_handling
async def get_aggregated_data(instrument: str=Path(..., description=
    'Trading instrument identifier (e.g., EUR_USD)'), start_time: datetime=
    Query(..., description='Start time for data retrieval (ISO format)'),
    end_time: datetime=Query(..., description=
    'End time for data retrieval (ISO format)'), source_timeframe:
    TimeframeEnum=Query(..., description='Source timeframe (e.g., 1m, 5m)'),
    target_timeframe: TimeframeEnum=Query(..., description=
    'Target timeframe (e.g., 1h, 1d)'), aggregation_method:
    AggregationMethodEnum=Query(AggregationMethodEnum.OHLCV, description=
    'Aggregation method to use'), ohlcv_service: OHLCVService=Depends(
    get_ohlcv_service)):
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


@router.get('/export/{instrument}/ohlcv', tags=['Data Export'])
@async_with_exception_handling
async def export_historical_ohlcv(instrument: str=Path(..., description=
    'Trading instrument identifier (e.g., EUR_USD)'), start_time: datetime=
    Query(..., description='Start time for data export (ISO format)'),
    end_time: datetime=Query(..., description=
    'End time for data export (ISO format)'), timeframe: TimeframeEnum=
    Query(..., description='Candle timeframe (e.g., 1m, 5m, 1h, 1d)'),
    format: ExportFormatEnum=Query(ExportFormatEnum.CSV, description=
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


@router.post('/internal/ohlcv_batch', tags=['Internal API'])
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
