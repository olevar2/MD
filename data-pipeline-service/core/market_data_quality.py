"""
Market Data Quality API

This module provides API endpoints for the Market Data Quality Framework.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field
from common_lib.exceptions import DataValidationError
from common_lib.monitoring import MetricsCollector, AlertManager
from common_lib.schemas import OHLCVData, TickData
from core.market_data_quality_framework import MarketDataQualityFramework, DataQualityLevel, DataQualitySLA, DataQualityMetrics, DataQualityReport
from data_pipeline_service.services.market_data_service import MarketDataService
from data_pipeline_service.dependencies import get_market_data_service, get_metrics_collector, get_alert_manager
logger = logging.getLogger(__name__)
router = APIRouter(prefix='/api/v1/market-data-quality', tags=[
    'Market Data Quality'])
_quality_framework: Optional[MarketDataQualityFramework] = None


from data_pipeline_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

def get_quality_framework(metrics_collector: MetricsCollector=Depends(
    get_metrics_collector), alert_manager: AlertManager=Depends(
    get_alert_manager)) ->MarketDataQualityFramework:
    """
    Get or initialize the Market Data Quality Framework.
    
    Args:
        metrics_collector: Metrics collector for monitoring
        alert_manager: Alert manager for notifications
        
    Returns:
        Market Data Quality Framework instance
    """
    global _quality_framework
    if _quality_framework is None:
        _quality_framework = MarketDataQualityFramework(metrics_collector=
            metrics_collector, alert_manager=alert_manager)
    return _quality_framework


class ValidateOHLCVRequest(BaseModel):
    """Request model for validating OHLCV data"""
    instrument: str = Field(..., description='Instrument identifier')
    timeframe: str = Field(..., description='Timeframe')
    start_time: Optional[datetime] = Field(None, description=
        'Start time for data retrieval')
    end_time: Optional[datetime] = Field(None, description=
        'End time for data retrieval')
    instrument_type: str = Field('forex', description=
        'Type of instrument (forex, crypto, stocks)')
    quality_level: DataQualityLevel = Field(DataQualityLevel.STANDARD,
        description='Quality level to apply')
    generate_report: bool = Field(False, description=
        'Whether to generate a detailed report')


    class Config:
    """
    Config class.
    
    Attributes:
        Add attributes here
    """

        schema_extra = {'example': {'instrument': 'EUR_USD', 'timeframe':
            '1h', 'start_time': '2023-01-01T00:00:00Z', 'end_time':
            '2023-01-31T23:59:59Z', 'instrument_type': 'forex',
            'quality_level': 'standard', 'generate_report': True}}


class ValidateTickRequest(BaseModel):
    """Request model for validating tick data"""
    instrument: str = Field(..., description='Instrument identifier')
    start_time: Optional[datetime] = Field(None, description=
        'Start time for data retrieval')
    end_time: Optional[datetime] = Field(None, description=
        'End time for data retrieval')
    instrument_type: str = Field('forex', description=
        'Type of instrument (forex, crypto, stocks)')
    quality_level: DataQualityLevel = Field(DataQualityLevel.STANDARD,
        description='Quality level to apply')
    generate_report: bool = Field(False, description=
        'Whether to generate a detailed report')


    class Config:
    """
    Config class.
    
    Attributes:
        Add attributes here
    """

        schema_extra = {'example': {'instrument': 'EUR_USD', 'start_time':
            '2023-01-01T00:00:00Z', 'end_time': '2023-01-01T01:00:00Z',
            'instrument_type': 'forex', 'quality_level': 'standard',
            'generate_report': True}}


class ValidateAlternativeDataRequest(BaseModel):
    """Request model for validating alternative data"""
    data_type: str = Field(..., description=
        'Type of alternative data (news, economic, sentiment)')
    start_time: Optional[datetime] = Field(None, description=
        'Start time for data retrieval')
    end_time: Optional[datetime] = Field(None, description=
        'End time for data retrieval')
    generate_report: bool = Field(False, description=
        'Whether to generate a detailed report')


    class Config:
    """
    Config class.
    
    Attributes:
        Add attributes here
    """

        schema_extra = {'example': {'data_type': 'news', 'start_time':
            '2023-01-01T00:00:00Z', 'end_time': '2023-01-31T23:59:59Z',
            'generate_report': True}}


class ValidationResponse(BaseModel):
    """Response model for validation requests"""
    is_valid: bool = Field(..., description='Whether the data is valid')
    message: str = Field(..., description='Validation message')
    timestamp: datetime = Field(default_factory=datetime.utcnow,
        description='Timestamp of validation')


class SLAUpdateRequest(BaseModel):
    """Request model for updating SLAs"""
    completeness: float = Field(99.5, description=
        'Percentage of required fields that must be present')
    timeliness: float = Field(99.0, description=
        'Percentage of data that must arrive within SLA timeframe')
    accuracy: float = Field(99.9, description=
        'Percentage of data that must pass validation checks')
    consistency: float = Field(99.5, description=
        'Percentage of data that must be consistent across sources')
    max_allowed_gaps_per_day: int = Field(0, description=
        'Maximum number of gaps allowed per day')
    max_allowed_spikes_per_day: int = Field(0, description=
        'Maximum number of price spikes allowed per day')
    max_latency_seconds: float = Field(1.0, description=
        'Maximum latency for real-time data in seconds')


    class Config:
    """
    Config class.
    
    Attributes:
        Add attributes here
    """

        schema_extra = {'example': {'completeness': 99.5, 'timeliness': 
            99.0, 'accuracy': 99.9, 'consistency': 99.5,
            'max_allowed_gaps_per_day': 0, 'max_allowed_spikes_per_day': 0,
            'max_latency_seconds': 1.0}}


@router.post('/validate/ohlcv', response_model=Union[ValidationResponse,
    DataQualityReport])
@async_with_exception_handling
async def validate_ohlcv_data(request: ValidateOHLCVRequest,
    quality_framework: MarketDataQualityFramework=Depends(
    get_quality_framework), market_data_service: MarketDataService=Depends(
    get_market_data_service)):
    """
    Validate OHLCV data for a specific instrument and timeframe.
    
    Args:
        request: Validation request
        quality_framework: Market Data Quality Framework
        market_data_service: Market Data Service
        
    Returns:
        Validation response or detailed report
    """
    try:
        if request.start_time is None:
            request.start_time = datetime.utcnow() - timedelta(days=7)
        if request.end_time is None:
            request.end_time = datetime.utcnow()
        ohlcv_data = await market_data_service.get_ohlcv_data(instrument=
            request.instrument, timeframe=request.timeframe, start_time=
            request.start_time, end_time=request.end_time)
        if not ohlcv_data:
            raise HTTPException(status_code=404, detail=
                'No OHLCV data found for the specified parameters')
        import pandas as pd
        df = pd.DataFrame([d.dict() for d in ohlcv_data])
        result = quality_framework.validate_ohlcv_data(data=df,
            instrument_type=request.instrument_type, quality_level=request.
            quality_level, generate_report=request.generate_report)
        if request.generate_report:
            return result
        else:
            return ValidationResponse(is_valid=result, message=
                'OHLCV data validation passed' if result else
                'OHLCV data validation failed', timestamp=datetime.utcnow())
    except DataValidationError as e:
        logger.error(f'Data validation error: {str(e)}')
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f'Error validating OHLCV data: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Error validating OHLCV data: {str(e)}')


@router.post('/validate/tick', response_model=Union[ValidationResponse,
    DataQualityReport])
@async_with_exception_handling
async def validate_tick_data(request: ValidateTickRequest,
    quality_framework: MarketDataQualityFramework=Depends(
    get_quality_framework), market_data_service: MarketDataService=Depends(
    get_market_data_service)):
    """
    Validate tick data for a specific instrument.
    
    Args:
        request: Validation request
        quality_framework: Market Data Quality Framework
        market_data_service: Market Data Service
        
    Returns:
        Validation response or detailed report
    """
    try:
        if request.start_time is None:
            request.start_time = datetime.utcnow() - timedelta(hours=1)
        if request.end_time is None:
            request.end_time = datetime.utcnow()
        tick_data = await market_data_service.get_tick_data(instrument=
            request.instrument, start_time=request.start_time, end_time=
            request.end_time)
        if not tick_data:
            raise HTTPException(status_code=404, detail=
                'No tick data found for the specified parameters')
        import pandas as pd
        df = pd.DataFrame([d.dict() for d in tick_data])
        result = quality_framework.validate_tick_data(data=df,
            instrument_type=request.instrument_type, quality_level=request.
            quality_level, generate_report=request.generate_report)
        if request.generate_report:
            return result
        else:
            return ValidationResponse(is_valid=result, message=
                'Tick data validation passed' if result else
                'Tick data validation failed', timestamp=datetime.utcnow())
    except DataValidationError as e:
        logger.error(f'Data validation error: {str(e)}')
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f'Error validating tick data: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Error validating tick data: {str(e)}')


@router.post('/validate/alternative', response_model=Union[
    ValidationResponse, DataQualityReport])
@async_with_exception_handling
async def validate_alternative_data(request: ValidateAlternativeDataRequest,
    quality_framework: MarketDataQualityFramework=Depends(
    get_quality_framework), market_data_service: MarketDataService=Depends(
    get_market_data_service)):
    """
    Validate alternative data.
    
    Args:
        request: Validation request
        quality_framework: Market Data Quality Framework
        market_data_service: Market Data Service
        
    Returns:
        Validation response or detailed report
    """
    try:
        if request.start_time is None:
            request.start_time = datetime.utcnow() - timedelta(days=7)
        if request.end_time is None:
            request.end_time = datetime.utcnow()
        alternative_data = await market_data_service.get_alternative_data(
            data_type=request.data_type, start_time=request.start_time,
            end_time=request.end_time)
        if not alternative_data:
            raise HTTPException(status_code=404, detail=
                'No alternative data found for the specified parameters')
        import pandas as pd
        df = pd.DataFrame([d.dict() for d in alternative_data])
        result = quality_framework.validate_alternative_data(data=df,
            data_type=request.data_type, generate_report=request.
            generate_report)
        if request.generate_report:
            return result
        else:
            return ValidationResponse(is_valid=result, message=
                'Alternative data validation passed' if result else
                'Alternative data validation failed', timestamp=datetime.
                utcnow())
    except DataValidationError as e:
        logger.error(f'Data validation error: {str(e)}')
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f'Error validating alternative data: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Error validating alternative data: {str(e)}')


@router.get('/metrics', response_model=List[DataQualityMetrics])
@async_with_exception_handling
async def get_data_quality_metrics(instrument: Optional[str]=Query(None,
    description='Instrument to get metrics for'), data_type: str=Query(
    'ohlcv', description='Type of data (ohlcv, tick)'), lookback_hours: int
    =Query(24, description='Hours to look back'), quality_framework:
    MarketDataQualityFramework=Depends(get_quality_framework)):
    """
    Get data quality metrics.
    
    Args:
        instrument: Instrument to get metrics for (optional)
        data_type: Type of data (ohlcv, tick)
        lookback_hours: Hours to look back
        quality_framework: Market Data Quality Framework
        
    Returns:
        List of data quality metrics
    """
    try:
        metrics = quality_framework.get_data_quality_metrics(instrument=
            instrument, data_type=data_type, lookback_hours=lookback_hours)
        return metrics
    except Exception as e:
        logger.exception(f'Error getting data quality metrics: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Error getting data quality metrics: {str(e)}')


@router.get('/sla/{instrument_type}/{data_type}', response_model=DataQualitySLA
    )
@async_with_exception_handling
async def get_data_quality_sla(instrument_type: str=Path(..., description=
    'Type of instrument (forex, crypto, stocks)'), data_type: str=Path(...,
    description='Type of data (ohlcv, tick)'), quality_framework:
    MarketDataQualityFramework=Depends(get_quality_framework)):
    """
    Get data quality SLA.
    
    Args:
        instrument_type: Type of instrument (forex, crypto, stocks)
        data_type: Type of data (ohlcv, tick)
        quality_framework: Market Data Quality Framework
        
    Returns:
        Data quality SLA
    """
    try:
        sla = quality_framework.get_data_quality_sla(instrument_type=
            instrument_type, data_type=data_type)
        return sla
    except Exception as e:
        logger.exception(f'Error getting data quality SLA: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Error getting data quality SLA: {str(e)}')


@router.put('/sla/{key}', response_model=DataQualitySLA)
@async_with_exception_handling
async def update_data_quality_sla(key: str=Path(..., description=
    'SLA key (default, ohlcv, tick, forex, crypto, stocks, etc.)'), sla:
    SLAUpdateRequest=Body(..., description='SLA to update'),
    quality_framework: MarketDataQualityFramework=Depends(
    get_quality_framework)):
    """
    Update data quality SLA.
    
    Args:
        key: SLA key (default, ohlcv, tick, forex, crypto, stocks, etc.)
        sla: SLA to update
        quality_framework: Market Data Quality Framework
        
    Returns:
        Updated data quality SLA
    """
    try:
        data_quality_sla = DataQualitySLA(completeness=sla.completeness,
            timeliness=sla.timeliness, accuracy=sla.accuracy, consistency=
            sla.consistency, max_allowed_gaps_per_day=sla.
            max_allowed_gaps_per_day, max_allowed_spikes_per_day=sla.
            max_allowed_spikes_per_day, max_latency_seconds=sla.
            max_latency_seconds)
        quality_framework.set_data_quality_sla(sla=data_quality_sla, key=key)
        return quality_framework.get_data_quality_sla(instrument_type=key if
            key in ['forex', 'crypto', 'stocks'] else 'forex', data_type=
            key if key in ['ohlcv', 'tick'] else 'ohlcv')
    except Exception as e:
        logger.exception(f'Error updating data quality SLA: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Error updating data quality SLA: {str(e)}')
