"""
Standardized Market Regime Analysis API for Analysis Engine Service.

This module provides standardized API endpoints for detecting market regimes
and analyzing tool effectiveness across different market conditions.

All endpoints follow the platform's standardized API design patterns.
"""
import logging
import time
from typing import Dict, List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Path
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from analysis_engine.monitoring.service_metrics import analysis_engine_metrics
from analysis_engine.db.connection import get_db_session
from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository
from analysis_engine.services import get_market_regime_service
from analysis_engine.services.market_regime_analysis import MarketRegimeAnalysisService
from analysis_engine.core.exceptions_bridge import ForexTradingPlatformError, AnalysisError, MarketRegimeError, InsufficientDataError, get_correlation_id_from_request
from analysis_engine.monitoring.structured_logging import get_structured_logger


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class DetectRegimeRequest(BaseModel):
    """Request model for detecting market regime"""
    symbol: str = Field(..., description="Trading symbol (e.g., 'EURUSD')")
    timeframe: str = Field(..., description=
        "Timeframe for analysis (e.g., '1h', '4h', 'D')")
    ohlc_data: List[Dict] = Field(..., description='OHLC price data')


    class Config:
    """
    Config class.
    
    Attributes:
        Add attributes here
    """

        schema_extra = {'example': {'symbol': 'EURUSD', 'timeframe': '1h',
            'ohlc_data': [{'timestamp': '2025-04-01T00:00:00', 'open': 
            1.0765, 'high': 1.078, 'low': 1.076, 'close': 1.0775, 'volume':
            1000}, {'timestamp': '2025-04-01T01:00:00', 'open': 1.0775,
            'high': 1.079, 'low': 1.077, 'close': 1.0785, 'volume': 1200}]}}


class RegimeHistoryRequest(BaseModel):
    """Request model for getting regime history"""
    symbol: str = Field(..., description="Trading symbol (e.g., 'EURUSD')")
    timeframe: str = Field(..., description=
        "Timeframe for analysis (e.g., '1h', '4h', 'D')")
    limit: Optional[int] = Field(10, description=
        'Maximum number of history entries to return')


class RegimeAnalysisRequest(BaseModel):
    """Request model for analyzing tool regime performance"""
    tool_id: str = Field(..., description='Identifier for the trading tool')
    timeframe: Optional[str] = Field(None, description=
        "Timeframe for analysis (e.g., '1h', '4h', 'D')")
    instrument: Optional[str] = Field(None, description=
        "Trading instrument (e.g., 'EUR_USD')")
    from_date: Optional[datetime] = Field(None, description=
        'Start date for analysis')
    to_date: Optional[datetime] = Field(None, description=
        'End date for analysis')


class OptimalConditionsRequest(BaseModel):
    """Request model for finding optimal market conditions"""
    tool_id: str = Field(..., description='Identifier for the trading tool')
    min_sample_size: int = Field(10, description=
        'Minimum sample size for reliable analysis')
    timeframe: Optional[str] = Field(None, description=
        "Timeframe for analysis (e.g., '1h', '4h', 'D')")
    instrument: Optional[str] = Field(None, description=
        "Trading instrument (e.g., 'EUR_USD')")
    from_date: Optional[datetime] = Field(None, description=
        'Start date for analysis')
    to_date: Optional[datetime] = Field(None, description=
        'End date for analysis')


class ToolComplementarityRequest(BaseModel):
    """Request model for analyzing tool complementarity"""
    tool_ids: List[str] = Field(..., description=
        'List of tool identifiers to analyze')
    timeframe: Optional[str] = Field(None, description=
        "Timeframe for analysis (e.g., '1h', '4h', 'D')")
    instrument: Optional[str] = Field(None, description=
        "Trading instrument (e.g., 'EUR_USD')")
    from_date: Optional[datetime] = Field(None, description=
        'Start date for analysis')
    to_date: Optional[datetime] = Field(None, description=
        'End date for analysis')


class RecommendationRequest(BaseModel):
    """Request model for recommending tools for a regime"""
    current_regime: str = Field(..., description='Current market regime')
    instrument: Optional[str] = Field(None, description=
        "Trading instrument (e.g., 'EUR_USD')")
    timeframe: Optional[str] = Field(None, description=
        "Timeframe for analysis (e.g., '1h', '4h', 'D')")
    min_sample_size: int = Field(10, description=
        'Minimum sample size for reliable analysis')
    min_win_rate: float = Field(50.0, description=
        'Minimum win rate for recommended tools')
    top_n: int = Field(3, description='Number of top tools to recommend')


class TrendAnalysisRequest(BaseModel):
    """Request model for analyzing effectiveness trends"""
    tool_id: str = Field(..., description='Identifier for the trading tool')
    timeframe: Optional[str] = Field(None, description=
        "Timeframe for analysis (e.g., '1h', '4h', 'D')")
    instrument: Optional[str] = Field(None, description=
        "Trading instrument (e.g., 'EUR_USD')")
    period_days: int = Field(30, description='Number of days to analyze')
    look_back_periods: int = Field(6, description=
        'Number of periods to look back')


class UnderperformingToolsRequest(BaseModel):
    """Request model for getting underperforming tools"""
    win_rate_threshold: float = Field(50.0, description=
        'Win rate threshold for underperforming tools')
    min_sample_size: int = Field(20, description=
        'Minimum sample size for reliable analysis')
    timeframe: Optional[str] = Field(None, description=
        "Timeframe for analysis (e.g., '1h', '4h', 'D')")
    instrument: Optional[str] = Field(None, description=
        "Trading instrument (e.g., 'EUR_USD')")
    from_date: Optional[datetime] = Field(None, description=
        'Start date for analysis')
    to_date: Optional[datetime] = Field(None, description=
        'End date for analysis')


class DashboardRequest(BaseModel):
    """Request model for generating performance report"""
    timeframe: Optional[str] = Field(None, description=
        "Timeframe for analysis (e.g., '1h', '4h', 'D')")
    instrument: Optional[str] = Field(None, description=
        "Trading instrument (e.g., 'EUR_USD')")
    from_date: Optional[datetime] = Field(None, description=
        'Start date for analysis')
    to_date: Optional[datetime] = Field(None, description=
        'End date for analysis')


def get_db_session_dependency():
    """Get database session dependency"""
    return get_db_session()


def get_market_regime_service_dependency():
    """Get market regime service dependency"""
    from analysis_engine.services import get_market_regime_service as get_service
    return get_service(use_standardized=True)


def get_market_regime_analysis_service(db: Session=Depends(
    get_db_session_dependency)):
    """Get market regime analysis service dependency"""
    repository = ToolEffectivenessRepository(db)
    return MarketRegimeAnalysisService(repository)


router = APIRouter(prefix='/v1/analysis/market-regimes', tags=[
    'Market Regime Analysis'])
logger = get_structured_logger(__name__)


@router.post('/detect', response_model=Dict, summary='Detect market regime',
    description='Detect the current market regime based on price data.')
@async_with_exception_handling
async def detect_market_regime(request_data: DetectRegimeRequest, request:
    Request, service: MarketRegimeService=Depends(
    get_market_regime_service_dependency)):
    """
    Detect the current market regime based on price data.
    """
    correlation_id = get_correlation_id_from_request(request)
    analysis_engine_metrics.market_regime_detection_operations_total.labels(
        service='analysis-engine-service', instrument=request_data.symbol,
        timeframe=request_data.timeframe).inc()
    start_time = time.time()
    try:
        if not request_data.ohlc_data or len(request_data.ohlc_data) == 0:
            raise InsufficientDataError(message=
                'No OHLC data provided for market regime detection', symbol
                =request_data.symbol, timeframe=request_data.timeframe,
                correlation_id=correlation_id)
        import pandas as pd
        df = pd.DataFrame(request_data.ohlc_data)
        if len(df) < 20:
            raise InsufficientDataError(message=
                'Insufficient data for market regime detection', symbol=
                request_data.symbol, timeframe=request_data.timeframe,
                available_points=len(df), required_points=20,
                correlation_id=correlation_id)
        regime_result = await service.detect_current_regime(symbol=
            request_data.symbol, timeframe=request_data.timeframe,
            price_data=df)
        logger.info(
            f'Market regime detected for {request_data.symbol}/{request_data.timeframe}'
            , extra={'correlation_id': correlation_id, 'symbol':
            request_data.symbol, 'timeframe': request_data.timeframe,
            'regime': regime_result.get('regime', 'unknown'), 'confidence':
            regime_result.get('confidence', 0.0)})
        duration = time.time() - start_time
        analysis_engine_metrics.market_regime_detection_duration_seconds.labels(
            service='analysis-engine-service', instrument=request_data.
            symbol, timeframe=request_data.timeframe).observe(duration)
        regime_type = regime_result.get('regime', 'unknown')
        analysis_engine_metrics.market_regime.labels(service=
            'analysis-engine-service', instrument=request_data.symbol,
            timeframe=request_data.timeframe, regime_type=regime_type).set(1.0)
        confidence = regime_result.get('confidence', 0.0)
        analysis_engine_metrics.market_regime_confidence.labels(service=
            'analysis-engine-service', instrument=request_data.symbol,
            timeframe=request_data.timeframe, regime_type=regime_type).set(
            confidence)
        return regime_result
    except InsufficientDataError:
        analysis_engine_metrics.analysis_errors_total.labels(service=
            'analysis-engine-service', analysis_type=
            'market_regime_detection', error_type='insufficient_data').inc()
        duration = time.time() - start_time
        analysis_engine_metrics.market_regime_detection_duration_seconds.labels(
            service='analysis-engine-service', instrument=request_data.
            symbol, timeframe=request_data.timeframe).observe(duration)
        raise
    except ForexTradingPlatformError as e:
        analysis_engine_metrics.analysis_errors_total.labels(service=
            'analysis-engine-service', analysis_type=
            'market_regime_detection', error_type=type(e).__name__).inc()
        duration = time.time() - start_time
        analysis_engine_metrics.market_regime_detection_duration_seconds.labels(
            service='analysis-engine-service', instrument=request_data.
            symbol, timeframe=request_data.timeframe).observe(duration)
        raise
    except Exception as e:
        analysis_engine_metrics.analysis_errors_total.labels(service=
            'analysis-engine-service', analysis_type=
            'market_regime_detection', error_type='unexpected_error').inc()
        duration = time.time() - start_time
        analysis_engine_metrics.market_regime_detection_duration_seconds.labels(
            service='analysis-engine-service', instrument=request_data.
            symbol, timeframe=request_data.timeframe).observe(duration)
        logger.error(
            f'Error detecting market regime for {request_data.symbol}/{request_data.timeframe}: {str(e)}'
            , extra={'correlation_id': correlation_id}, exc_info=True)
        raise MarketRegimeError(message=
            f'Failed to detect market regime for {request_data.symbol}/{request_data.timeframe}'
            , symbol=request_data.symbol, timeframe=request_data.timeframe,
            correlation_id=correlation_id)


@router.post('/history', response_model=List[Dict], summary=
    'Get regime history', description=
    'Get historical regime data for a specific symbol and timeframe.')
@async_with_exception_handling
async def get_regime_history(request_data: RegimeHistoryRequest, request:
    Request, service: MarketRegimeService=Depends(
    get_market_regime_service_dependency)):
    """
    Get historical regime data for a specific symbol and timeframe.
    """
    correlation_id = get_correlation_id_from_request(request)
    try:
        history = await service.get_regime_history(symbol=request_data.
            symbol, timeframe=request_data.timeframe, limit=request_data.limit)
        logger.info(
            f'Retrieved regime history for {request_data.symbol}/{request_data.timeframe}'
            , extra={'correlation_id': correlation_id, 'symbol':
            request_data.symbol, 'timeframe': request_data.timeframe,
            'limit': request_data.limit, 'records_found': len(history)})
        return history
    except ForexTradingPlatformError:
        raise
    except Exception as e:
        logger.error(
            f'Error getting regime history for {request_data.symbol}/{request_data.timeframe}: {str(e)}'
            , extra={'correlation_id': correlation_id}, exc_info=True)
        raise MarketRegimeError(message=
            f'Failed to get regime history for {request_data.symbol}/{request_data.timeframe}'
            , symbol=request_data.symbol, timeframe=request_data.timeframe,
            correlation_id=correlation_id)


@router.post('/tools/regime-analysis', response_model=Dict, summary=
    'Analyze tool regime performance', description=
    'Get the performance metrics of a tool across different market regimes.')
@async_with_exception_handling
async def analyze_tool_regime_performance(request_data:
    RegimeAnalysisRequest, request: Request, service:
    MarketRegimeAnalysisService=Depends(get_market_regime_analysis_service)):
    """
    Get the performance metrics of a tool across different market regimes.
    """
    correlation_id = get_correlation_id_from_request(request)
    try:
        result = await service.get_regime_performance_matrix(tool_id=
            request_data.tool_id, timeframe=request_data.timeframe,
            instrument=request_data.instrument, from_date=request_data.
            from_date, to_date=request_data.to_date)
        logger.info(
            f'Analyzed regime performance for tool {request_data.tool_id}',
            extra={'correlation_id': correlation_id, 'tool_id':
            request_data.tool_id, 'timeframe': request_data.timeframe,
            'instrument': request_data.instrument, 'regimes_analyzed': len(
            result.get('regimes', {}))})
        return result
    except ForexTradingPlatformError:
        raise
    except Exception as e:
        logger.error(
            f'Error analyzing regime performance for tool {request_data.tool_id}: {str(e)}'
            , extra={'correlation_id': correlation_id}, exc_info=True)
        from analysis_engine.core.exceptions_bridge import ToolEffectivenessError
        raise ToolEffectivenessError(message=
            f'Failed to analyze regime performance for tool {request_data.tool_id}'
            , tool_id=request_data.tool_id, correlation_id=correlation_id)


@router.post('/tools/optimal-conditions', response_model=Dict, summary=
    'Find optimal market conditions', description=
    'Find the optimal market conditions for a specific tool.')
@async_with_exception_handling
async def find_optimal_market_conditions(request_data:
    OptimalConditionsRequest, request: Request, service:
    MarketRegimeAnalysisService=Depends(get_market_regime_analysis_service)):
    """
    Find the optimal market conditions for a specific tool.
    """
    correlation_id = get_correlation_id_from_request(request)
    try:
        result = await service.find_optimal_market_conditions(tool_id=
            request_data.tool_id, min_sample_size=request_data.
            min_sample_size, timeframe=request_data.timeframe, instrument=
            request_data.instrument, from_date=request_data.from_date,
            to_date=request_data.to_date)
        logger.info(
            f'Found optimal market conditions for tool {request_data.tool_id}',
            extra={'correlation_id': correlation_id, 'tool_id':
            request_data.tool_id, 'timeframe': request_data.timeframe,
            'instrument': request_data.instrument, 'best_regime': result.
            get('best_regime', {}).get('regime')})
        return result
    except ForexTradingPlatformError:
        raise
    except Exception as e:
        logger.error(
            f'Error finding optimal market conditions for tool {request_data.tool_id}: {str(e)}'
            , extra={'correlation_id': correlation_id}, exc_info=True)
        from analysis_engine.core.exceptions_bridge import ToolEffectivenessError
        raise ToolEffectivenessError(message=
            f'Failed to find optimal market conditions for tool {request_data.tool_id}'
            , tool_id=request_data.tool_id, correlation_id=correlation_id)


@router.post('/tools/complementarity', response_model=Dict, summary=
    'Analyze tool complementarity', description=
    'Analyze how well different tools complement each other across market regimes.'
    )
@async_with_exception_handling
async def analyze_tool_complementarity(request_data:
    ToolComplementarityRequest, request: Request, service:
    MarketRegimeAnalysisService=Depends(get_market_regime_analysis_service)):
    """
    Analyze how well different tools complement each other across market regimes.
    """
    correlation_id = get_correlation_id_from_request(request)
    try:
        if not request_data.tool_ids or len(request_data.tool_ids) < 2:
            from analysis_engine.core.exceptions_bridge import InvalidAnalysisParametersError
            raise InvalidAnalysisParametersError(message=
                'At least two tool IDs are required for complementarity analysis'
                , parameters={'tool_ids': request_data.tool_ids},
                correlation_id=correlation_id)
        result = await service.compute_tool_complementarity(tool_ids=
            request_data.tool_ids, timeframe=request_data.timeframe,
            instrument=request_data.instrument, from_date=request_data.
            from_date, to_date=request_data.to_date)
        logger.info(
            f'Analyzed complementarity for {len(request_data.tool_ids)} tools',
            extra={'correlation_id': correlation_id, 'tool_ids':
            request_data.tool_ids, 'timeframe': request_data.timeframe,
            'instrument': request_data.instrument})
        return result
    except ForexTradingPlatformError:
        raise
    except Exception as e:
        logger.error(f'Error analyzing tool complementarity: {str(e)}',
            extra={'correlation_id': correlation_id}, exc_info=True)
        from analysis_engine.core.exceptions_bridge import ToolEffectivenessError
        raise ToolEffectivenessError(message=
            'Failed to analyze tool complementarity', tool_id=','.join(
            request_data.tool_ids) if request_data.tool_ids else 'unknown',
            correlation_id=correlation_id)


@router.post('/performance-report', response_model=Dict, summary=
    'Generate performance report', description=
    'Generate a comprehensive performance report for all tools across market regimes.'
    )
@async_with_exception_handling
async def generate_performance_report(request_data: DashboardRequest,
    request: Request, service: MarketRegimeAnalysisService=Depends(
    get_market_regime_analysis_service)):
    """
    Generate a comprehensive performance report for all tools across market regimes.
    """
    correlation_id = get_correlation_id_from_request(request)
    try:
        result = await service.generate_performance_report(timeframe=
            request_data.timeframe, instrument=request_data.instrument,
            from_date=request_data.from_date, to_date=request_data.to_date)
        logger.info(
            f'Generated performance report for {request_data.instrument}/{request_data.timeframe}'
            , extra={'correlation_id': correlation_id, 'timeframe':
            request_data.timeframe, 'instrument': request_data.instrument,
            'from_date': request_data.from_date, 'to_date': request_data.
            to_date, 'tools_analyzed': len(result.get('tools', []))})
        return result
    except ForexTradingPlatformError:
        raise
    except Exception as e:
        logger.error(f'Error generating performance report: {str(e)}',
            extra={'correlation_id': correlation_id}, exc_info=True)
        raise AnalysisError(message='Failed to generate performance report',
            error_code='PERFORMANCE_REPORT_ERROR', details={'timeframe':
            request_data.timeframe, 'instrument': request_data.instrument,
            'from_date': str(request_data.from_date), 'to_date': str(
            request_data.to_date), 'error': str(e)}, correlation_id=
            correlation_id)


@router.post('/tools/recommendations', response_model=Dict, summary=
    'Recommend tools for regime', description=
    'Recommend the best trading tools for the current market regime.')
@async_with_exception_handling
async def recommend_tools_for_regime(request_data: RecommendationRequest,
    request: Request, service: MarketRegimeAnalysisService=Depends(
    get_market_regime_analysis_service)):
    """
    Recommend the best trading tools for the current market regime.
    """
    correlation_id = get_correlation_id_from_request(request)
    try:
        if not request_data.current_regime:
            from analysis_engine.core.exceptions_bridge import InvalidAnalysisParametersError
            raise InvalidAnalysisParametersError(message=
                'Current market regime must be specified', parameters={
                'current_regime': request_data.current_regime},
                correlation_id=correlation_id)
        result = await service.recommend_tools_for_current_regime(
            current_regime=request_data.current_regime, instrument=
            request_data.instrument, timeframe=request_data.timeframe,
            min_sample_size=request_data.min_sample_size, min_win_rate=
            request_data.min_win_rate, top_n=request_data.top_n)
        logger.info(
            f'Recommended tools for {request_data.current_regime} regime',
            extra={'correlation_id': correlation_id, 'current_regime':
            request_data.current_regime, 'instrument': request_data.
            instrument, 'timeframe': request_data.timeframe,
            'recommended_tools': len(result.get('recommended_tools', []))})
        return result
    except ForexTradingPlatformError:
        raise
    except Exception as e:
        logger.error(
            f'Error recommending tools for {request_data.current_regime} regime: {str(e)}'
            , extra={'correlation_id': correlation_id}, exc_info=True)
        raise MarketRegimeError(message=
            f'Failed to recommend tools for {request_data.current_regime} regime'
            , correlation_id=correlation_id)


@router.post('/tools/effectiveness-trends', response_model=Dict, summary=
    'Analyze effectiveness trends', description=
    'Analyze how the effectiveness of a tool has changed over time across market regimes.'
    )
@async_with_exception_handling
async def analyze_effectiveness_trends(request_data: TrendAnalysisRequest,
    request: Request, service: MarketRegimeAnalysisService=Depends(
    get_market_regime_analysis_service)):
    """
    Analyze how the effectiveness of a tool has changed over time across market regimes.
    """
    correlation_id = get_correlation_id_from_request(request)
    try:
        if not request_data.tool_id:
            from analysis_engine.core.exceptions_bridge import InvalidAnalysisParametersError
            raise InvalidAnalysisParametersError(message=
                'Tool ID must be specified for trend analysis', parameters=
                {'tool_id': request_data.tool_id}, correlation_id=
                correlation_id)
        result = await service.analyze_effectiveness_trends(tool_id=
            request_data.tool_id, timeframe=request_data.timeframe,
            instrument=request_data.instrument, period_days=request_data.
            period_days, look_back_periods=request_data.look_back_periods)
        logger.info(
            f'Analyzed effectiveness trends for tool {request_data.tool_id}',
            extra={'correlation_id': correlation_id, 'tool_id':
            request_data.tool_id, 'timeframe': request_data.timeframe,
            'instrument': request_data.instrument, 'period_days':
            request_data.period_days, 'look_back_periods': request_data.
            look_back_periods})
        return result
    except ForexTradingPlatformError:
        raise
    except Exception as e:
        logger.error(
            f'Error analyzing effectiveness trends for tool {request_data.tool_id}: {str(e)}'
            , extra={'correlation_id': correlation_id}, exc_info=True)
        from analysis_engine.core.exceptions_bridge import ToolEffectivenessError
        raise ToolEffectivenessError(message=
            f'Failed to analyze effectiveness trends for tool {request_data.tool_id}'
            , tool_id=request_data.tool_id, correlation_id=correlation_id)


@router.post('/tools/underperforming', response_model=Dict, summary=
    'Get underperforming tools', description=
    'Identify underperforming trading tools that may need optimization or retirement.'
    )
@async_with_exception_handling
async def get_underperforming_tools(request_data:
    UnderperformingToolsRequest, request: Request, service:
    MarketRegimeAnalysisService=Depends(get_market_regime_analysis_service)):
    """
    Identify underperforming trading tools that may need optimization or retirement.
    """
    correlation_id = get_correlation_id_from_request(request)
    try:
        result = await service.get_underperforming_tools(win_rate_threshold
            =request_data.win_rate_threshold, min_sample_size=request_data.
            min_sample_size, timeframe=request_data.timeframe, instrument=
            request_data.instrument, from_date=request_data.from_date,
            to_date=request_data.to_date)
        logger.info(
            f'Identified underperforming tools for {request_data.instrument}/{request_data.timeframe}'
            , extra={'correlation_id': correlation_id, 'timeframe':
            request_data.timeframe, 'instrument': request_data.instrument,
            'win_rate_threshold': request_data.win_rate_threshold,
            'min_sample_size': request_data.min_sample_size,
            'underperforming_tools_count': len(result.get(
            'underperforming_tools', []))})
        return result
    except ForexTradingPlatformError:
        raise
    except Exception as e:
        logger.error(f'Error getting underperforming tools: {str(e)}',
            extra={'correlation_id': correlation_id}, exc_info=True)
        from analysis_engine.core.exceptions_bridge import ToolEffectivenessError
        raise ToolEffectivenessError(message=
            'Failed to identify underperforming tools', correlation_id=
            correlation_id)


legacy_router = APIRouter(prefix='/market-regime', tags=[
    'Market Regime (Legacy)'])


@legacy_router.post('/detect/', response_model=Dict)
async def legacy_detect_market_regime(request_data: DetectRegimeRequest,
    request: Request, service: MarketRegimeService=Depends(
    get_market_regime_service_dependency)):
    """
    Legacy endpoint for detecting market regime.
    Consider migrating to /api/v1/analysis/market-regimes/detect
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/market-regimes/detect'
        )
    return await detect_market_regime(request_data, request, service)


@legacy_router.post('/history/', response_model=List[Dict])
async def legacy_get_regime_history(request_data: RegimeHistoryRequest,
    request: Request, service: MarketRegimeService=Depends(
    get_market_regime_service_dependency)):
    """
    Legacy endpoint for getting regime history.
    Consider migrating to /api/v1/analysis/market-regimes/history
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/market-regimes/history'
        )
    return await get_regime_history(request_data, request, service)


@legacy_router.post('/regime-analysis/', response_model=Dict)
async def legacy_analyze_tool_regime_performance(request_data:
    RegimeAnalysisRequest, request: Request, service:
    MarketRegimeAnalysisService=Depends(get_market_regime_analysis_service)):
    """
    Legacy endpoint for analyzing tool regime performance.
    Consider migrating to /api/v1/analysis/market-regimes/tools/regime-analysis
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/market-regimes/tools/regime-analysis'
        )
    return await analyze_tool_regime_performance(request_data, request, service
        )


def setup_market_regime_routes(app: FastAPI) ->None:
    """
    Set up market regime routes.

    Args:
        app: FastAPI application
    """
    app.include_router(router, prefix='/api')
    app.include_router(legacy_router)
