"""
Market Regime Analysis API endpoints

This module provides API endpoints for detecting market regimes
and analyzing tool effectiveness across different market conditions.
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel
from analysis_engine.db.connection import get_db_session
from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository
from analysis_engine.services.market_regime_detector import MarketRegimeService
from analysis_engine.services.market_regime_analysis import MarketRegimeAnalysisService
from analysis_engine.core.exceptions_bridge import ForexTradingPlatformError, AnalysisError, MarketRegimeError, InsufficientDataError, get_correlation_id_from_request
from analysis_engine.core.logging import get_logger


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class DetectRegimeRequest(BaseModel):
    """
    DetectRegimeRequest class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    symbol: str
    timeframe: str
    ohlc_data: List[Dict]


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
    """
    RegimeHistoryRequest class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    symbol: str
    timeframe: str
    limit: Optional[int] = 10


class RegimeAnalysisRequest(BaseModel):
    """
    RegimeAnalysisRequest class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    tool_id: str
    timeframe: Optional[str] = None
    instrument: Optional[str] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None


class OptimalConditionsRequest(BaseModel):
    """
    OptimalConditionsRequest class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    tool_id: str
    min_sample_size: int = 10
    timeframe: Optional[str] = None
    instrument: Optional[str] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None


class ToolComplementarityRequest(BaseModel):
    """
    ToolComplementarityRequest class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    tool_ids: List[str]
    timeframe: Optional[str] = None
    instrument: Optional[str] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None


class RecommendationRequest(BaseModel):
    """
    RecommendationRequest class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    current_regime: str
    instrument: Optional[str] = None
    timeframe: Optional[str] = None
    min_sample_size: int = 10
    min_win_rate: float = 50.0
    top_n: int = 3


class TrendAnalysisRequest(BaseModel):
    """
    TrendAnalysisRequest class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    tool_id: str
    timeframe: Optional[str] = None
    instrument: Optional[str] = None
    period_days: int = 30
    look_back_periods: int = 6


class UnderperformingToolsRequest(BaseModel):
    """
    UnderperformingToolsRequest class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    win_rate_threshold: float = 50.0
    min_sample_size: int = 20
    timeframe: Optional[str] = None
    instrument: Optional[str] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None


class DashboardRequest(BaseModel):
    """
    DashboardRequest class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    timeframe: Optional[str] = None
    instrument: Optional[str] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None


router = APIRouter(prefix='/market-regime', tags=['market-regime'])
logger = get_logger(__name__)


@router.post('/detect/', response_model=Dict)
@async_with_exception_handling
async def detect_market_regime(request_data: DetectRegimeRequest, request:
    Request, db: Session=Depends(get_db_session)):
    """Detect the current market regime based on price data"""
    correlation_id = get_correlation_id_from_request(request)
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
        service = MarketRegimeService()
        regime_result = await service.detect_current_regime(symbol=
            request_data.symbol, timeframe=request_data.timeframe,
            price_data=df)
        logger.info(
            f'Market regime detected for {request_data.symbol}/{request_data.timeframe}'
            , extra={'correlation_id': correlation_id, 'symbol':
            request_data.symbol, 'timeframe': request_data.timeframe,
            'regime': regime_result.get('regime', 'unknown'), 'confidence':
            regime_result.get('confidence', 0.0)})
        return regime_result
    except InsufficientDataError:
        raise
    except ForexTradingPlatformError as e:
        raise
    except Exception as e:
        logger.error(
            f'Error detecting market regime for {request_data.symbol}/{request_data.timeframe}: {str(e)}'
            , extra={'correlation_id': correlation_id}, exc_info=True)
        raise MarketRegimeError(message=
            f'Failed to detect market regime for {request_data.symbol}/{request_data.timeframe}'
            , symbol=request_data.symbol, timeframe=request_data.timeframe,
            correlation_id=correlation_id)


@router.post('/history/', response_model=List[Dict])
@async_with_exception_handling
async def get_regime_history(request_data: RegimeHistoryRequest, request:
    Request, db: Session=Depends(get_db_session)):
    """Get historical regime data for a specific symbol and timeframe"""
    correlation_id = get_correlation_id_from_request(request)
    try:
        service = MarketRegimeService()
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


@router.post('/regime-analysis/', response_model=Dict)
@async_with_exception_handling
async def analyze_tool_regime_performance(request_data:
    RegimeAnalysisRequest, request: Request, db: Session=Depends(
    get_db_session)):
    """Get the performance metrics of a tool across different market regimes"""
    correlation_id = get_correlation_id_from_request(request)
    try:
        repository = ToolEffectivenessRepository(db)
        service = MarketRegimeAnalysisService(repository)
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


@router.post('/optimal-conditions/', response_model=Dict)
@async_with_exception_handling
async def find_optimal_market_conditions(request_data:
    OptimalConditionsRequest, request: Request, db: Session=Depends(
    get_db_session)):
    """Find the optimal market conditions for a specific tool"""
    correlation_id = get_correlation_id_from_request(request)
    try:
        repository = ToolEffectivenessRepository(db)
        service = MarketRegimeAnalysisService(repository)
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


@router.post('/complementarity/', response_model=Dict)
@async_with_exception_handling
async def analyze_tool_complementarity(request_data:
    ToolComplementarityRequest, request: Request, db: Session=Depends(
    get_db_session)):
    """Analyze how well different tools complement each other"""
    correlation_id = get_correlation_id_from_request(request)
    try:
        if not request_data.tool_ids or len(request_data.tool_ids) < 2:
            from analysis_engine.core.exceptions_bridge import InvalidAnalysisParametersError
            raise InvalidAnalysisParametersError(message=
                'At least two tool IDs are required for complementarity analysis'
                , parameters={'tool_ids': request_data.tool_ids},
                correlation_id=correlation_id)
        repository = ToolEffectivenessRepository(db)
        service = MarketRegimeAnalysisService(repository)
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


@router.post('/performance-report/', response_model=Dict)
@async_with_exception_handling
async def generate_performance_report(request_data: DashboardRequest,
    request: Request, db: Session=Depends(get_db_session)):
    """Generate a comprehensive performance report for all tools"""
    correlation_id = get_correlation_id_from_request(request)
    try:
        repository = ToolEffectivenessRepository(db)
        service = MarketRegimeAnalysisService(repository)
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


@router.post('/recommend-tools/', response_model=Dict)
@async_with_exception_handling
async def recommend_tools_for_regime(request_data: RecommendationRequest,
    request: Request, db: Session=Depends(get_db_session)):
    """Recommend the best trading tools for the current market regime"""
    correlation_id = get_correlation_id_from_request(request)
    try:
        if not request_data.current_regime:
            from analysis_engine.core.exceptions_bridge import InvalidAnalysisParametersError
            raise InvalidAnalysisParametersError(message=
                'Current market regime must be specified', parameters={
                'current_regime': request_data.current_regime},
                correlation_id=correlation_id)
        repository = ToolEffectivenessRepository(db)
        service = MarketRegimeAnalysisService(repository)
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


@router.post('/effectiveness-trends/', response_model=Dict)
@async_with_exception_handling
async def analyze_effectiveness_trends(request_data: TrendAnalysisRequest,
    request: Request, db: Session=Depends(get_db_session)):
    """Analyze how the effectiveness of a tool has changed over time"""
    correlation_id = get_correlation_id_from_request(request)
    try:
        if not request_data.tool_id:
            from analysis_engine.core.exceptions_bridge import InvalidAnalysisParametersError
            raise InvalidAnalysisParametersError(message=
                'Tool ID must be specified for trend analysis', parameters=
                {'tool_id': request_data.tool_id}, correlation_id=
                correlation_id)
        repository = ToolEffectivenessRepository(db)
        service = MarketRegimeAnalysisService(repository)
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


@router.post('/underperforming-tools/', response_model=Dict)
@async_with_exception_handling
async def get_underperforming_tools(request_data:
    UnderperformingToolsRequest, request: Request, db: Session=Depends(
    get_db_session)):
    """Identify underperforming trading tools that may need optimization or retirement"""
    correlation_id = get_correlation_id_from_request(request)
    try:
        repository = ToolEffectivenessRepository(db)
        service = MarketRegimeAnalysisService(repository)
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
