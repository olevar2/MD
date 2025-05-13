"""
Standardized Signal Quality API for Analysis Engine Service.

This module provides standardized API endpoints for evaluating signal quality,
analyzing the relationship between signal quality and outcomes, and tracking
quality trends over time.

All endpoints follow the platform's standardized API design patterns.
"""
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from analysis_engine.monitoring.service_metrics import analysis_engine_metrics
from analysis_engine.services.signal_quality_evaluator import SignalQualityEvaluator, SignalQualityAnalyzer
from analysis_engine.services.tool_effectiveness import SignalEvent, MarketRegime
from analysis_engine.core.exceptions_bridge import ForexTradingPlatformError, AnalysisError, SignalQualityError, InsufficientDataError, get_correlation_id_from_request
from analysis_engine.db.connection import get_db_session
from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository
from analysis_engine.monitoring.structured_logging import get_structured_logger


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class QualityEvaluationRequest(BaseModel):
    """Request model for evaluating signal quality"""
    signal_id: str = Field(..., description='ID of the signal to evaluate')
    market_context: Optional[Dict[str, Any]] = Field(default={},
        description='Additional market context')
    historical_data: Optional[Dict[str, Any]] = Field(default={},
        description='Historical performance data')


    class Config:
    """
    Config class.
    
    Attributes:
        Add attributes here
    """

        schema_extra = {'example': {'signal_id': 'sig_12345',
            'market_context': {'market_regime': 'trending', 'volatility':
            'medium', 'evaluate_confluence': True}, 'historical_data': {
            'win_rate': 0.65, 'average_profit': 1.2, 'average_loss': 0.8}}}


class SignalQualityResponse(BaseModel):
    """Response model for signal quality evaluation"""
    signal_id: str
    tool_id: str
    base_quality: float
    timing_quality: float
    confluence: Optional[float]
    historical_reliability: Optional[float]
    regime_compatibility: Optional[float]
    overall_quality: float
    evaluation_timestamp: datetime


class QualityAnalysisRequest(BaseModel):
    """Request model for analyzing signal quality"""
    tool_id: Optional[str] = Field(None, description=
        'Filter by specific tool ID')
    timeframe: Optional[str] = Field(None, description=
        'Filter by specific timeframe')
    market_regime: Optional[str] = Field(None, description=
        'Filter by market regime')
    days: Optional[int] = Field(30, description='Number of days to analyze')


class QualityAnalysisResponse(BaseModel):
    """Response model for quality analysis"""
    average_quality: float
    average_success_rate: float
    correlation: float
    sample_size: int
    quality_brackets: List[Dict[str, Any]]
    tool_id: Optional[str]
    timeframe: Optional[str]
    market_regime: Optional[str]


class QualityTrendRequest(BaseModel):
    """Request model for analyzing quality trends"""
    tool_id: str = Field(..., description='Tool ID to analyze')
    window_size: int = Field(20, description=
        'Size of moving window for trend analysis')
    days: int = Field(90, description='Number of days to analyze')


class QualityTrendResponse(BaseModel):
    """Response model for quality trend analysis"""
    quality_trend: float
    success_trend: float
    data_points: int
    window_size: int
    moving_averages: List[Dict[str, Any]]
    tool_id: Optional[str]


def get_db_session_dependency():
    """Get database session dependency"""
    return get_db_session()


def get_repository(db: Session=Depends(get_db_session_dependency)):
    """Get tool effectiveness repository dependency"""
    return ToolEffectivenessRepository(db)


def get_quality_evaluator():
    """Get signal quality evaluator dependency"""
    return SignalQualityEvaluator()


def get_quality_analyzer():
    """Get signal quality analyzer dependency"""
    return SignalQualityAnalyzer()


router = APIRouter(prefix='/v1/analysis/signal-quality', tags=[
    'Signal Quality'])
logger = get_structured_logger(__name__)


@router.post('/signals/{signal_id}/evaluate', response_model=
    SignalQualityResponse, summary='Evaluate signal quality', description=
    'Evaluate the quality of a specific trading signal and store the results.')
@async_with_exception_handling
async def evaluate_signal_quality(signal_id: str=Path(..., description=
    'ID of the signal to evaluate'), request: QualityEvaluationRequest=None,
    request_obj: Request=None, repository: ToolEffectivenessRepository=
    Depends(get_repository), evaluator: SignalQualityEvaluator=Depends(
    get_quality_evaluator)):
    """
    Evaluate the quality of a specific trading signal and store the results.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    analysis_engine_metrics.signal_generation_operations_total.labels(service
        ='analysis-engine-service', signal_type='quality_evaluation',
        instrument=request.market_context.get('instrument', 'unknown'),
        timeframe=request.market_context.get('timeframe', 'unknown')).inc()
    start_time = time.time()
    try:
        if request is None:
            request = QualityEvaluationRequest(signal_id=signal_id)
        signal = repository.get_signal(signal_id)
        if not signal:
            raise InsufficientDataError(message=
                f'Signal not found: {signal_id}', correlation_id=correlation_id
                )
        signal_event = SignalEvent(tool_name=signal.tool_id, signal_type=
            signal.signal_type, direction=signal.signal_type, strength=
            signal.confidence, timestamp=signal.timestamp, symbol=signal.
            instrument, timeframe=signal.timeframe, price_at_signal=0.0,
            metadata=signal.additional_data or {}, market_context=request.
            market_context or {})
        additional_signals = []
        if ('evaluate_confluence' in request.market_context and request.
            market_context['evaluate_confluence']):
            window_start = signal.timestamp - timedelta(hours=1)
            window_end = signal.timestamp + timedelta(hours=1)
            other_signals = repository.get_signals(instrument=signal.
                instrument, from_date=window_start, to_date=window_end,
                limit=50)
            for s in other_signals:
                if str(s.signal_id) != signal_id:
                    additional_signals.append(SignalEvent(tool_name=s.
                        tool_id, signal_type=s.signal_type, direction=s.
                        signal_type, strength=s.confidence, timestamp=s.
                        timestamp, symbol=s.instrument, timeframe=s.
                        timeframe, price_at_signal=0.0, metadata=s.
                        additional_data or {}, market_context=s.
                        additional_data.get('market_context', {}) if s.
                        additional_data else {}))
        quality_metrics = evaluator.evaluate_signal_quality(signal=
            signal_event, market_context=request.market_context,
            additional_signals=additional_signals if additional_signals else
            None, historical_performance=request.historical_data)
        updated_additional_data = signal.additional_data or {}
        updated_additional_data['quality'] = quality_metrics
        updated_additional_data['quality_evaluated_at'] = datetime.utcnow(
            ).isoformat()
        repository.update_tool(signal.tool_id, {'additional_data':
            updated_additional_data})
        logger.info(f'Evaluated quality for signal {signal_id}', extra={
            'correlation_id': correlation_id, 'signal_id': signal_id,
            'tool_id': signal.tool_id, 'overall_quality': quality_metrics[
            'overall_quality']})
        response = SignalQualityResponse(signal_id=signal_id, tool_id=
            signal.tool_id, base_quality=quality_metrics['base_quality'],
            timing_quality=quality_metrics['timing_quality'], confluence=
            quality_metrics.get('confluence'), historical_reliability=
            quality_metrics.get('historical_reliability'),
            regime_compatibility=quality_metrics.get('regime_compatibility'
            ), overall_quality=quality_metrics['overall_quality'],
            evaluation_timestamp=datetime.utcnow())
        duration = time.time() - start_time
        analysis_engine_metrics.signal_generation_duration_seconds.labels(
            service='analysis-engine-service', signal_type=
            'quality_evaluation', instrument=request.market_context.get(
            'instrument', 'unknown'), timeframe=request.market_context.get(
            'timeframe', 'unknown')).observe(duration)
        analysis_engine_metrics.signal_quality_score.labels(service=
            'analysis-engine-service', signal_type='quality_evaluation',
            instrument=request.market_context.get('instrument', 'unknown'),
            timeframe=request.market_context.get('timeframe', 'unknown')).set(
            quality_metrics['overall_quality'])
        return response
    except InsufficientDataError:
        analysis_engine_metrics.analysis_errors_total.labels(service=
            'analysis-engine-service', analysis_type=
            'signal_quality_evaluation', error_type='insufficient_data').inc()
        duration = time.time() - start_time
        analysis_engine_metrics.signal_generation_duration_seconds.labels(
            service='analysis-engine-service', signal_type=
            'quality_evaluation', instrument=request.market_context.get(
            'instrument', 'unknown'), timeframe=request.market_context.get(
            'timeframe', 'unknown')).observe(duration)
        raise
    except ForexTradingPlatformError as e:
        analysis_engine_metrics.analysis_errors_total.labels(service=
            'analysis-engine-service', analysis_type=
            'signal_quality_evaluation', error_type=type(e).__name__).inc()
        duration = time.time() - start_time
        analysis_engine_metrics.signal_generation_duration_seconds.labels(
            service='analysis-engine-service', signal_type=
            'quality_evaluation', instrument=request.market_context.get(
            'instrument', 'unknown'), timeframe=request.market_context.get(
            'timeframe', 'unknown')).observe(duration)
        raise
    except Exception as e:
        analysis_engine_metrics.analysis_errors_total.labels(service=
            'analysis-engine-service', analysis_type=
            'signal_quality_evaluation', error_type='unexpected_error').inc()
        duration = time.time() - start_time
        analysis_engine_metrics.signal_generation_duration_seconds.labels(
            service='analysis-engine-service', signal_type=
            'quality_evaluation', instrument=request.market_context.get(
            'instrument', 'unknown'), timeframe=request.market_context.get(
            'timeframe', 'unknown')).observe(duration)
        logger.error(
            f'Error evaluating signal quality for {signal_id}: {str(e)}',
            extra={'correlation_id': correlation_id}, exc_info=True)
        raise SignalQualityError(message=
            f'Failed to evaluate signal quality for {signal_id}',
            correlation_id=correlation_id)


@router.post('/analyze', response_model=QualityAnalysisResponse, summary=
    'Analyze signal quality', description=
    'Analyze the relationship between signal quality and outcomes.')
@async_with_exception_handling
async def analyze_signal_quality(request: QualityAnalysisRequest,
    request_obj: Request=None, repository: ToolEffectivenessRepository=
    Depends(get_repository), analyzer: SignalQualityAnalyzer=Depends(
    get_quality_analyzer)):
    """
    Analyze the relationship between signal quality and outcomes.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=request.days)
        signals = repository.get_signals(tool_id=request.tool_id, timeframe
            =request.timeframe, market_regime=request.market_regime,
            from_date=start_date, to_date=end_date, limit=1000)
        if not signals:
            raise InsufficientDataError(message=
                'No signals found matching the criteria', correlation_id=
                correlation_id)
        signal_ids = [str(s.signal_id) for s in signals]
        signal_outcomes = []
        for signal_id in signal_ids:
            signal = repository.get_signal(signal_id)
            outcomes = repository.get_outcomes_for_signal(signal_id)
            for outcome in outcomes:
                signal_event = SignalEvent(tool_name=signal.tool_id,
                    signal_type=signal.signal_type, direction=signal.
                    signal_type, strength=signal.confidence, timestamp=
                    signal.timestamp, symbol=signal.instrument, timeframe=
                    signal.timeframe, price_at_signal=0.0, metadata=signal.
                    additional_data or {}, market_context={})
                signal_outcome = SignalOutcome(signal_event=signal_event,
                    outcome='success' if outcome.success else 'failure',
                    exit_price=None, exit_timestamp=outcome.timestamp,
                    profit_loss=outcome.realized_profit)
                signal_outcomes.append(signal_outcome)
        if not signal_outcomes:
            raise InsufficientDataError(message=
                'No outcomes found for the signals', correlation_id=
                correlation_id)
        analysis_result = analyzer.analyze_quality_vs_outcomes(signal_outcomes)
        if 'error' in analysis_result:
            raise AnalysisError(message=analysis_result['error'],
                correlation_id=correlation_id)
        logger.info(
            f'Analyzed signal quality for {len(signal_outcomes)} signals',
            extra={'correlation_id': correlation_id, 'tool_id': request.
            tool_id, 'timeframe': request.timeframe, 'market_regime':
            request.market_regime, 'sample_size': analysis_result[
            'sample_size'], 'correlation': analysis_result['correlation']})
        response = QualityAnalysisResponse(average_quality=analysis_result[
            'average_quality'], average_success_rate=analysis_result[
            'average_success_rate'], correlation=analysis_result[
            'correlation'], sample_size=analysis_result['sample_size'],
            quality_brackets=analysis_result['quality_brackets'], tool_id=
            request.tool_id, timeframe=request.timeframe, market_regime=
            request.market_regime)
        return response
    except InsufficientDataError:
        raise
    except ForexTradingPlatformError:
        raise
    except Exception as e:
        logger.error(f'Error analyzing signal quality: {str(e)}', extra={
            'correlation_id': correlation_id}, exc_info=True)
        raise SignalQualityError(message='Failed to analyze signal quality',
            correlation_id=correlation_id)


@router.post('/trends', response_model=QualityTrendResponse, summary=
    'Analyze quality trends', description=
    'Analyze trends in signal quality over time for a specific tool.')
@async_with_exception_handling
async def analyze_quality_trends(request: QualityTrendRequest, request_obj:
    Request=None, repository: ToolEffectivenessRepository=Depends(
    get_repository), analyzer: SignalQualityAnalyzer=Depends(
    get_quality_analyzer)):
    """
    Analyze trends in signal quality over time for a specific tool.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=request.days)
        signals = repository.get_signals(tool_id=request.tool_id, from_date
            =start_date, to_date=end_date, limit=1000)
        if not signals or len(signals) < request.window_size:
            raise InsufficientDataError(message=
                f'Insufficient signals found. Need at least {request.window_size}, found {len(signals) if signals else 0}'
                , correlation_id=correlation_id)
        signal_ids = [str(s.signal_id) for s in signals]
        signal_outcomes = []
        for signal_id in signal_ids:
            signal = repository.get_signal(signal_id)
            outcomes = repository.get_outcomes_for_signal(signal_id)
            for outcome in outcomes:
                signal_event = SignalEvent(tool_name=signal.tool_id,
                    signal_type=signal.signal_type, direction=signal.
                    signal_type, strength=signal.confidence, timestamp=
                    signal.timestamp, symbol=signal.instrument, timeframe=
                    signal.timeframe, price_at_signal=0.0, metadata=signal.
                    additional_data or {}, market_context={})
                signal_outcome = SignalOutcome(signal_event=signal_event,
                    outcome='success' if outcome.success else 'failure',
                    exit_price=None, exit_timestamp=outcome.timestamp,
                    profit_loss=outcome.realized_profit)
                signal_outcomes.append(signal_outcome)
        trend_result = analyzer.analyze_quality_trends(signal_outcomes,
            request.window_size)
        if 'error' in trend_result:
            raise AnalysisError(message=trend_result['error'],
                correlation_id=correlation_id)
        logger.info(f'Analyzed quality trends for tool {request.tool_id}',
            extra={'correlation_id': correlation_id, 'tool_id': request.
            tool_id, 'window_size': request.window_size, 'days': request.
            days, 'data_points': trend_result['data_points'],
            'quality_trend': trend_result['quality_trend'], 'success_trend':
            trend_result['success_trend']})
        response = QualityTrendResponse(quality_trend=trend_result[
            'quality_trend'], success_trend=trend_result['success_trend'],
            data_points=trend_result['data_points'], window_size=
            trend_result['window_size'], moving_averages=trend_result[
            'moving_averages'], tool_id=request.tool_id)
        return response
    except InsufficientDataError:
        raise
    except ForexTradingPlatformError:
        raise
    except Exception as e:
        logger.error(
            f'Error analyzing quality trends for tool {request.tool_id}: {str(e)}'
            , extra={'correlation_id': correlation_id}, exc_info=True)
        raise SignalQualityError(message=
            f'Failed to analyze quality trends for tool {request.tool_id}',
            correlation_id=correlation_id)


legacy_router = APIRouter(prefix='/api/v1/signal-quality', tags=[
    'Signal Quality (Legacy)'])


@legacy_router.post('/signals/{signal_id}/quality', response_model=
    SignalQualityResponse)
async def legacy_evaluate_signal_quality(signal_id: str, request:
    QualityEvaluationRequest, request_obj: Request=None, repository:
    ToolEffectivenessRepository=Depends(get_repository), evaluator:
    SignalQualityEvaluator=Depends(get_quality_evaluator)):
    """
    Legacy endpoint for evaluating signal quality.
    Consider migrating to /api/v1/analysis/signal-quality/signals/{signal_id}/evaluate
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/signal-quality/signals/{signal_id}/evaluate'
        )
    return await evaluate_signal_quality(signal_id, request, request_obj,
        repository, evaluator)


@legacy_router.get('/quality-analysis', response_model=QualityAnalysisResponse)
async def legacy_analyze_signal_quality(tool_id: Optional[str]=Query(None,
    description='Filter by specific tool ID'), timeframe: Optional[str]=
    Query(None, description='Filter by specific timeframe'), market_regime:
    Optional[str]=Query(None, description='Filter by market regime'), days:
    Optional[int]=Query(30, description='Number of days to analyze'),
    request_obj: Request=None, repository: ToolEffectivenessRepository=
    Depends(get_repository), analyzer: SignalQualityAnalyzer=Depends(
    get_quality_analyzer)):
    """
    Legacy endpoint for analyzing signal quality.
    Consider migrating to /api/v1/analysis/signal-quality/analyze
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/signal-quality/analyze'
        )
    request = QualityAnalysisRequest(tool_id=tool_id, timeframe=timeframe,
        market_regime=market_regime, days=days)
    return await analyze_signal_quality(request, request_obj, repository,
        analyzer)


@legacy_router.get('/quality-trends', response_model=QualityTrendResponse)
async def legacy_analyze_quality_trends(tool_id: str=Query(..., description
    ='Tool ID to analyze'), window_size: int=Query(20, description=
    'Size of moving window for trend analysis'), days: int=Query(90,
    description='Number of days to analyze'), request_obj: Request=None,
    repository: ToolEffectivenessRepository=Depends(get_repository),
    analyzer: SignalQualityAnalyzer=Depends(get_quality_analyzer)):
    """
    Legacy endpoint for analyzing quality trends.
    Consider migrating to /api/v1/analysis/signal-quality/trends
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/signal-quality/trends'
        )
    request = QualityTrendRequest(tool_id=tool_id, window_size=window_size,
        days=days)
    return await analyze_quality_trends(request, request_obj, repository,
        analyzer)


def setup_signal_quality_routes(app: FastAPI) ->None:
    """
    Set up signal quality routes.

    Args:
        app: FastAPI application
    """
    app.include_router(router, prefix='/api')
    app.include_router(legacy_router)
