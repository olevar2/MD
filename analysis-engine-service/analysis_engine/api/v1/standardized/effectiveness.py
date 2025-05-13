"""
Standardized Tool Effectiveness API for Analysis Engine Service.

This module provides standardized API endpoints for tracking and analyzing the
effectiveness of trading tools, including signal registration, outcome tracking,
and performance metrics.

All endpoints follow the platform's standardized API design patterns.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from analysis_engine.services.tool_effectiveness import ToolEffectivenessTracker, Signal, SignalOutcome, WinRateMetric, ProfitFactorMetric, ExpectedPayoffMetric, ReliabilityByMarketRegimeMetric
from analysis_engine.db.connection import get_db_session
from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository
from analysis_engine.core.exceptions_bridge import ForexTradingPlatformError, AnalysisError, EffectivenessAnalysisError, InsufficientDataError, get_correlation_id_from_request
from analysis_engine.monitoring.structured_logging import get_structured_logger


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class MarketRegimeEnum(str, Enum):
    """Enum for market regime types"""
    TRENDING = 'trending'
    RANGING = 'ranging'
    VOLATILE = 'volatile'
    UNKNOWN = 'unknown'


class SignalRequest(BaseModel):
    """Request model for registering a signal"""
    tool_id: str = Field(..., description=
        'Identifier for the trading tool that generated the signal')
    signal_type: str = Field(..., description=
        'Type of signal (buy, sell, etc.)')
    instrument: str = Field(..., description=
        "Trading instrument (e.g., 'EUR_USD')")
    timestamp: datetime = Field(..., description=
        'When the signal was generated')
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description=
        'Signal confidence level (0.0-1.0)')
    timeframe: str = Field(..., description=
        "Timeframe of the analysis (e.g., '1H', '4H', '1D')")
    market_regime: MarketRegimeEnum = Field(default=MarketRegimeEnum.
        UNKNOWN, description='Market regime at signal time')
    additional_data: Optional[Dict] = Field(default=None, description=
        'Any additional signal metadata')


    class Config:
    """
    Config class.
    
    Attributes:
        Add attributes here
    """

        schema_extra = {'example': {'tool_id': 'macd_crossover_v1',
            'signal_type': 'buy', 'instrument': 'EUR_USD', 'timestamp':
            '2025-04-01T12:00:00Z', 'confidence': 0.85, 'timeframe': '1H',
            'market_regime': 'trending', 'additional_data': {'macd_value': 
            0.0025, 'signal_line': 0.001, 'histogram': 0.0015}}}


class OutcomeRequest(BaseModel):
    """Request model for registering a signal outcome"""
    signal_id: str = Field(..., description=
        'ID of the signal that this outcome is associated with')
    success: bool = Field(..., description=
        'Whether the signal led to a successful trade')
    realized_profit: float = Field(default=0.0, description=
        'Profit/loss realized from the trade')
    timestamp: datetime = Field(..., description=
        'When the outcome was recorded')
    additional_data: Optional[Dict] = Field(default=None, description=
        'Any additional outcome metadata')


    class Config:
    """
    Config class.
    
    Attributes:
        Add attributes here
    """

        schema_extra = {'example': {'signal_id': 'sig_12345', 'success': 
            True, 'realized_profit': 0.75, 'timestamp':
            '2025-04-01T14:30:00Z', 'additional_data': {'exit_reason':
            'take_profit', 'pips': 15, 'trade_duration_minutes': 150}}}


class EffectivenessMetric(BaseModel):
    """Model for an effectiveness metric"""
    name: str
    value: float
    description: str


class ToolEffectiveness(BaseModel):
    """Model for tool effectiveness data"""
    tool_id: str
    metrics: List[EffectivenessMetric]
    signal_count: int
    first_signal_date: Optional[datetime]
    last_signal_date: Optional[datetime]
    success_rate: float


class SignalResponse(BaseModel):
    """Response model for signal registration"""
    signal_id: str
    status: str = 'success'
    message: str = 'Signal registered successfully'


class OutcomeResponse(BaseModel):
    """Response model for outcome registration"""
    status: str = 'success'
    message: str = 'Outcome registered successfully'


class DashboardData(BaseModel):
    """Response model for dashboard data"""
    summary: Dict[str, Any]
    filters: Dict[str, Any]
    top_performing_tools: List[Dict[str, Any]]


class ReportRequest(BaseModel):
    """Request model for creating an effectiveness report"""
    name: str = Field(..., description='Name of the report')
    description: Optional[str] = Field(None, description=
        'Description of the report')
    tool_id: Optional[str] = Field(None, description=
        'Filter by specific tool ID')
    timeframe: Optional[str] = Field(None, description=
        'Filter by specific timeframe')
    instrument: Optional[str] = Field(None, description=
        'Filter by specific instrument')
    market_regime: Optional[MarketRegimeEnum] = Field(None, description=
        'Filter by market regime')
    from_date: Optional[datetime] = Field(None, description=
        'Start date for metrics')
    to_date: Optional[datetime] = Field(None, description=
        'End date for metrics')


class ReportResponse(BaseModel):
    """Response model for report creation"""
    status: str = 'success'
    message: str = 'Report saved successfully'
    report_id: int


class ReportSummary(BaseModel):
    """Summary model for a report"""
    id: int
    name: str
    description: Optional[str]
    filters: Dict[str, Any]
    created_at: datetime


class ReportDetail(BaseModel):
    """Detailed model for a report"""
    id: int
    name: str
    description: Optional[str]
    filters: Dict[str, Any]
    created_at: datetime
    report_data: Dict[str, Any]


tracker = ToolEffectivenessTracker()
router = APIRouter(prefix='/v1/analysis/effectiveness', tags=[
    'Tool Effectiveness'])
logger = get_structured_logger(__name__)


@router.post('/signals', response_model=SignalResponse, status_code=201,
    summary='Register a signal', description=
    'Register a new signal from a trading tool for effectiveness tracking.')
@async_with_exception_handling
async def register_signal(request: SignalRequest, request_obj: Request, db:
    Session=Depends(get_db_session)):
    """
    Register a new signal from a trading tool for effectiveness tracking.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        signal = Signal(tool_id=request.tool_id, signal_type=request.
            signal_type, instrument=request.instrument, timestamp=request.
            timestamp, confidence=request.confidence, timeframe=request.
            timeframe, market_regime=request.market_regime.value,
            additional_data=request.additional_data or {})
        signal_id = tracker.register_signal(signal)
        repository = ToolEffectivenessRepository(db)
        tool = repository.get_tool(request.tool_id)
        if not tool:
            repository.create_tool({'tool_id': request.tool_id, 'name':
                request.tool_id})
        repository.create_signal({'signal_id': signal_id, 'tool_id':
            request.tool_id, 'signal_type': request.signal_type,
            'instrument': request.instrument, 'timestamp': request.
            timestamp, 'confidence': request.confidence, 'timeframe':
            request.timeframe, 'market_regime': request.market_regime.value,
            'additional_data': request.additional_data or {}})
        logger.info(f'Registered signal {signal_id} for tool {request.tool_id}'
            , extra={'correlation_id': correlation_id, 'signal_id':
            signal_id, 'tool_id': request.tool_id, 'instrument': request.
            instrument, 'timeframe': request.timeframe})
        return SignalResponse(signal_id=signal_id)
    except Exception as e:
        logger.error(f'Error registering signal: {str(e)}', extra={
            'correlation_id': correlation_id}, exc_info=True)
        raise EffectivenessAnalysisError(message=
            f'Failed to register signal: {str(e)}', correlation_id=
            correlation_id)


@router.post('/outcomes', response_model=OutcomeResponse, status_code=201,
    summary='Register an outcome', description=
    'Register the outcome of a previously registered signal.')
@async_with_exception_handling
async def register_outcome(request: OutcomeRequest, request_obj: Request,
    db: Session=Depends(get_db_session)):
    """
    Register the outcome of a previously registered signal.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        outcome = SignalOutcome(signal_id=request.signal_id, success=
            request.success, realized_profit=request.realized_profit,
            timestamp=request.timestamp, additional_data=request.
            additional_data or {})
        repository = ToolEffectivenessRepository(db)
        signal = repository.get_signal(request.signal_id)
        if not signal:
            raise InsufficientDataError(message=
                f'Signal not found: {request.signal_id}', correlation_id=
                correlation_id)
        tracker.register_outcome(outcome)
        repository.create_outcome({'signal_id': request.signal_id,
            'success': request.success, 'realized_profit': request.
            realized_profit, 'timestamp': request.timestamp,
            'additional_data': request.additional_data or {}})
        tool_id = signal.tool_id
        now = datetime.utcnow()
        start_date = now - timedelta(days=30)
        signals = repository.get_signals(tool_id=tool_id, from_date=
            start_date, to_date=now, limit=1000)
        signal_ids = [s.signal_id for s in signals]
        outcomes = []
        for sig_id in signal_ids:
            outcomes.extend(repository.get_outcomes_for_signal(sig_id))
        if signals and outcomes:
            service_signals = [Signal(id=str(s.signal_id), tool_id=s.
                tool_id, signal_type=s.signal_type, instrument=s.instrument,
                timestamp=s.timestamp, confidence=s.confidence, timeframe=s
                .timeframe, market_regime=s.market_regime, additional_data=
                s.additional_data) for s in signals]
            service_outcomes = [SignalOutcome(signal_id=str(o.signal_id),
                success=o.success, realized_profit=o.realized_profit,
                timestamp=o.timestamp, additional_data=o.additional_data) for
                o in outcomes]
            win_rate = WinRateMetric().calculate(service_signals,
                service_outcomes)
            profit_factor = ProfitFactorMetric().calculate(service_signals,
                service_outcomes)
            expected_payoff = ExpectedPayoffMetric().calculate(service_signals,
                service_outcomes)
            repository.save_effectiveness_metric({'tool_id': tool_id,
                'metric_type': 'win_rate', 'value': win_rate, 'start_date':
                start_date, 'end_date': now, 'signal_count': len(signals),
                'outcome_count': len(outcomes)})
            repository.save_effectiveness_metric({'tool_id': tool_id,
                'metric_type': 'profit_factor', 'value': profit_factor,
                'start_date': start_date, 'end_date': now, 'signal_count':
                len(signals), 'outcome_count': len(outcomes)})
            repository.save_effectiveness_metric({'tool_id': tool_id,
                'metric_type': 'expected_payoff', 'value': expected_payoff,
                'start_date': start_date, 'end_date': now, 'signal_count':
                len(signals), 'outcome_count': len(outcomes)})
        logger.info(f'Registered outcome for signal {request.signal_id}',
            extra={'correlation_id': correlation_id, 'signal_id': request.
            signal_id, 'success': request.success, 'realized_profit':
            request.realized_profit})
        return OutcomeResponse()
    except InsufficientDataError:
        raise
    except KeyError as e:
        logger.error(f'Signal not found: {str(e)}', extra={'correlation_id':
            correlation_id}, exc_info=True)
        raise InsufficientDataError(message=f'Signal not found: {str(e)}',
            correlation_id=correlation_id)
    except Exception as e:
        logger.error(f'Error registering outcome: {str(e)}', extra={
            'correlation_id': correlation_id}, exc_info=True)
        raise EffectivenessAnalysisError(message=
            f'Failed to register outcome: {str(e)}', correlation_id=
            correlation_id)


@router.get('/metrics', response_model=List[ToolEffectiveness], summary=
    'Get effectiveness metrics', description=
    'Retrieve effectiveness metrics for trading tools with optional filtering.'
    )
@async_with_exception_handling
async def get_effectiveness_metrics(request_obj: Request, tool_id: Optional
    [str]=Query(None, description='Filter by specific tool ID'), timeframe:
    Optional[str]=Query(None, description='Filter by specific timeframe'),
    instrument: Optional[str]=Query(None, description=
    'Filter by specific instrument'), market_regime: Optional[
    MarketRegimeEnum]=Query(None, description='Filter by market regime'),
    from_date: Optional[datetime]=Query(None, description=
    'Start date for metrics calculation'), to_date: Optional[datetime]=
    Query(None, description='End date for metrics calculation'), db:
    Session=Depends(get_db_session)):
    """
    Retrieve effectiveness metrics for trading tools with optional filtering.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        repository = ToolEffectivenessRepository(db)
        if tool_id:
            tools = [repository.get_tool(tool_id)]
            if not tools[0]:
                return []
        else:
            tools = repository.get_tools(limit=100)
        result = []
        for tool in tools:
            metrics_data = repository.get_effectiveness_metrics(tool_id=
                tool.tool_id, timeframe=timeframe, instrument=instrument,
                market_regime=market_regime.value if market_regime else
                None, from_date=from_date, to_date=to_date)
            if not metrics_data:
                continue
            metrics_dict = {}
            for metric in metrics_data:
                metrics_dict[metric.metric_type] = metric
            signals = repository.get_signals(tool_id=tool.tool_id,
                timeframe=timeframe, instrument=instrument, market_regime=
                market_regime.value if market_regime else None, from_date=
                from_date, to_date=to_date, limit=1000)
            if not signals:
                continue
            signal_ids = [s.signal_id for s in signals]
            outcomes = []
            for sig_id in signal_ids:
                outcomes.extend(repository.get_outcomes_for_signal(sig_id))
            success_count = sum(1 for o in outcomes if o.success)
            success_rate = success_count / len(outcomes) if outcomes else 0.0
            timestamps = [s.timestamp for s in signals]
            first_signal_date = min(timestamps) if timestamps else None
            last_signal_date = max(timestamps) if timestamps else None
            formatted_metrics = []
            if 'win_rate' in metrics_dict:
                formatted_metrics.append(EffectivenessMetric(name=
                    'Win Rate', value=metrics_dict['win_rate'].value,
                    description='Percentage of successful signals'))
            if 'profit_factor' in metrics_dict:
                formatted_metrics.append(EffectivenessMetric(name=
                    'Profit Factor', value=metrics_dict['profit_factor'].
                    value, description=
                    'Ratio of gross profits to gross losses'))
            if 'expected_payoff' in metrics_dict:
                formatted_metrics.append(EffectivenessMetric(name=
                    'Expected Payoff', value=metrics_dict['expected_payoff'
                    ].value, description='Average profit/loss per signal'))
            for metric in metrics_data:
                if metric.metric_type.startswith('reliability_'):
                    regime = metric.metric_type.split('_')[1]
                    formatted_metrics.append(EffectivenessMetric(name=
                        f'Reliability in {regime.capitalize()} Market',
                        value=metric.value, description=
                        f'Success rate in {regime} market conditions'))
            tool_effectiveness = ToolEffectiveness(tool_id=tool.tool_id,
                metrics=formatted_metrics, signal_count=len(signals),
                first_signal_date=first_signal_date, last_signal_date=
                last_signal_date, success_rate=success_rate)
            result.append(tool_effectiveness)
        logger.info(f'Retrieved effectiveness metrics for {len(result)} tools',
            extra={'correlation_id': correlation_id, 'tool_id': tool_id,
            'timeframe': timeframe, 'instrument': instrument,
            'market_regime': market_regime.value if market_regime else None})
        return result
    except Exception as e:
        logger.error(f'Error retrieving effectiveness metrics: {str(e)}',
            extra={'correlation_id': correlation_id}, exc_info=True)
        raise EffectivenessAnalysisError(message=
            f'Failed to retrieve effectiveness metrics: {str(e)}',
            correlation_id=correlation_id)


@router.get('/dashboard-data', response_model=DashboardData, summary=
    'Get dashboard data', description=
    'Get aggregated data suitable for dashboard visualization.')
@async_with_exception_handling
async def get_dashboard_data(request_obj: Request, db: Session=Depends(
    get_db_session)):
    """
    Get aggregated data suitable for dashboard visualization.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        repository = ToolEffectivenessRepository(db)
        tools = repository.get_tools(limit=100)
        tool_ids = [tool.tool_id for tool in tools]
        all_signals = []
        for tool_id in tool_ids:
            all_signals.extend(repository.get_signals(tool_id=tool_id,
                limit=1000))
        if not all_signals:
            return DashboardData(summary={'total_signals': 0,
                'total_outcomes': 0, 'overall_success_rate': 0}, filters={
                'tools': tool_ids, 'timeframes': [], 'instruments': [],
                'regimes': []}, top_performing_tools=[])
        timeframes = list(set(s.timeframe for s in all_signals))
        instruments = list(set(s.instrument for s in all_signals))
        regimes = list(set(s.market_regime for s in all_signals))
        tool_performance = []
        for tool in tools:
            latest_metrics = repository.get_latest_tool_metrics(tool.tool_id)
            win_rate = latest_metrics.get('win_rate', 0)
            signals = repository.get_signals(tool_id=tool.tool_id, limit=1000)
            signal_count = len(signals)
            outcome_count = 0
            success_count = 0
            for signal in signals:
                outcomes = repository.get_outcomes_for_signal(signal.signal_id)
                outcome_count += len(outcomes)
                success_count += sum(1 for o in outcomes if o.success)
            success_rate = (success_count / outcome_count * 100 if 
                outcome_count > 0 else 0)
            tool_performance.append({'tool_id': tool.tool_id, 'name': tool.
                name, 'signals_count': signal_count, 'outcomes_count':
                outcome_count, 'success_rate': success_rate, 'win_rate':
                win_rate})
        top_tools = sorted(tool_performance, key=lambda x: x['success_rate'
            ], reverse=True)[:5]
        total_signals = len(all_signals)
        total_outcomes = sum(t['outcomes_count'] for t in tool_performance)
        overall_success_rate = sum(t['success_rate'] * t['outcomes_count'] for
            t in tool_performance
            ) / total_outcomes if total_outcomes > 0 else 0
        logger.info('Retrieved dashboard data', extra={'correlation_id':
            correlation_id, 'total_signals': total_signals,
            'total_outcomes': total_outcomes, 'tool_count': len(tools)})
        return DashboardData(summary={'total_signals': total_signals,
            'total_outcomes': total_outcomes, 'overall_success_rate':
            overall_success_rate}, filters={'tools': tool_ids, 'timeframes':
            timeframes, 'instruments': instruments, 'regimes': regimes},
            top_performing_tools=top_tools)
    except Exception as e:
        logger.error(f'Error retrieving dashboard data: {str(e)}', extra={
            'correlation_id': correlation_id}, exc_info=True)
        raise EffectivenessAnalysisError(message=
            f'Failed to retrieve dashboard data: {str(e)}', correlation_id=
            correlation_id)


@router.post('/reports', response_model=ReportResponse, status_code=201,
    summary='Save effectiveness report', description=
    'Save a new effectiveness report.')
@async_with_exception_handling
async def save_effectiveness_report(request: ReportRequest, request_obj:
    Request, db: Session=Depends(get_db_session)):
    """
    Save a new effectiveness report.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        repository = ToolEffectivenessRepository(db)
        metrics_result = await get_effectiveness_metrics(request_obj=
            request_obj, tool_id=request.tool_id, timeframe=request.
            timeframe, instrument=request.instrument, market_regime=request
            .market_regime, from_date=request.from_date, to_date=request.
            to_date, db=db)
        report_data = {'metrics': [metric.dict() for metric in
            metrics_result], 'generated_at': datetime.utcnow().isoformat(),
            'summary': {'tool_count': len(metrics_result), 'total_signals':
            sum(m.signal_count for m in metrics_result), 'avg_success_rate':
            sum(m.success_rate for m in metrics_result) / len(
            metrics_result) if metrics_result else 0}}
        filters = {'tool_id': request.tool_id, 'timeframe': request.
            timeframe, 'instrument': request.instrument, 'market_regime': 
            request.market_regime.value if request.market_regime else None,
            'from_date': request.from_date.isoformat() if request.from_date
             else None, 'to_date': request.to_date.isoformat() if request.
            to_date else None}
        report = repository.save_report({'name': request.name,
            'description': request.description, 'report_data': report_data,
            'filters': filters, 'created_at': datetime.utcnow()})
        logger.info(f'Saved effectiveness report: {request.name}', extra={
            'correlation_id': correlation_id, 'report_id': report.id,
            'tool_id': request.tool_id, 'timeframe': request.timeframe,
            'instrument': request.instrument})
        return ReportResponse(report_id=report.id)
    except Exception as e:
        logger.error(f'Error saving effectiveness report: {str(e)}', extra=
            {'correlation_id': correlation_id}, exc_info=True)
        raise EffectivenessAnalysisError(message=
            f'Failed to save effectiveness report: {str(e)}',
            correlation_id=correlation_id)


@router.get('/reports', response_model=List[ReportSummary], summary=
    'Get effectiveness reports', description=
    'Get all saved effectiveness reports.')
@async_with_exception_handling
async def get_effectiveness_reports(request_obj: Request, skip: int=Query(0,
    description='Skip items for pagination'), limit: int=Query(100,
    description='Limit items for pagination'), db: Session=Depends(
    get_db_session)):
    """
    Get all saved effectiveness reports.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        repository = ToolEffectivenessRepository(db)
        reports = repository.get_reports(skip=skip, limit=limit)
        logger.info(f'Retrieved {len(reports)} effectiveness reports',
            extra={'correlation_id': correlation_id, 'skip': skip, 'limit':
            limit})
        return [ReportSummary(id=report.id, name=report.name, description=
            report.description, filters=report.filters, created_at=report.
            created_at) for report in reports]
    except Exception as e:
        logger.error(f'Error retrieving effectiveness reports: {str(e)}',
            extra={'correlation_id': correlation_id}, exc_info=True)
        raise EffectivenessAnalysisError(message=
            f'Failed to retrieve effectiveness reports: {str(e)}',
            correlation_id=correlation_id)


@router.get('/reports/{report_id}', response_model=ReportDetail, summary=
    'Get effectiveness report', description=
    'Get a specific effectiveness report by ID.')
@async_with_exception_handling
async def get_effectiveness_report(report_id: int=Path(..., description=
    'ID of the report to retrieve'), request_obj: Request=None, db: Session
    =Depends(get_db_session)):
    """
    Get a specific effectiveness report by ID.
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        repository = ToolEffectivenessRepository(db)
        report = repository.get_report(report_id)
        if not report:
            raise InsufficientDataError(message=
                f'Report not found: {report_id}', correlation_id=correlation_id
                )
        logger.info(f'Retrieved effectiveness report {report_id}', extra={
            'correlation_id': correlation_id, 'report_id': report_id,
            'report_name': report.name})
        return ReportDetail(id=report.id, name=report.name, description=
            report.description, filters=report.filters, created_at=report.
            created_at, report_data=report.report_data)
    except InsufficientDataError:
        raise
    except Exception as e:
        logger.error(
            f'Error retrieving effectiveness report {report_id}: {str(e)}',
            extra={'correlation_id': correlation_id}, exc_info=True)
        raise EffectivenessAnalysisError(message=
            f'Failed to retrieve effectiveness report: {str(e)}',
            correlation_id=correlation_id)


@router.delete('/tools/{tool_id}/data', status_code=200, summary=
    'Clear tool data', description=
    'Clear all data for a specific tool (for testing or resetting purposes).')
@async_with_exception_handling
async def clear_tool_data(tool_id: str=Path(..., description=
    'ID of the tool to clear data for'), request_obj: Request=None, db:
    Session=Depends(get_db_session)):
    """
    Clear all data for a specific tool (for testing or resetting purposes).
    """
    correlation_id = get_correlation_id_from_request(request_obj)
    try:
        original_signal_count = len(tracker.signals)
        tracker.signals = [s for s in tracker.signals if s.tool_id != tool_id]
        remaining_signal_ids = {s.id for s in tracker.signals}
        original_outcome_count = len(tracker.outcomes)
        tracker.outcomes = [o for o in tracker.outcomes if o.signal_id in
            remaining_signal_ids]
        signals_removed = original_signal_count - len(tracker.signals)
        outcomes_removed = original_outcome_count - len(tracker.outcomes)
        repository = ToolEffectivenessRepository(db)
        db_signals_removed, db_outcomes_removed = repository.delete_tool_data(
            tool_id)
        logger.info(f'Cleared data for tool {tool_id}', extra={
            'correlation_id': correlation_id, 'tool_id': tool_id,
            'signals_removed': signals_removed + db_signals_removed,
            'outcomes_removed': outcomes_removed + db_outcomes_removed})
        return {'status': 'success', 'signals_removed': signals_removed +
            db_signals_removed, 'outcomes_removed': outcomes_removed +
            db_outcomes_removed, 'message':
            f'Removed all data for tool {tool_id}'}
    except Exception as e:
        logger.error(f'Error clearing tool data for {tool_id}: {str(e)}',
            extra={'correlation_id': correlation_id}, exc_info=True)
        raise EffectivenessAnalysisError(message=
            f'Failed to clear tool data: {str(e)}', correlation_id=
            correlation_id)


legacy_router = APIRouter(prefix='/api/v1/tool-effectiveness', tags=[
    'Tool Effectiveness (Legacy)'])


@legacy_router.post('/signals/', status_code=201, response_model=Dict[str, str]
    )
async def legacy_register_signal(signal_data: SignalRequest, request_obj:
    Request=None, db: Session=Depends(get_db_session)):
    """
    Legacy endpoint for registering a signal.
    Consider migrating to /api/v1/analysis/effectiveness/signals
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/effectiveness/signals'
        )
    response = await register_signal(signal_data, request_obj, db)
    return {'signal_id': response.signal_id}


@legacy_router.post('/outcomes/', status_code=201)
async def legacy_register_outcome(outcome_data: OutcomeRequest, request_obj:
    Request=None, db: Session=Depends(get_db_session)):
    """
    Legacy endpoint for registering an outcome.
    Consider migrating to /api/v1/analysis/effectiveness/outcomes
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/effectiveness/outcomes'
        )
    await register_outcome(outcome_data, request_obj, db)
    return {'status': 'success', 'message': 'Outcome registered successfully'}


@legacy_router.get('/metrics/', response_model=List[ToolEffectiveness])
async def legacy_get_effectiveness_metrics(tool_id: Optional[str]=Query(
    None, description='Filter by specific tool ID'), timeframe: Optional[
    str]=Query(None, description='Filter by specific timeframe'),
    instrument: Optional[str]=Query(None, description=
    'Filter by specific instrument'), market_regime: Optional[
    MarketRegimeEnum]=Query(None, description='Filter by market regime'),
    from_date: Optional[datetime]=Query(None, description=
    'Start date for metrics calculation'), to_date: Optional[datetime]=
    Query(None, description='End date for metrics calculation'),
    request_obj: Request=None, db: Session=Depends(get_db_session)):
    """
    Legacy endpoint for retrieving effectiveness metrics.
    Consider migrating to /api/v1/analysis/effectiveness/metrics
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/effectiveness/metrics'
        )
    return await get_effectiveness_metrics(request_obj, tool_id, timeframe,
        instrument, market_regime, from_date, to_date, db)


@legacy_router.get('/dashboard-data/', response_model=Dict)
async def legacy_get_dashboard_data(request_obj: Request=None, db: Session=
    Depends(get_db_session)):
    """
    Legacy endpoint for retrieving dashboard data.
    Consider migrating to /api/v1/analysis/effectiveness/dashboard-data
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/effectiveness/dashboard-data'
        )
    dashboard_data = await get_dashboard_data(request_obj, db)
    return dashboard_data.dict()


@legacy_router.post('/reports/', status_code=201, response_model=Dict)
async def legacy_save_effectiveness_report(name: str=Query(..., description
    ='Name of the report'), description: Optional[str]=Query(None,
    description='Description of the report'), tool_id: Optional[str]=Query(
    None, description='Filter by specific tool ID'), timeframe: Optional[
    str]=Query(None, description='Filter by specific timeframe'),
    instrument: Optional[str]=Query(None, description=
    'Filter by specific instrument'), market_regime: Optional[
    MarketRegimeEnum]=Query(None, description='Filter by market regime'),
    from_date: Optional[datetime]=Query(None, description=
    'Start date for metrics'), to_date: Optional[datetime]=Query(None,
    description='End date for metrics'), request_obj: Request=None, db:
    Session=Depends(get_db_session)):
    """
    Legacy endpoint for saving an effectiveness report.
    Consider migrating to /api/v1/analysis/effectiveness/reports
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/effectiveness/reports'
        )
    request = ReportRequest(name=name, description=description, tool_id=
        tool_id, timeframe=timeframe, instrument=instrument, market_regime=
        market_regime, from_date=from_date, to_date=to_date)
    response = await save_effectiveness_report(request, request_obj, db)
    return {'status': 'success', 'message': 'Report saved successfully',
        'report_id': response.report_id}


@legacy_router.get('/reports/', response_model=List[Dict])
async def legacy_get_effectiveness_reports(skip: int=Query(0, description=
    'Skip items for pagination'), limit: int=Query(100, description=
    'Limit items for pagination'), request_obj: Request=None, db: Session=
    Depends(get_db_session)):
    """
    Legacy endpoint for retrieving effectiveness reports.
    Consider migrating to /api/v1/analysis/effectiveness/reports
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/effectiveness/reports'
        )
    reports = await get_effectiveness_reports(request_obj, skip, limit, db)
    return [report.dict() for report in reports]


@legacy_router.get('/reports/{report_id}', response_model=Dict)
async def legacy_get_effectiveness_report(report_id: int, request_obj:
    Request=None, db: Session=Depends(get_db_session)):
    """
    Legacy endpoint for retrieving a specific effectiveness report.
    Consider migrating to /api/v1/analysis/effectiveness/reports/{report_id}
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/effectiveness/reports/{report_id}'
        )
    report = await get_effectiveness_report(report_id, request_obj, db)
    return report.dict()


@legacy_router.delete('/tool/{tool_id}/data/', status_code=200)
async def legacy_clear_tool_data(tool_id: str, request_obj: Request=None,
    db: Session=Depends(get_db_session)):
    """
    Legacy endpoint for clearing tool data.
    Consider migrating to /api/v1/analysis/effectiveness/tools/{tool_id}/data
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /api/v1/analysis/effectiveness/tools/{tool_id}/data'
        )
    return await clear_tool_data(tool_id, request_obj, db)


def setup_effectiveness_routes(app: FastAPI) ->None:
    """
    Set up effectiveness routes.

    Args:
        app: FastAPI application
    """
    app.include_router(router, prefix='/api')
    app.include_router(legacy_router)
