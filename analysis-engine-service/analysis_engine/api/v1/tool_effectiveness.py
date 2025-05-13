"""
Tool effectiveness module.

This module provides functionality for...
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, List, Optional
import datetime
from enum import Enum
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from analysis_engine.services.tool_effectiveness import ToolEffectivenessTracker, Signal, SignalOutcome, WinRateMetric, ProfitFactorMetric, ExpectedPayoffMetric, ReliabilityByMarketRegimeMetric
from analysis_engine.db.connection import get_db_session
from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository
router = APIRouter()


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class MarketRegimeEnum(str, Enum):
    """
    MarketRegimeEnum class that inherits from str, Enum.
    
    Attributes:
        Add attributes here
    """

    TRENDING = 'trending'
    RANGING = 'ranging'
    VOLATILE = 'volatile'
    UNKNOWN = 'unknown'


class SignalRequest(BaseModel):
    """
    SignalRequest class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    tool_id: str = Field(..., description=
        'Identifier for the trading tool that generated the signal')
    signal_type: str = Field(..., description=
        'Type of signal (buy, sell, etc.)')
    instrument: str = Field(..., description=
        "Trading instrument (e.g., 'EUR_USD')")
    timestamp: datetime.datetime = Field(..., description=
        'When the signal was generated')
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description=
        'Signal confidence level (0.0-1.0)')
    timeframe: str = Field(..., description=
        "Timeframe of the analysis (e.g., '1H', '4H', '1D')")
    market_regime: MarketRegimeEnum = Field(default=MarketRegimeEnum.
        UNKNOWN, description='Market regime at signal time')
    additional_data: Optional[Dict] = Field(default=None, description=
        'Any additional signal metadata')


class OutcomeRequest(BaseModel):
    """
    OutcomeRequest class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    signal_id: str = Field(..., description=
        'ID of the signal that this outcome is associated with')
    success: bool = Field(..., description=
        'Whether the signal led to a successful trade')
    realized_profit: float = Field(default=0.0, description=
        'Profit/loss realized from the trade')
    timestamp: datetime.datetime = Field(..., description=
        'When the outcome was recorded')
    additional_data: Optional[Dict] = Field(default=None, description=
        'Any additional outcome metadata')


class EffectivenessMetric(BaseModel):
    """
    EffectivenessMetric class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    name: str
    value: float
    description: str


class ToolEffectiveness(BaseModel):
    """
    ToolEffectiveness class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    tool_id: str
    metrics: List[EffectivenessMetric]
    signal_count: int
    first_signal_date: Optional[datetime.datetime]
    last_signal_date: Optional[datetime.datetime]
    success_rate: float


tracker = ToolEffectivenessTracker()


@router.post('/signals/', status_code=201, response_model=Dict[str, str])
def register_signal(signal_data: SignalRequest, db: Session=Depends(
    get_db_session)):
    """

    Register a new signal from a trading tool for effectiveness tracking

    """
    signal = Signal(tool_id=signal_data.tool_id, signal_type=signal_data.
        signal_type, instrument=signal_data.instrument, timestamp=
        signal_data.timestamp, confidence=signal_data.confidence, timeframe
        =signal_data.timeframe, market_regime=signal_data.market_regime.
        value, additional_data=signal_data.additional_data or {})
    signal_id = tracker.register_signal(signal)
    repository = ToolEffectivenessRepository(db)
    tool = repository.get_tool(signal_data.tool_id)
    if not tool:
        repository.create_tool({'tool_id': signal_data.tool_id, 'name':
            signal_data.tool_id})
    repository.create_signal({'signal_id': signal_id, 'tool_id':
        signal_data.tool_id, 'signal_type': signal_data.signal_type,
        'instrument': signal_data.instrument, 'timestamp': signal_data.
        timestamp, 'confidence': signal_data.confidence, 'timeframe':
        signal_data.timeframe, 'market_regime': signal_data.market_regime.
        value, 'additional_data': signal_data.additional_data or {}})
    return {'signal_id': signal_id}


@router.post('/outcomes/', status_code=201)
@with_exception_handling
def register_outcome(outcome_data: OutcomeRequest, db: Session=Depends(
    get_db_session)):
    """

    Register the outcome of a previously registered signal

    """
    outcome = SignalOutcome(signal_id=outcome_data.signal_id, success=
        outcome_data.success, realized_profit=outcome_data.realized_profit,
        timestamp=outcome_data.timestamp, additional_data=outcome_data.
        additional_data or {})
    repository = ToolEffectivenessRepository(db)
    signal = repository.get_signal(outcome_data.signal_id)
    if not signal:
        raise HTTPException(status_code=404, detail=
            f'Signal not found: {outcome_data.signal_id}')
    try:
        tracker.register_outcome(outcome)
        repository.create_outcome({'signal_id': outcome_data.signal_id,
            'success': outcome_data.success, 'realized_profit':
            outcome_data.realized_profit, 'timestamp': outcome_data.
            timestamp, 'additional_data': outcome_data.additional_data or {}})
        tool_id = signal.tool_id
        now = datetime.datetime.utcnow()
        start_date = now - datetime.timedelta(days=30)
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
        return {'status': 'success', 'message':
            'Outcome registered successfully'}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=
            f'Signal not found: {str(e)}')
    except Exception as e:
        raise HTTPException(status_code=400, detail=
            f'Failed to register outcome: {str(e)}')


@router.get('/metrics/', response_model=List[ToolEffectiveness])
def get_effectiveness_metrics(tool_id: Optional[str]=Query(None,
    description='Filter by specific tool ID'), timeframe: Optional[str]=
    Query(None, description='Filter by specific timeframe'), instrument:
    Optional[str]=Query(None, description='Filter by specific instrument'),
    market_regime: Optional[MarketRegimeEnum]=Query(None, description=
    'Filter by market regime'), from_date: Optional[datetime.datetime]=
    Query(None, description='Start date for metrics calculation'), to_date:
    Optional[datetime.datetime]=Query(None, description=
    'End date for metrics calculation'), db: Session=Depends(get_db_session)):
    """

    Retrieve effectiveness metrics for trading tools with optional filtering

    """
    repository = ToolEffectivenessRepository(db)
    if tool_id:
        tools = [repository.get_tool(tool_id)]
        if not tools[0]:
            return []
    else:
        tools = repository.get_tools(limit=100)
    result = []
    for tool in tools:
        metrics_data = repository.get_effectiveness_metrics(tool_id=tool.
            tool_id, timeframe=timeframe, instrument=instrument,
            market_regime=market_regime.value if market_regime else None,
            from_date=from_date, to_date=to_date)
        if not metrics_data:
            continue
        metrics_dict = {}
        for metric in metrics_data:
            metrics_dict[metric.metric_type] = metric
        signals = repository.get_signals(tool_id=tool.tool_id, timeframe=
            timeframe, instrument=instrument, market_regime=market_regime.
            value if market_regime else None, from_date=from_date, to_date=
            to_date, limit=1000)
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
            formatted_metrics.append(EffectivenessMetric(name='Win Rate',
                value=metrics_dict['win_rate'].value, description=
                'Percentage of successful signals'))
        if 'profit_factor' in metrics_dict:
            formatted_metrics.append(EffectivenessMetric(name=
                'Profit Factor', value=metrics_dict['profit_factor'].value,
                description='Ratio of gross profits to gross losses'))
        if 'expected_payoff' in metrics_dict:
            formatted_metrics.append(EffectivenessMetric(name=
                'Expected Payoff', value=metrics_dict['expected_payoff'].
                value, description='Average profit/loss per signal'))
        for metric in metrics_data:
            if metric.metric_type.startswith('reliability_'):
                regime = metric.metric_type.split('_')[1]
                formatted_metrics.append(EffectivenessMetric(name=
                    f'Reliability in {regime.capitalize()} Market', value=
                    metric.value, description=
                    f'Success rate in {regime} market conditions'))
        tool_effectiveness = ToolEffectiveness(tool_id=tool.tool_id,
            metrics=formatted_metrics, signal_count=len(signals),
            first_signal_date=first_signal_date, last_signal_date=
            last_signal_date, success_rate=success_rate)
        result.append(tool_effectiveness)
    return result


@router.delete('/tool/{tool_id}/data/', status_code=200)
def clear_tool_data(tool_id: str, db: Session=Depends(get_db_session)):
    """

    Clear all data for a specific tool (for testing or resetting purposes)

    """
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
    return {'status': 'success', 'signals_removed': signals_removed +
        db_signals_removed, 'outcomes_removed': outcomes_removed +
        db_outcomes_removed, 'message': f'Removed all data for tool {tool_id}'}


@router.get('/dashboard-data/', response_model=Dict)
def get_dashboard_data(db: Session=Depends(get_db_session)):
    """

    Get aggregated data suitable for dashboard visualization

    """
    repository = ToolEffectivenessRepository(db)
    tools = repository.get_tools(limit=100)
    tool_ids = [tool.tool_id for tool in tools]
    all_signals = []
    for tool_id in tool_ids:
        all_signals.extend(repository.get_signals(tool_id=tool_id, limit=1000))
    if not all_signals:
        return {'summary': {'total_signals': 0, 'total_outcomes': 0,
            'overall_success_rate': 0}, 'filters': {'tools': tool_ids,
            'timeframes': [], 'instruments': [], 'regimes': []},
            'top_performing_tools': []}
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
        tool_performance.append({'tool_id': tool.tool_id, 'name': tool.name,
            'signals_count': signal_count, 'outcomes_count': outcome_count,
            'success_rate': success_rate, 'win_rate': win_rate})
    top_tools = sorted(tool_performance, key=lambda x: x['success_rate'],
        reverse=True)[:5]
    total_signals = len(all_signals)
    total_outcomes = sum(t['outcomes_count'] for t in tool_performance)
    overall_success_rate = sum(t['success_rate'] * t['outcomes_count'] for
        t in tool_performance) / total_outcomes if total_outcomes > 0 else 0
    return {'summary': {'total_signals': total_signals, 'total_outcomes':
        total_outcomes, 'overall_success_rate': overall_success_rate},
        'filters': {'tools': tool_ids, 'timeframes': timeframes,
        'instruments': instruments, 'regimes': regimes},
        'top_performing_tools': top_tools}


@router.post('/reports/', status_code=201, response_model=Dict)
def save_effectiveness_report(name: str=Query(..., description=
    'Name of the report'), description: Optional[str]=Query(None,
    description='Description of the report'), tool_id: Optional[str]=Query(
    None, description='Filter by specific tool ID'), timeframe: Optional[
    str]=Query(None, description='Filter by specific timeframe'),
    instrument: Optional[str]=Query(None, description=
    'Filter by specific instrument'), market_regime: Optional[
    MarketRegimeEnum]=Query(None, description='Filter by market regime'),
    from_date: Optional[datetime.datetime]=Query(None, description=
    'Start date for metrics'), to_date: Optional[datetime.datetime]=Query(
    None, description='End date for metrics'), db: Session=Depends(
    get_db_session)):
    """

    Save a new effectiveness report

    """
    repository = ToolEffectivenessRepository(db)
    metrics_result = get_effectiveness_metrics(tool_id=tool_id, timeframe=
        timeframe, instrument=instrument, market_regime=market_regime,
        from_date=from_date, to_date=to_date, db=db)
    report_data = {'metrics': [metric.dict() for metric in metrics_result],
        'generated_at': datetime.datetime.utcnow().isoformat(), 'summary':
        {'tool_count': len(metrics_result), 'total_signals': sum(m.
        signal_count for m in metrics_result), 'avg_success_rate': sum(m.
        success_rate for m in metrics_result) / len(metrics_result) if
        metrics_result else 0}}
    filters = {'tool_id': tool_id, 'timeframe': timeframe, 'instrument':
        instrument, 'market_regime': market_regime.value if market_regime else
        None, 'from_date': from_date.isoformat() if from_date else None,
        'to_date': to_date.isoformat() if to_date else None}
    report = repository.save_report({'name': name, 'description':
        description, 'report_data': report_data, 'filters': filters,
        'created_at': datetime.datetime.utcnow()})
    return {'status': 'success', 'message': 'Report saved successfully',
        'report_id': report.id}


@router.get('/reports/', response_model=List[Dict])
def get_effectiveness_reports(skip: int=Query(0, description=
    'Skip items for pagination'), limit: int=Query(100, description=
    'Limit items for pagination'), db: Session=Depends(get_db_session)):
    """

    Get all saved effectiveness reports

    """
    repository = ToolEffectivenessRepository(db)
    reports = repository.get_reports(skip=skip, limit=limit)
    return [{'id': report.id, 'name': report.name, 'description': report.
        description, 'filters': report.filters, 'created_at': report.
        created_at} for report in reports]


@router.get('/reports/{report_id}', response_model=Dict)
def get_effectiveness_report(report_id: int, db: Session=Depends(
    get_db_session)):
    """

    Get a specific effectiveness report by ID

    """
    repository = ToolEffectivenessRepository(db)
    report = repository.get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail=
            f'Report not found: {report_id}')
    return {'id': report.id, 'name': report.name, 'description': report.
        description, 'filters': report.filters, 'created_at': report.
        created_at, 'report_data': report.report_data}
