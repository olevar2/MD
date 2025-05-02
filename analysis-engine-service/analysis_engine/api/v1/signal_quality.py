"""
Signal Quality Evaluation API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

# Import services
from analysis_engine.services.signal_quality_evaluator import (
    SignalQualityEvaluator,
    SignalQualityAnalyzer
)
from analysis_engine.services.tool_effectiveness import SignalEvent, MarketRegime

# Import database dependencies
from analysis_engine.db.connection import get_db_session
from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository

router = APIRouter()

# API models
class QualityEvaluationRequest(BaseModel):
    signal_id: str = Field(..., description="ID of the signal to evaluate")
    market_context: Optional[Dict[str, Any]] = Field(default={}, description="Additional market context")
    historical_data: Optional[Dict[str, Any]] = Field(default={}, description="Historical performance data")

class SignalQualityResponse(BaseModel):
    signal_id: str
    tool_id: str
    base_quality: float
    timing_quality: float
    confluence: Optional[float]
    historical_reliability: Optional[float]
    regime_compatibility: Optional[float]
    overall_quality: float
    evaluation_timestamp: datetime

class QualityAnalysisResponse(BaseModel):
    average_quality: float
    average_success_rate: float
    correlation: float
    sample_size: int
    quality_brackets: List[Dict[str, Any]]
    tool_id: Optional[str]
    timeframe: Optional[str]
    market_regime: Optional[str]

class QualityTrendResponse(BaseModel):
    quality_trend: float
    success_trend: float
    data_points: int
    window_size: int
    moving_averages: List[Dict[str, Any]]
    tool_id: Optional[str]

# Initialize services
quality_evaluator = SignalQualityEvaluator()
quality_analyzer = SignalQualityAnalyzer()

@router.post("/signals/{signal_id}/quality", response_model=SignalQualityResponse)
def evaluate_signal_quality(
    signal_id: str,
    request: QualityEvaluationRequest,
    db: Session = Depends(get_db_session)
):
    """
    Evaluate the quality of a specific trading signal and store the results
    """
    repository = ToolEffectivenessRepository(db)
    
    # Get the signal from the database
    signal = repository.get_signal(signal_id)
    if not signal:
        raise HTTPException(status_code=404, detail=f"Signal not found: {signal_id}")
    
    # Convert to SignalEvent for evaluation
    signal_event = SignalEvent(
        tool_name=signal.tool_id,
        signal_type=signal.signal_type,
        direction=signal.signal_type,  # Assuming signal_type contains direction info
        strength=signal.confidence,
        timestamp=signal.timestamp,
        symbol=signal.instrument,
        timeframe=signal.timeframe,
        price_at_signal=0.0,  # This would need to be populated from the database
        metadata=signal.additional_data or {},
        market_context=request.market_context or {}
    )
    
    # Get similar signals for confluence analysis (optional)
    additional_signals = []
    if "evaluate_confluence" in request.market_context and request.market_context["evaluate_confluence"]:
        # Get signals within a time window (e.g., 1 hour)
        window_start = signal.timestamp - timedelta(hours=1)
        window_end = signal.timestamp + timedelta(hours=1)
        
        other_signals = repository.get_signals(
            instrument=signal.instrument,
            from_date=window_start,
            to_date=window_end,
            limit=50
        )
        
        # Convert to SignalEvent objects
        for s in other_signals:
            if str(s.signal_id) != signal_id:  # Skip the current signal
                additional_signals.append(SignalEvent(
                    tool_name=s.tool_id,
                    signal_type=s.signal_type,
                    direction=s.signal_type,  # Assuming signal_type contains direction
                    strength=s.confidence,
                    timestamp=s.timestamp,
                    symbol=s.instrument,
                    timeframe=s.timeframe,
                    price_at_signal=0.0,
                    metadata=s.additional_data or {},
                    market_context=s.additional_data.get('market_context', {}) if s.additional_data else {}
                ))
    
    # Evaluate signal quality
    quality_metrics = quality_evaluator.evaluate_signal_quality(
        signal=signal_event,
        market_context=request.market_context,
        additional_signals=additional_signals if additional_signals else None,
        historical_performance=request.historical_data
    )
    
    # Update the signal in the database with quality metrics
    updated_additional_data = signal.additional_data or {}
    updated_additional_data['quality'] = quality_metrics
    updated_additional_data['quality_evaluated_at'] = datetime.utcnow().isoformat()
    
    repository.update_tool(signal.tool_id, {'additional_data': updated_additional_data})
    
    # Prepare response
    response = SignalQualityResponse(
        signal_id=signal_id,
        tool_id=signal.tool_id,
        base_quality=quality_metrics['base_quality'],
        timing_quality=quality_metrics['timing_quality'],
        confluence=quality_metrics.get('confluence'),
        historical_reliability=quality_metrics.get('historical_reliability'),
        regime_compatibility=quality_metrics.get('regime_compatibility'),
        overall_quality=quality_metrics['overall_quality'],
        evaluation_timestamp=datetime.utcnow()
    )
    
    return response

@router.get("/quality-analysis", response_model=QualityAnalysisResponse)
def analyze_signal_quality(
    tool_id: Optional[str] = Query(None, description="Filter by specific tool ID"),
    timeframe: Optional[str] = Query(None, description="Filter by specific timeframe"),
    market_regime: Optional[str] = Query(None, description="Filter by market regime"),
    days: Optional[int] = Query(30, description="Number of days to analyze"),
    db: Session = Depends(get_db_session)
):
    """
    Analyze the relationship between signal quality and outcomes
    """
    repository = ToolEffectivenessRepository(db)
    
    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # Get signals with the specified filters
    signals = repository.get_signals(
        tool_id=tool_id,
        timeframe=timeframe,
        market_regime=market_regime,
        from_date=start_date,
        to_date=end_date,
        limit=1000
    )
    
    if not signals:
        raise HTTPException(status_code=404, detail="No signals found matching the criteria")
    
    # Get outcomes for these signals
    signal_ids = [str(s.signal_id) for s in signals]
    signal_outcomes = []
    
    for signal_id in signal_ids:
        signal = repository.get_signal(signal_id)
        outcomes = repository.get_outcomes_for_signal(signal_id)
        
        for outcome in outcomes:
            # Create SignalEvent
            signal_event = SignalEvent(
                tool_name=signal.tool_id,
                signal_type=signal.signal_type,
                direction=signal.signal_type,  # Assuming signal_type contains direction
                strength=signal.confidence,
                timestamp=signal.timestamp,
                symbol=signal.instrument,
                timeframe=signal.timeframe,
                price_at_signal=0.0,
                metadata=signal.additional_data or {},
                market_context={}
            )
            
            # Create SignalOutcome
            signal_outcome = SignalOutcome(
                signal_event=signal_event,
                outcome="success" if outcome.success else "failure",
                exit_price=None,  # Not available in our current schema
                exit_timestamp=outcome.timestamp,
                profit_loss=outcome.realized_profit
            )
            
            signal_outcomes.append(signal_outcome)
    
    if not signal_outcomes:
        raise HTTPException(status_code=404, detail="No outcomes found for the signals")
    
    # Analyze quality vs outcomes
    analysis_result = quality_analyzer.analyze_quality_vs_outcomes(signal_outcomes)
    
    if "error" in analysis_result:
        raise HTTPException(status_code=400, detail=analysis_result["error"])
    
    # Prepare response
    response = QualityAnalysisResponse(
        average_quality=analysis_result["average_quality"],
        average_success_rate=analysis_result["average_success_rate"],
        correlation=analysis_result["correlation"],
        sample_size=analysis_result["sample_size"],
        quality_brackets=analysis_result["quality_brackets"],
        tool_id=tool_id,
        timeframe=timeframe,
        market_regime=market_regime
    )
    
    return response

@router.get("/quality-trends", response_model=QualityTrendResponse)
def analyze_quality_trends(
    tool_id: str = Query(..., description="Tool ID to analyze"),
    window_size: int = Query(20, description="Size of moving window for trend analysis"),
    days: int = Query(90, description="Number of days to analyze"),
    db: Session = Depends(get_db_session)
):
    """
    Analyze trends in signal quality over time for a specific tool
    """
    repository = ToolEffectivenessRepository(db)
    
    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # Get signals for this tool
    signals = repository.get_signals(
        tool_id=tool_id,
        from_date=start_date,
        to_date=end_date,
        limit=1000
    )
    
    if not signals or len(signals) < window_size:
        raise HTTPException(
            status_code=404, 
            detail=f"Insufficient signals found. Need at least {window_size}, found {len(signals) if signals else 0}"
        )
    
    # Get outcomes for these signals
    signal_ids = [str(s.signal_id) for s in signals]
    signal_outcomes = []
    
    for signal_id in signal_ids:
        signal = repository.get_signal(signal_id)
        outcomes = repository.get_outcomes_for_signal(signal_id)
        
        for outcome in outcomes:
            # Create SignalEvent
            signal_event = SignalEvent(
                tool_name=signal.tool_id,
                signal_type=signal.signal_type,
                direction=signal.signal_type,
                strength=signal.confidence,
                timestamp=signal.timestamp,
                symbol=signal.instrument,
                timeframe=signal.timeframe,
                price_at_signal=0.0,
                metadata=signal.additional_data or {},
                market_context={}
            )
            
            # Create SignalOutcome
            signal_outcome = SignalOutcome(
                signal_event=signal_event,
                outcome="success" if outcome.success else "failure",
                exit_price=None,
                exit_timestamp=outcome.timestamp,
                profit_loss=outcome.realized_profit
            )
            
            signal_outcomes.append(signal_outcome)
    
    # Analyze quality trends
    trend_result = quality_analyzer.analyze_quality_trends(signal_outcomes, window_size)
    
    if "error" in trend_result:
        raise HTTPException(status_code=400, detail=trend_result["error"])
    
    # Prepare response
    response = QualityTrendResponse(
        quality_trend=trend_result["quality_trend"],
        success_trend=trend_result["success_trend"],
        data_points=trend_result["data_points"],
        window_size=trend_result["window_size"],
        moving_averages=trend_result["moving_averages"],
        tool_id=tool_id
    )
    
    return response
