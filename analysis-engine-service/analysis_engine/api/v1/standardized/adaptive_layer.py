"""
Standardized Adaptive Layer API for Analysis Engine Service.

This module provides standardized API endpoints for the Adaptive Layer,
allowing interaction with adaptation capabilities, parameter adjustments,
feedback processing, and retrieving historical data and insights.

All endpoints follow the platform's standardized API design patterns.
"""
import logging
import pandas as pd
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from analysis_engine.db.connection import get_db_session
from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository
from analysis_engine.services.market_regime_detector import MarketRegimeService
from analysis_engine.services.adaptive_layer import AdaptiveLayer, AdaptationStrategy
from analysis_engine.services.adaptive_integration import AdaptiveLayerIntegrationService, AdaptiveStrategyOptimizer
from analysis_engine.adaptive_layer.adaptation_engine import AdaptationEngine
from analysis_engine.adaptive_layer.parameter_adjustment_service import ParameterAdjustmentService
from analysis_engine.adaptive_layer.feedback_loop import FeedbackLoop
from analysis_engine.services.tool_effectiveness_service import ToolEffectivenessService
from analysis_engine.tools.market_regime_identifier import MarketRegimeIdentifier
from analysis_engine.adaptive_layer.models import TradeFeedback, FeedbackSource, FeedbackCategory, FeedbackStatus
from analysis_engine.monitoring.structured_logging import get_structured_logger


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class GenerateAdaptiveParamsRequest(BaseModel):
    """Request model for generating adaptive parameters"""
    strategy_id: str
    symbol: str
    timeframe: str
    ohlc_data: List[Dict]
    available_tools: List[str]
    adaptation_strategy: Optional[str] = 'moderate'


    class Config:
    """
    Config class.
    
    Attributes:
        Add attributes here
    """

        schema_extra = {'example': {'strategy_id': 'macd_rsi_strategy_v1',
            'symbol': 'EURUSD', 'timeframe': '1h', 'ohlc_data': [{
            'timestamp': '2025-04-01T00:00:00', 'open': 1.0765, 'high': 
            1.078, 'low': 1.076, 'close': 1.0775, 'volume': 1000}, {
            'timestamp': '2025-04-01T01:00:00', 'open': 1.0775, 'high': 
            1.079, 'low': 1.077, 'close': 1.0785, 'volume': 1200}],
            'available_tools': ['macd', 'rsi', 'bollinger_bands',
            'fibonacci_retracement'], 'adaptation_strategy': 'moderate'}}


class UpdateStrategyParamsRequest(BaseModel):
    """Request model for updating strategy parameters"""
    strategy_id: str
    symbol: str
    timeframe: str
    ohlc_data: List[Dict]
    available_tools: List[str]
    adaptation_strategy: Optional[str] = 'moderate'
    strategy_execution_api_url: Optional[str] = None


class GenerateStrategyRecommendationsRequest(BaseModel):
    """Request model for generating strategy recommendations"""
    strategy_id: str
    symbol: str
    timeframe: str
    ohlc_data: List[Dict]
    current_tools: List[str]
    all_available_tools: List[str]


class AnalyzeStrategyEffectivenessRequest(BaseModel):
    """Request model for analyzing strategy effectiveness"""
    strategy_id: str
    symbol: str
    timeframe: str
    period_days: int = 30
    look_back_periods: int = 6


class ParameterAdjustmentRequest(BaseModel):
    """Request model for parameter adjustment"""
    strategy_id: str = Field(..., description='Identifier for the strategy')
    instrument: str = Field(..., description=
        "Trading instrument (e.g., 'EUR_USD')")
    timeframe: str = Field(..., description=
        "Timeframe for analysis (e.g., '1H', '4H', 'D')")
    current_parameters: Dict[str, Any] = Field(..., description=
        'Current strategy parameters')
    context: Optional[Dict[str, Any]] = Field(default=None, description=
        'Additional context information (e.g., market data, regime)')


class ParameterAdjustmentResponse(BaseModel):
    """Response model for parameter adjustment"""
    strategy_id: str = Field(..., description='Identifier for the strategy')
    instrument: str = Field(..., description='Trading instrument')
    timeframe: str = Field(..., description='Timeframe for analysis')
    original_parameters: Dict[str, Any] = Field(..., description=
        'Original strategy parameters')
    adjusted_parameters: Dict[str, Any] = Field(..., description=
        'Adjusted strategy parameters')
    significant_changes: Dict[str, Any] = Field(default={}, description=
        'Significant parameter changes')
    adaptation_id: Optional[str] = Field(None, description=
        'Unique ID for this adaptation instance')


class StrategyOutcomeRequest(BaseModel):
    """Request model for recording strategy outcomes"""
    strategy_id: str = Field(..., description='Identifier for the strategy')
    instrument: str = Field(..., description=
        "Trading instrument (e.g., 'EUR_USD')")
    timeframe: str = Field(..., description=
        "Timeframe for analysis (e.g., '1H', '4H', 'D')")
    adaptation_id: str = Field(..., description=
        'Identifier for the specific adaptation being evaluated')
    outcome_metrics: Dict[str, Any] = Field(..., description=
        'Performance metrics from strategy execution (e.g., pnl, win_rate)')
    market_regime: Optional[str] = Field(None, description=
        'Market regime during execution (if known)')
    feedback_content: Optional[Dict[str, Any]] = Field(default=None,
        description='Additional qualitative or quantitative feedback')


class AdaptationHistoryResponse(BaseModel):
    """Response model for adaptation history"""
    adaptations: List[Dict[str, Any]] = Field(default=[], description=
        'List of adaptation records')


class ParameterHistoryResponse(BaseModel):
    """Response model for parameter history"""
    history: List[Dict[str, Any]] = Field(default=[], description=
        'List of parameter adjustment records')


class AdaptationInsightResponse(BaseModel):
    """Response model for adaptation insights"""
    insights: List[Dict[str, Any]] = Field(default=[], description=
        'List of insights derived from feedback')


class AdaptationPerformanceResponse(BaseModel):
    """Response model for adaptation performance by regime"""
    performance: Dict[str, Any] = Field(default={}, description=
        'Performance metrics aggregated by market regime')


def get_db_session_dependency():
    """Get database session dependency"""
    return get_db_session()


def get_tool_effectiveness_repository(db: Session=Depends(
    get_db_session_dependency)):
    """Get tool effectiveness repository dependency"""
    return ToolEffectivenessRepository(db)


def get_market_regime_service():
    """Get market regime service dependency"""
    return MarketRegimeService()


def get_market_regime_identifier():
    """Get market regime identifier dependency"""
    return MarketRegimeIdentifier()


def get_adaptation_engine():
    """Get adaptation engine dependency"""
    tool_effectiveness_service = ToolEffectivenessService()
    market_regime_identifier = MarketRegimeIdentifier()
    return AdaptationEngine(tool_effectiveness_service=
        tool_effectiveness_service, market_regime_identifier=
        market_regime_identifier)


def get_adaptive_layer(tool_effectiveness_repository:
    ToolEffectivenessRepository=Depends(get_tool_effectiveness_repository),
    market_regime_service: MarketRegimeService=Depends(
    get_market_regime_service)):
    """Get adaptive layer dependency"""
    adaptation_strategy = AdaptationStrategy.MODERATE
    return AdaptiveLayer(tool_effectiveness_repository=
        tool_effectiveness_repository, market_regime_service=
        market_regime_service, adaptation_strategy=adaptation_strategy)


def get_adaptive_integration_service(repository:
    ToolEffectivenessRepository=Depends(get_tool_effectiveness_repository),
    market_regime_service: MarketRegimeService=Depends(
    get_market_regime_service)):
    """Get adaptive integration service dependency"""
    strategy_execution_api_url = None
    return AdaptiveLayerIntegrationService(repository=repository,
        market_regime_service=market_regime_service,
        strategy_execution_api_url=strategy_execution_api_url)


def get_adaptive_strategy_optimizer(repository: ToolEffectivenessRepository
    =Depends(get_tool_effectiveness_repository), market_regime_service:
    MarketRegimeService=Depends(get_market_regime_service)):
    """Get adaptive strategy optimizer dependency"""
    return AdaptiveStrategyOptimizer(repository=repository,
        market_regime_service=market_regime_service)


def get_parameter_adjustment_service(adaptation_engine: AdaptationEngine=
    Depends(get_adaptation_engine)):
    """Get parameter adjustment service dependency"""
    return ParameterAdjustmentService(adaptation_engine=adaptation_engine)


def get_feedback_loop(adaptation_engine: AdaptationEngine=Depends(
    get_adaptation_engine)):
    """Get feedback loop dependency"""
    return FeedbackLoop(adaptation_engine=adaptation_engine)


router = APIRouter(prefix='/v1/analysis/adaptations', tags=['Adaptive Layer'])
logger = get_structured_logger(__name__)


@router.post('/parameters/generate', response_model=Dict, summary=
    'Generate adaptive parameters', description=
    'Generate adaptive parameters based on current market conditions and tool effectiveness.'
    )
@async_with_exception_handling
async def generate_adaptive_parameters(request:
    GenerateAdaptiveParamsRequest, adaptive_layer: AdaptiveLayer=Depends(
    get_adaptive_layer)):
    """
    Generate adaptive parameters based on current market conditions and tool effectiveness.
    """
    try:
        df = pd.DataFrame(request.ohlc_data)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        try:
            adaptation_strategy = AdaptationStrategy(request.
                adaptation_strategy)
        except ValueError:
            logger.warning(
                f"Invalid adaptation strategy '{request.adaptation_strategy}' in request, using default."
                )
            adaptation_strategy = adaptive_layer.adaptation_strategy
        adaptive_params = await adaptive_layer.generate_adaptive_parameters(
            symbol=request.symbol, timeframe=request.timeframe, price_data=
            df, available_tools=request.available_tools)
        result = {'strategy_id': request.strategy_id, 'adaptation_strategy':
            str(adaptation_strategy), 'parameters': adaptive_params,
            'timestamp': datetime.now().isoformat()}
        return result
    except Exception as e:
        logger.error(f'Error generating adaptive parameters: {str(e)}',
            exc_info=True)
        raise HTTPException(status_code=500, detail=
            f'Failed to generate adaptive parameters: {str(e)}')


@router.post('/parameters/adjust', response_model=
    ParameterAdjustmentResponse, summary='Adjust strategy parameters',
    description=
    'Adjust strategy parameters based on market conditions and tool effectiveness.'
    )
@async_with_exception_handling
async def adjust_parameters(request: ParameterAdjustmentRequest, service:
    ParameterAdjustmentService=Depends(get_parameter_adjustment_service)):
    """
    Adjust strategy parameters based on market conditions and tool effectiveness.
    """
    try:
        adjusted_parameters, adaptation_id = await service.adjust_parameters(
            strategy_id=request.strategy_id, instrument=request.instrument,
            timeframe=request.timeframe, current_parameters=request.
            current_parameters, context=request.context)
        significant_changes = {}
        for param_name, new_value in adjusted_parameters.items():
            if param_name not in request.current_parameters:
                continue
            old_value = request.current_parameters[param_name]
            if not isinstance(new_value, (int, float)) or not isinstance(
                old_value, (int, float)):
                continue
            if old_value == 0:
                if new_value != 0:
                    significant_changes[param_name
                        ] = f'{old_value} → {new_value}'
            else:
                pct_change = abs(new_value - old_value) / abs(old_value)
                if pct_change > 0.15:
                    significant_changes[param_name
                        ] = f'{old_value} → {new_value} ({pct_change:.1%} change)'
        return ParameterAdjustmentResponse(strategy_id=request.strategy_id,
            instrument=request.instrument, timeframe=request.timeframe,
            original_parameters=request.current_parameters,
            adjusted_parameters=adjusted_parameters, significant_changes=
            significant_changes, adaptation_id=adaptation_id)
    except Exception as e:
        logger.error(f'Parameter adjustment failed: {str(e)}', exc_info=True)
        raise HTTPException(status_code=500, detail=
            f'Parameter adjustment failed: {str(e)}')


@router.post('/strategy/update', response_model=Dict, summary=
    'Update strategy parameters', description=
    'Generate adaptive parameters and apply them to the strategy execution engine.'
    )
@async_with_exception_handling
async def update_strategy_parameters(request: UpdateStrategyParamsRequest,
    integration_service: AdaptiveLayerIntegrationService=Depends(
    get_adaptive_integration_service)):
    """
    Generate adaptive parameters and apply them to the strategy execution engine.
    """
    try:
        df = pd.DataFrame(request.ohlc_data)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        result = await integration_service.update_strategy_parameters(
            strategy_id=request.strategy_id, symbol=request.symbol,
            timeframe=request.timeframe, price_data=df, available_tools=
            request.available_tools)
        return result
    except Exception as e:
        logger.error(f'Error updating strategy parameters: {str(e)}',
            exc_info=True)
        raise HTTPException(status_code=500, detail=
            f'Failed to update strategy parameters: {str(e)}')


@router.post('/strategy/recommendations', response_model=Dict, summary=
    'Generate strategy recommendations', description=
    'Generate recommendations for optimizing a strategy based on effectiveness data.'
    )
@async_with_exception_handling
async def generate_strategy_recommendations(request:
    GenerateStrategyRecommendationsRequest, optimizer:
    AdaptiveStrategyOptimizer=Depends(get_adaptive_strategy_optimizer)):
    """
    Generate recommendations for optimizing a strategy based on effectiveness data.
    """
    try:
        df = pd.DataFrame(request.ohlc_data)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        result = await optimizer.generate_strategy_recommendations(strategy_id
            =request.strategy_id, symbol=request.symbol, timeframe=request.
            timeframe, price_data=df, current_tools=request.current_tools,
            all_available_tools=request.all_available_tools)
        return result
    except Exception as e:
        logger.error(f'Error generating strategy recommendations: {str(e)}',
            exc_info=True)
        raise HTTPException(status_code=500, detail=
            f'Failed to generate strategy recommendations: {str(e)}')


@router.post('/strategy/effectiveness-trend', response_model=Dict, summary=
    'Analyze strategy effectiveness trend', description=
    "Analyze how a strategy's effectiveness has changed over time.")
@async_with_exception_handling
async def analyze_strategy_effectiveness_trend(request:
    AnalyzeStrategyEffectivenessRequest, optimizer:
    AdaptiveStrategyOptimizer=Depends(get_adaptive_strategy_optimizer)):
    """
    Analyze how a strategy's effectiveness has changed over time.
    """
    try:
        result = await optimizer.analyze_strategy_effectiveness_trend(
            strategy_id=request.strategy_id, symbol=request.symbol,
            timeframe=request.timeframe, period_days=request.period_days,
            look_back_periods=request.look_back_periods)
        return result
    except Exception as e:
        logger.error(f'Error analyzing strategy effectiveness trend: {str(e)}',
            exc_info=True)
        raise HTTPException(status_code=500, detail=
            f'Failed to analyze strategy effectiveness trend: {str(e)}')


@router.post('/feedback/outcomes', status_code=202, summary=
    'Record strategy outcome', description=
    'Record the outcome of a strategy execution with adapted parameters.')
@async_with_exception_handling
async def record_strategy_outcome(request: StrategyOutcomeRequest,
    feedback_loop: FeedbackLoop=Depends(get_feedback_loop)):
    """
    Record the outcome of a strategy execution with adapted parameters.
    """
    try:
        feedback = TradeFeedback(id=str(uuid.uuid4()), strategy_id=request.
            strategy_id, instrument=request.instrument, timeframe=request.
            timeframe, source=FeedbackSource.TRADING_OUTCOME, category=
            FeedbackCategory.UNCATEGORIZED, status=FeedbackStatus.RECEIVED,
            outcome_metrics=request.outcome_metrics, metadata={
            'adaptation_id': request.adaptation_id, 'market_regime':
            request.market_regime, **request.feedback_content or {}},
            timestamp=datetime.utcnow().isoformat())
        await feedback_loop.process_incoming_feedback(feedback)
        return {'status': 'accepted', 'message':
            'Strategy outcome feedback received for processing',
            'feedback_id': feedback.id}
    except Exception as e:
        logger.error(f'Failed to record strategy outcome: {str(e)}',
            exc_info=True)
        raise HTTPException(status_code=500, detail=
            f'Failed to record strategy outcome: {str(e)}')


@router.get('/adaptations/history', response_model=
    AdaptationHistoryResponse, summary='Get adaptation history',
    description=
    'Get the history of adaptation decisions from the adaptation engine.')
@async_with_exception_handling
async def get_adaptation_history(adaptation_engine: AdaptationEngine=
    Depends(get_adaptation_engine)):
    """
    Get the history of adaptation decisions from the adaptation engine.
    """
    try:
        history = await adaptation_engine.get_adaptation_history()
        return AdaptationHistoryResponse(adaptations=history)
    except Exception as e:
        logger.error(f'Failed to retrieve adaptation history: {str(e)}',
            exc_info=True)
        raise HTTPException(status_code=500, detail=
            f'Failed to retrieve adaptation history: {str(e)}')


@router.get('/parameters/history/{strategy-id}/{instrument}/{timeframe}',
    response_model=ParameterHistoryResponse, summary=
    'Get parameter history', description=
    'Get parameter adjustment history for a specific strategy, instrument, and timeframe.'
    )
@async_with_exception_handling
async def get_parameter_history(strategy_id: str=Path(..., description=
    'Identifier for the strategy', alias='strategy-id'), instrument: str=
    Path(..., description='Trading instrument'), timeframe: str=Path(...,
    description='Timeframe for analysis'), limit: int=Query(10, description
    ='Maximum number of history entries to return'), service:
    ParameterAdjustmentService=Depends(get_parameter_adjustment_service)):
    """
    Get parameter adjustment history for a specific strategy, instrument, and timeframe.
    """
    try:
        history = await service.get_parameter_history(strategy_id=
            strategy_id, instrument=instrument, timeframe=timeframe, limit=
            limit)
        return ParameterHistoryResponse(history=history)
    except Exception as e:
        logger.error(f'Failed to retrieve parameter history: {str(e)}',
            exc_info=True)
        raise HTTPException(status_code=500, detail=
            f'Failed to retrieve parameter history: {str(e)}')


@router.get('/feedback/insights/{strategy-id}', response_model=
    AdaptationInsightResponse, summary='Get adaptation insights',
    description='Generate insights from feedback data for a specific strategy.'
    )
@async_with_exception_handling
async def get_adaptation_insights(strategy_id: str=Path(..., description=
    'Identifier for the strategy', alias='strategy-id'), feedback_loop:
    FeedbackLoop=Depends(get_feedback_loop)):
    """
    Generate insights from feedback data for a specific strategy.
    """
    try:
        insights = await feedback_loop.generate_insights(strategy_id=
            strategy_id)
        return AdaptationInsightResponse(insights=insights)
    except Exception as e:
        logger.error(f'Failed to generate insights: {str(e)}', exc_info=True)
        raise HTTPException(status_code=500, detail=
            f'Failed to generate insights: {str(e)}')


@router.get('/feedback/performance/{strategy-id}', response_model=
    AdaptationPerformanceResponse, summary='Get performance by regime',
    description=
    'Get aggregated performance metrics by market regime for a specific strategy.'
    )
@async_with_exception_handling
async def get_performance_by_regime(strategy_id: str=Path(..., description=
    'Identifier for the strategy', alias='strategy-id'), market_regime:
    Optional[str]=Query(None, description=
    'Optional filter for a specific market regime'), feedback_loop:
    FeedbackLoop=Depends(get_feedback_loop)):
    """
    Get aggregated performance metrics by market regime for a specific strategy.
    """
    try:
        performance = await feedback_loop.get_performance_by_regime(strategy_id
            =strategy_id, market_regime=market_regime)
        return AdaptationPerformanceResponse(performance=performance)
    except Exception as e:
        logger.error(f'Failed to retrieve performance metrics: {str(e)}',
            exc_info=True)
        raise HTTPException(status_code=500, detail=
            f'Failed to retrieve performance metrics: {str(e)}')


legacy_router = APIRouter(prefix='/api/v1/adaptive-layer', tags=[
    'Adaptive Layer (Legacy)'])


@legacy_router.post('/parameters/generate', response_model=Dict)
async def legacy_generate_adaptive_parameters(request:
    GenerateAdaptiveParamsRequest, adaptive_layer: AdaptiveLayer=Depends(
    get_adaptive_layer)):
    """
    Legacy endpoint for generating adaptive parameters.
    Consider migrating to /v1/analysis/adaptations/parameters/generate
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /v1/analysis/adaptations/parameters/generate'
        )
    return await generate_adaptive_parameters(request, adaptive_layer)


@legacy_router.post('/parameters/adjust', response_model=
    ParameterAdjustmentResponse)
async def legacy_adjust_parameters(request: ParameterAdjustmentRequest,
    service: ParameterAdjustmentService=Depends(
    get_parameter_adjustment_service)):
    """
    Legacy endpoint for adjusting parameters.
    Consider migrating to /v1/analysis/adaptations/parameters/adjust
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /v1/analysis/adaptations/parameters/adjust'
        )
    return await adjust_parameters(request, service)


@legacy_router.post('/strategy/update', response_model=Dict)
async def legacy_update_strategy_parameters(request:
    UpdateStrategyParamsRequest, integration_service:
    AdaptiveLayerIntegrationService=Depends(get_adaptive_integration_service)):
    """
    Legacy endpoint for updating strategy parameters.
    Consider migrating to /v1/analysis/adaptations/strategy/update
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /v1/analysis/adaptations/strategy/update'
        )
    return await update_strategy_parameters(request, integration_service)


@legacy_router.post('/strategy/recommendations', response_model=Dict)
async def legacy_generate_strategy_recommendations(request:
    GenerateStrategyRecommendationsRequest, optimizer:
    AdaptiveStrategyOptimizer=Depends(get_adaptive_strategy_optimizer)):
    """
    Legacy endpoint for generating strategy recommendations.
    Consider migrating to /v1/analysis/adaptations/strategy/recommendations
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /v1/analysis/adaptations/strategy/recommendations'
        )
    return await generate_strategy_recommendations(request, optimizer)


@legacy_router.post('/strategy/effectiveness-trend', response_model=Dict)
async def legacy_analyze_strategy_effectiveness_trend(request:
    AnalyzeStrategyEffectivenessRequest, optimizer:
    AdaptiveStrategyOptimizer=Depends(get_adaptive_strategy_optimizer)):
    """
    Legacy endpoint for analyzing strategy effectiveness trend.
    Consider migrating to /v1/analysis/adaptations/strategy/effectiveness-trend
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /v1/analysis/adaptations/strategy/effectiveness-trend'
        )
    return await analyze_strategy_effectiveness_trend(request, optimizer)


@legacy_router.post('/feedback/outcomes', status_code=202)
async def legacy_record_strategy_outcome(request: StrategyOutcomeRequest,
    feedback_loop: FeedbackLoop=Depends(get_feedback_loop)):
    """
    Legacy endpoint for recording strategy outcomes.
    Consider migrating to /v1/analysis/adaptations/feedback/outcomes
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /v1/analysis/adaptations/feedback/outcomes'
        )
    return await record_strategy_outcome(request, feedback_loop)


@legacy_router.get('/adaptations/history', response_model=
    AdaptationHistoryResponse)
async def legacy_get_adaptation_history(adaptation_engine: AdaptationEngine
    =Depends(get_adaptation_engine)):
    """
    Legacy endpoint for getting adaptation history.
    Consider migrating to /v1/analysis/adaptations/adaptations/history
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /v1/analysis/adaptations/adaptations/history'
        )
    return await get_adaptation_history(adaptation_engine)


@legacy_router.get('/parameters/history/{strategy_id}/{instrument}/{timeframe}'
    , response_model=ParameterHistoryResponse)
async def legacy_get_parameter_history(strategy_id: str=Path(...,
    description='Identifier for the strategy'), instrument: str=Path(...,
    description='Trading instrument'), timeframe: str=Path(..., description
    ='Timeframe for analysis'), limit: int=Query(10, description=
    'Maximum number of history entries to return'), service:
    ParameterAdjustmentService=Depends(get_parameter_adjustment_service)):
    """
    Legacy endpoint for getting parameter history.
    Consider migrating to /v1/analysis/adaptations/parameters/history/{strategy-id}/{instrument}/{timeframe}
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /v1/analysis/adaptations/parameters/history/{strategy-id}/{instrument}/{timeframe}'
        )
    return await get_parameter_history(strategy_id, instrument, timeframe,
        limit, service)


@legacy_router.get('/feedback/insights/{strategy_id}', response_model=
    AdaptationInsightResponse)
async def legacy_get_adaptation_insights(strategy_id: str=Path(...,
    description='Identifier for the strategy'), feedback_loop: FeedbackLoop
    =Depends(get_feedback_loop)):
    """
    Legacy endpoint for getting adaptation insights.
    Consider migrating to /v1/analysis/adaptations/feedback/insights/{strategy-id}
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /v1/analysis/adaptations/feedback/insights/{strategy-id}'
        )
    return await get_adaptation_insights(strategy_id, feedback_loop)


@legacy_router.get('/feedback/performance/{strategy_id}', response_model=
    AdaptationPerformanceResponse)
async def legacy_get_performance_by_regime(strategy_id: str=Path(...,
    description='Identifier for the strategy'), market_regime: Optional[str
    ]=Query(None, description=
    'Optional filter for a specific market regime'), feedback_loop:
    FeedbackLoop=Depends(get_feedback_loop)):
    """
    Legacy endpoint for getting performance by regime.
    Consider migrating to /v1/analysis/adaptations/feedback/performance/{strategy-id}
    """
    logger.info(
        'Legacy endpoint called - consider migrating to /v1/analysis/adaptations/feedback/performance/{strategy-id}'
        )
    return await get_performance_by_regime(strategy_id, market_regime,
        feedback_loop)


def setup_adaptive_layer_routes(app: FastAPI) ->None:
    """
    Set up adaptive layer routes.

    Args:
        app: FastAPI application
    """
    app.include_router(router, prefix='/api')
    app.include_router(legacy_router)
