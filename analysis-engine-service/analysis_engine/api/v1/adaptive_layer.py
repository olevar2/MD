"""
Adaptive Layer API endpoints (v1)

This module provides consolidated API endpoints for the Adaptive Layer,
allowing interaction with adaptation capabilities, parameter adjustments,
feedback processing, and retrieving historical data and insights.
It merges endpoints from previous API route files.
"""

import logging
import pandas as pd
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
from analysis_engine.services.tool_effectiveness_service import ToolEffectivenessService # Assuming this exists
from analysis_engine.tools.market_regime_identifier import MarketRegimeIdentifier # Assuming this exists


# --- API Request/Response Models (Consolidated) ---

class GenerateAdaptiveParamsRequest(BaseModel):
    """Request model for generating adaptive parameters"""
    strategy_id: str
    symbol: str
    timeframe: str
    ohlc_data: List[Dict]
    available_tools: List[str]
    adaptation_strategy: Optional[str] = "moderate"

    class Config:
        schema_extra = {
            "example": {
                "strategy_id": "macd_rsi_strategy_v1",
                "symbol": "EURUSD",
                "timeframe": "1h",
                "ohlc_data": [
                    {"timestamp": "2025-04-01T00:00:00", "open": 1.0765, "high": 1.0780, "low": 1.0760, "close": 1.0775, "volume": 1000},
                    {"timestamp": "2025-04-01T01:00:00", "open": 1.0775, "high": 1.0790, "low": 1.0770, "close": 1.0785, "volume": 1200},
                ],
                "available_tools": ["macd", "rsi", "bollinger_bands", "fibonacci_retracement"],
                "adaptation_strategy": "moderate"
            }
        }

class UpdateStrategyParamsRequest(BaseModel):
    """Request model for updating strategy parameters"""
    strategy_id: str
    symbol: str
    timeframe: str
    ohlc_data: List[Dict]
    available_tools: List[str]
    adaptation_strategy: Optional[str] = "moderate"
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

# Models from analysis_engine/api/routes/adaptive_layer.py
class ParameterAdjustmentRequest(BaseModel):
    """Request model for parameter adjustment"""
    strategy_id: str = Field(..., description="Identifier for the strategy")
    instrument: str = Field(..., description="Trading instrument (e.g., 'EUR_USD')")
    timeframe: str = Field(..., description="Timeframe for analysis (e.g., '1H', '4H', 'D')")
    current_parameters: Dict[str, Any] = Field(..., description="Current strategy parameters")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context information (e.g., market data, regime)")

class ParameterAdjustmentResponse(BaseModel):
    """Response model for parameter adjustment"""
    strategy_id: str = Field(..., description="Identifier for the strategy")
    instrument: str = Field(..., description="Trading instrument")
    timeframe: str = Field(..., description="Timeframe for analysis")
    original_parameters: Dict[str, Any] = Field(..., description="Original strategy parameters")
    adjusted_parameters: Dict[str, Any] = Field(..., description="Adjusted strategy parameters")
    significant_changes: Dict[str, Any] = Field(default={}, description="Significant parameter changes")
    adaptation_id: Optional[str] = Field(None, description="Unique ID for this adaptation instance") # Added

class StrategyOutcomeRequest(BaseModel):
    """Request model for recording strategy outcomes"""
    strategy_id: str = Field(..., description="Identifier for the strategy")
    instrument: str = Field(..., description="Trading instrument (e.g., 'EUR_USD')")
    timeframe: str = Field(..., description="Timeframe for analysis (e.g., '1H', '4H', 'D')")
    adaptation_id: str = Field(..., description="Identifier for the specific adaptation being evaluated")
    outcome_metrics: Dict[str, Any] = Field(..., description="Performance metrics from strategy execution (e.g., pnl, win_rate)")
    market_regime: Optional[str] = Field(None, description="Market regime during execution (if known)")
    feedback_content: Optional[Dict[str, Any]] = Field(default=None, description="Additional qualitative or quantitative feedback") # Added

class AdaptationHistoryResponse(BaseModel):
    """Response model for adaptation history"""
    adaptations: List[Dict[str, Any]] = Field(default=[], description="List of adaptation records")

class ParameterHistoryResponse(BaseModel):
    """Response model for parameter history"""
    history: List[Dict[str, Any]] = Field(default=[], description="List of parameter adjustment records")

class AdaptationInsightResponse(BaseModel):
    """Response model for adaptation insights"""
    insights: List[Dict[str, Any]] = Field(default=[], description="List of insights derived from feedback")

class AdaptationPerformanceResponse(BaseModel):
    """Response model for adaptation performance by regime"""
    performance: Dict[str, Any] = Field(default={}, description="Performance metrics aggregated by market regime")


# --- Dependency Injection Setup (Consolidated) ---

# These should ideally use a proper DI container (like the one potentially used by factories)
# For now, we keep the simple dependency functions but acknowledge they need improvement.

def get_db_session_dependency():
    # Placeholder: Replace with actual dependency injection for DB session
    return get_db_session() # Assuming get_db_session is available

def get_tool_effectiveness_repository(db: Session = Depends(get_db_session_dependency)):
    return ToolEffectivenessRepository(db)

def get_market_regime_service():
    # Placeholder: Replace with actual dependency injection
    return MarketRegimeService()

def get_market_regime_identifier():
     # Placeholder: Replace with actual dependency injection
    return MarketRegimeIdentifier()

def get_adaptation_engine(
    # tool_effectiveness_service: ToolEffectivenessService = Depends(...), # Add actual dependencies
    # market_regime_identifier: MarketRegimeIdentifier = Depends(get_market_regime_identifier),
    # config_manager: ConfigurationManager = Depends(...)
):
    # Placeholder: Replace with actual dependency injection using a container
    # Example: container.resolve(AdaptationEngine)
    tool_effectiveness_service = ToolEffectivenessService() # Placeholder
    market_regime_identifier = MarketRegimeIdentifier() # Placeholder
    return AdaptationEngine(
        tool_effectiveness_service=tool_effectiveness_service,
        market_regime_identifier=market_regime_identifier
    )

def get_adaptive_layer(
    tool_effectiveness_repository: ToolEffectivenessRepository = Depends(get_tool_effectiveness_repository),
    market_regime_service: MarketRegimeService = Depends(get_market_regime_service),
    # config_manager: ConfigurationManager = Depends(...)
):
     # Placeholder: Replace with actual dependency injection using a container
     # Example: container.resolve(AdaptiveLayer)
     # Determine adaptation strategy from config or request? Defaulting for now.
     adaptation_strategy = AdaptationStrategy.MODERATE
     return AdaptiveLayer(
         tool_effectiveness_repository=tool_effectiveness_repository,
         market_regime_service=market_regime_service,
         adaptation_strategy=adaptation_strategy
     )

def get_adaptive_integration_service(
    repository: ToolEffectivenessRepository = Depends(get_tool_effectiveness_repository),
    market_regime_service: MarketRegimeService = Depends(get_market_regime_service),
    # config_manager: ConfigurationManager = Depends(...)
    # strategy_execution_api_url: str = Depends(...) # Needs config injection
):
    # Placeholder: Replace with actual dependency injection using a container
    # Example: container.resolve(AdaptiveLayerIntegrationService)
    strategy_execution_api_url = None # Needs configuration
    return AdaptiveLayerIntegrationService(
        repository=repository,
        market_regime_service=market_regime_service,
        strategy_execution_api_url=strategy_execution_api_url
    )

def get_adaptive_strategy_optimizer(
    repository: ToolEffectivenessRepository = Depends(get_tool_effectiveness_repository),
    market_regime_service: MarketRegimeService = Depends(get_market_regime_service),
    # config_manager: ConfigurationManager = Depends(...)
):
    # Placeholder: Replace with actual dependency injection using a container
    # Example: container.resolve(AdaptiveStrategyOptimizer)
    return AdaptiveStrategyOptimizer(
        repository=repository,
        market_regime_service=market_regime_service
    )

def get_parameter_adjustment_service(adaptation_engine: AdaptationEngine = Depends(get_adaptation_engine)):
    # Placeholder: Replace with actual dependency injection using a container
    # Example: container.resolve(ParameterAdjustmentService)
    return ParameterAdjustmentService(adaptation_engine=adaptation_engine)

def get_feedback_loop(adaptation_engine: AdaptationEngine = Depends(get_adaptation_engine)):
    # Placeholder: Replace with actual dependency injection using a container
    # Example: container.resolve(FeedbackLoop)
    return FeedbackLoop(adaptation_engine=adaptation_engine)


# --- API Router Setup (Consolidated) ---

router = APIRouter(
    prefix="/api/v1/adaptive-layer", # Consistent prefix
    tags=["Adaptive Layer"], # Unified tag
    responses={404: {"description": "Not found"}},
)

# Setup logging
logger = logging.getLogger(__name__)


# --- API Endpoints (Consolidated) ---

@router.post("/parameters/generate", response_model=Dict) # Renamed from /generate-parameters/
async def generate_adaptive_parameters_endpoint( # Renamed function
    request: GenerateAdaptiveParamsRequest,
    adaptive_layer: AdaptiveLayer = Depends(get_adaptive_layer) # Use consolidated dependency
):
    """
    Generate adaptive parameters based on current market conditions and tool effectiveness.
    (Formerly in analysis_engine.api.v1.adaptive_layer.py)
    """
    try:
        df = pd.DataFrame(request.ohlc_data)
        if 'timestamp' in df.columns:
             df['timestamp'] = pd.to_datetime(df['timestamp'])
             df = df.set_index('timestamp')

        # Allow overriding adaptation strategy via request
        try:
            adaptation_strategy = AdaptationStrategy(request.adaptation_strategy)
            # Update the strategy in the injected AdaptiveLayer instance if possible/needed
            # adaptive_layer.adaptation_strategy = adaptation_strategy
        except ValueError:
            logger.warning(f"Invalid adaptation strategy '{request.adaptation_strategy}' in request, using default.")
            adaptation_strategy = adaptive_layer.adaptation_strategy # Use default from service

        adaptive_params = await adaptive_layer.generate_adaptive_parameters( # Assuming async now
            symbol=request.symbol,
            timeframe=request.timeframe,
            price_data=df,
            available_tools=request.available_tools
        )

        result = {
            "strategy_id": request.strategy_id,
            "adaptation_strategy": str(adaptation_strategy),
            "parameters": adaptive_params,
            "timestamp": datetime.now().isoformat()
        }
        return result
    except Exception as e:
        logger.error(f"Error generating adaptive parameters: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate adaptive parameters: {str(e)}")


@router.post("/parameters/adjust", response_model=ParameterAdjustmentResponse) # Keep path from routes
async def adjust_parameters_endpoint( # Renamed function
    request: ParameterAdjustmentRequest,
    service: ParameterAdjustmentService = Depends(get_parameter_adjustment_service)
):
    """
    Adjust strategy parameters based on market conditions and tool effectiveness.
    (Formerly in analysis_engine.api.routes.adaptive_layer.py)
    """
    try:
        # Assuming adjust_parameters is now async
        adjusted_parameters, adaptation_id = await service.adjust_parameters(
            strategy_id=request.strategy_id,
            instrument=request.instrument,
            timeframe=request.timeframe,
            current_parameters=request.current_parameters,
            context=request.context
        )

        # Find significant changes (logic remains the same)
        significant_changes = {}
        for param_name, new_value in adjusted_parameters.items():
            if param_name not in request.current_parameters:
                continue
            old_value = request.current_parameters[param_name]
            if not isinstance(new_value, (int, float)) or not isinstance(old_value, (int, float)):
                continue
            if old_value == 0:
                if new_value != 0:
                    significant_changes[param_name] = f"{old_value} → {new_value}"
            else:
                pct_change = abs(new_value - old_value) / abs(old_value)
                if pct_change > 0.15: # 15% threshold
                    significant_changes[param_name] = f"{old_value} → {new_value} ({pct_change:.1%} change)"

        return ParameterAdjustmentResponse(
            strategy_id=request.strategy_id,
            instrument=request.instrument,
            timeframe=request.timeframe,
            original_parameters=request.current_parameters,
            adjusted_parameters=adjusted_parameters,
            significant_changes=significant_changes,
            adaptation_id=adaptation_id # Include adaptation ID in response
        )
    except Exception as e:
        logger.error(f"Parameter adjustment failed: {str(e)}", exc_info=True) # Added logging
        raise HTTPException(status_code=500, detail=f"Parameter adjustment failed: {str(e)}")


@router.post("/strategy/update", response_model=Dict) # Renamed from /update-strategy/
async def update_strategy_parameters_endpoint( # Renamed function
    request: UpdateStrategyParamsRequest,
    integration_service: AdaptiveLayerIntegrationService = Depends(get_adaptive_integration_service) # Use consolidated dependency
):
    """
    Generate adaptive parameters and apply them to the strategy execution engine.
    (Formerly in analysis_engine.api.v1.adaptive_layer.py)
    """
    try:
        df = pd.DataFrame(request.ohlc_data)
        if 'timestamp' in df.columns:
             df['timestamp'] = pd.to_datetime(df['timestamp'])
             df = df.set_index('timestamp')

        # Assuming update_strategy_parameters is now async
        result = await integration_service.update_strategy_parameters(
            strategy_id=request.strategy_id,
            symbol=request.symbol,
            timeframe=request.timeframe,
            price_data=df,
            available_tools=request.available_tools,
            # Pass adaptation strategy if the service method accepts it
            # adaptation_strategy=request.adaptation_strategy
        )
        return result
    except Exception as e:
        logger.error(f"Error updating strategy parameters: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update strategy parameters: {str(e)}")


@router.post("/strategy/recommendations", response_model=Dict) # Renamed from /strategy-recommendations/
async def generate_strategy_recommendations_endpoint( # Renamed function
    request: GenerateStrategyRecommendationsRequest,
    optimizer: AdaptiveStrategyOptimizer = Depends(get_adaptive_strategy_optimizer) # Use consolidated dependency
):
    """
    Generate recommendations for optimizing a strategy based on effectiveness data.
    (Formerly in analysis_engine.api.v1.adaptive_layer.py)
    """
    try:
        df = pd.DataFrame(request.ohlc_data)
        if 'timestamp' in df.columns:
             df['timestamp'] = pd.to_datetime(df['timestamp'])
             df = df.set_index('timestamp')

        # Assuming generate_strategy_recommendations is now async
        result = await optimizer.generate_strategy_recommendations(
            strategy_id=request.strategy_id,
            symbol=request.symbol,
            timeframe=request.timeframe,
            price_data=df,
            current_tools=request.current_tools,
            all_available_tools=request.all_available_tools
        )
        return result
    except Exception as e:
        logger.error(f"Error generating strategy recommendations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate strategy recommendations: {str(e)}")


@router.post("/strategy/effectiveness-trend", response_model=Dict) # Renamed from /effectiveness-trend/
async def analyze_strategy_effectiveness_trend_endpoint( # Renamed function
    request: AnalyzeStrategyEffectivenessRequest,
    optimizer: AdaptiveStrategyOptimizer = Depends(get_adaptive_strategy_optimizer) # Use consolidated dependency
):
    """
    Analyze how a strategy's effectiveness has changed over time.
    (Formerly in analysis_engine.api.v1.adaptive_layer.py)
    """
    try:
        # Assuming analyze_strategy_effectiveness_trend is now async
        result = await optimizer.analyze_strategy_effectiveness_trend(
            strategy_id=request.strategy_id,
            symbol=request.symbol,
            timeframe=request.timeframe,
            period_days=request.period_days,
            look_back_periods=request.look_back_periods
        )
        return result
    except Exception as e:
        logger.error(f"Error analyzing strategy effectiveness trend: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to analyze strategy effectiveness trend: {str(e)}")


@router.post("/feedback/outcomes", status_code=202) # Keep path from routes, use 202 Accepted
async def record_strategy_outcome_endpoint( # Renamed function
    request: StrategyOutcomeRequest,
    feedback_loop: FeedbackLoop = Depends(get_feedback_loop)
):
    """
    Record the outcome of a strategy execution with adapted parameters.
    (Formerly in analysis_engine.api.routes.adaptive_layer.py)
    """
    try:
        # Assuming record_strategy_outcome is now async
        # Convert request to TradeFeedback object before passing
        feedback = TradeFeedback(
            id=str(uuid.uuid4()), # Generate a new ID for this feedback instance
            strategy_id=request.strategy_id,
            instrument=request.instrument,
            timeframe=request.timeframe,
            source=FeedbackSource.TRADING_OUTCOME,
            category=FeedbackCategory.UNCATEGORIZED, # Let the loop categorize
            status=FeedbackStatus.RECEIVED,
            outcome_metrics=request.outcome_metrics,
            metadata={
                "adaptation_id": request.adaptation_id,
                "market_regime": request.market_regime,
                **(request.feedback_content or {})
            },
            timestamp=datetime.utcnow().isoformat()
        )

        await feedback_loop.process_incoming_feedback(feedback) # Pass the feedback object

        return {"status": "accepted", "message": "Strategy outcome feedback received for processing", "feedback_id": feedback.id}
    except Exception as e:
        logger.error(f"Failed to record strategy outcome: {str(e)}", exc_info=True) # Added logging
        raise HTTPException(status_code=500, detail=f"Failed to record strategy outcome: {str(e)}")


@router.get("/adaptations/history", response_model=AdaptationHistoryResponse) # Keep path from routes
async def get_adaptation_history_endpoint( # Renamed function
    adaptation_engine: AdaptationEngine = Depends(get_adaptation_engine)
):
    """
    Get the history of adaptation decisions from the adaptation engine.
    (Formerly in analysis_engine.api.routes.adaptive_layer.py)
    """
    try:
        # Assuming get_adaptation_history is now async
        history = await adaptation_engine.get_adaptation_history()
        return AdaptationHistoryResponse(adaptations=history)
    except Exception as e:
        logger.error(f"Failed to retrieve adaptation history: {str(e)}", exc_info=True) # Added logging
        raise HTTPException(status_code=500, detail=f"Failed to retrieve adaptation history: {str(e)}")


@router.get("/parameters/history/{strategy_id}/{instrument}/{timeframe}", response_model=ParameterHistoryResponse) # Keep path from routes
async def get_parameter_history_endpoint( # Renamed function
    strategy_id: str = Path(..., description="Identifier for the strategy"),
    instrument: str = Path(..., description="Trading instrument"),
    timeframe: str = Path(..., description="Timeframe for analysis"),
    limit: int = Query(10, description="Maximum number of history entries to return"),
    service: ParameterAdjustmentService = Depends(get_parameter_adjustment_service)
):
    """
    Get parameter adjustment history for a specific strategy, instrument, and timeframe.
    (Formerly in analysis_engine.api.routes.adaptive_layer.py)
    """
    try:
        # Assuming get_parameter_history is now async
        history = await service.get_parameter_history(
            strategy_id=strategy_id,
            instrument=instrument,
            timeframe=timeframe,
            limit=limit
        )
        return ParameterHistoryResponse(history=history)
    except Exception as e:
        logger.error(f"Failed to retrieve parameter history: {str(e)}", exc_info=True) # Added logging
        raise HTTPException(status_code=500, detail=f"Failed to retrieve parameter history: {str(e)}")


@router.get("/feedback/insights/{strategy_id}", response_model=AdaptationInsightResponse) # Keep path from routes
async def get_adaptation_insights_endpoint( # Renamed function
    strategy_id: str = Path(..., description="Identifier for the strategy"),
    feedback_loop: FeedbackLoop = Depends(get_feedback_loop)
):
    """
    Generate insights from feedback data for a specific strategy.
    (Formerly in analysis_engine.api.routes.adaptive_layer.py)
    """
    try:
        # Assuming generate_insights is now async
        insights = await feedback_loop.generate_insights(strategy_id=strategy_id)
        return AdaptationInsightResponse(insights=insights)
    except Exception as e:
        logger.error(f"Failed to generate insights: {str(e)}", exc_info=True) # Added logging
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")


@router.get("/feedback/performance/{strategy_id}", response_model=AdaptationPerformanceResponse) # Keep path from routes
async def get_performance_by_regime_endpoint( # Renamed function
    strategy_id: str = Path(..., description="Identifier for the strategy"),
    market_regime: Optional[str] = Query(None, description="Optional filter for a specific market regime"),
    feedback_loop: FeedbackLoop = Depends(get_feedback_loop)
):
    """
    Get aggregated performance metrics by market regime for a specific strategy.
    (Formerly in analysis_engine.api.routes.adaptive_layer.py)
    """
    try:
        # Assuming get_performance_by_regime is now async
        performance = await feedback_loop.get_performance_by_regime(
            strategy_id=strategy_id,
            market_regime=market_regime
        )
        return AdaptationPerformanceResponse(performance=performance)
    except Exception as e:
        logger.error(f"Failed to retrieve performance metrics: {str(e)}", exc_info=True) # Added logging
        raise HTTPException(status_code=500, detail=f"Failed to retrieve performance metrics: {str(e)}")
