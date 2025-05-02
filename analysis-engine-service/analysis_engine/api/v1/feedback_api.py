"""
Feedback Management API Endpoints

This module implements the complete set of API endpoints for the feedback system,
including advanced insights and analytics.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query, Body, Path, status, Security

from core_foundations.utils.logger import get_logger
from core_foundations.models.feedback import FeedbackSource, FeedbackCategory, FeedbackStatus
from core_foundations.api.pagination import PaginatedResponse, PaginationParams

from analysis_engine.adaptive_layer.feedback_loop import FeedbackLoop
from analysis_engine.adaptive_layer.parameter_tracking_service import ParameterTrackingService
from analysis_engine.adaptive_layer.strategy_mutation_service import StrategyMutationService
from analysis_engine.adaptive_layer.trading_feedback_collector import TradingFeedbackCollector
from analysis_engine.adaptive_layer.feedback_loop_connector import FeedbackLoopConnector
from analysis_engine.api.dependencies import get_feedback_loop, get_parameter_tracking, get_strategy_mutation
from analysis_engine.api.dependencies import get_trading_feedback_collector, get_feedback_loop_connector

from pydantic import BaseModel, Field

logger = get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1/feedback",
    tags=["feedback"],
    responses={404: {"description": "Not found"}}
)


# ===== Request/Response Models =====

class FeedbackSubmissionModel(BaseModel):
    """Model for submitting feedback"""
    source: FeedbackSource = Field(..., description="Source of the feedback")
    category: FeedbackCategory = Field(..., description="Category of the feedback")
    strategy_id: Optional[str] = Field(None, description="Strategy identifier")
    model_id: Optional[str] = Field(None, description="Model identifier")
    instrument: Optional[str] = Field(None, description="Trading instrument")
    timeframe: Optional[str] = Field(None, description="Timeframe")
    description: Optional[str] = Field(None, description="Description of the feedback")
    outcome_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class StrategyOutcomeModel(BaseModel):
    """Model for strategy execution outcome"""
    strategy_id: str = Field(..., description="Strategy identifier")
    version_id: Optional[str] = Field(None, description="Strategy version")
    instrument: str = Field(..., description="Trading instrument")
    timeframe: str = Field(..., description="Chart timeframe")
    adaptation_id: Optional[str] = Field(None, description="Adaptation identifier (if any)")
    outcome_metrics: Dict[str, Any] = Field(..., description="Outcome metrics")
    market_regime: Optional[str] = Field(None, description="Market regime during execution")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ParameterPerformanceModel(BaseModel):
    """Model for parameter performance details"""
    parameter_name: str = Field(..., description="Parameter name")
    strategy_id: str = Field(..., description="Strategy identifier")
    statistical_significance: bool = Field(..., description="Whether changes are statistically significant")
    confidence: float = Field(..., description="Confidence level (0.0-1.0)")
    sample_size: int = Field(..., description="Number of samples")
    effectiveness_score: float = Field(..., description="Effectiveness score (0.0-1.0)")
    p_value: Optional[float] = Field(None, description="P-value from statistical test")
    effect_size: Optional[float] = Field(None, description="Effect size")


class StrategyVersionModel(BaseModel):
    """Model for strategy version details"""
    version_id: str = Field(..., description="Version identifier")
    parent_id: Optional[str] = Field(None, description="Parent version ID")
    generation: int = Field(..., description="Generation number")
    active: bool = Field(..., description="Whether this is active")
    creation_timestamp: str = Field(..., description="Creation timestamp")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class FeedbackSystemStatusModel(BaseModel):
    """Model for feedback system status response"""
    status: str = Field(..., description="Overall system status")
    components: Dict[str, Dict[str, Any]] = Field(..., description="Component statuses")
    statistics: Dict[str, Any] = Field(..., description="System statistics")
    last_updated: str = Field(..., description="Status timestamp")


class FeedbackLearningModel(BaseModel):
    """Model for feedback learning insights"""
    strategy_id: str = Field(..., description="Strategy identifier")
    insights: List[Dict[str, Any]] = Field(..., description="Learning insights")
    total_feedback_count: int = Field(..., description="Total feedback count")
    influential_parameters: List[Dict[str, Any]] = Field(..., description="Most influential parameters")
    success_factors: List[Dict[str, Any]] = Field(..., description="Success factors")
    risk_factors: List[Dict[str, Any]] = Field(..., description="Risk factors")
    timestamp: str = Field(..., description="Analysis timestamp")


# ===== API Endpoints =====

@router.post("/submit", status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    submission: FeedbackSubmissionModel,
    feedback_collector: TradingFeedbackCollector = Depends(get_trading_feedback_collector)
) -> Dict[str, Any]:
    """
    Submit feedback to the system for processing
    """
    try:
        # Create feedback object and submit
        feedback = submission.dict()
        feedback_id = await feedback_collector.collect_feedback(feedback)
        
        return {
            "feedback_id": feedback_id,
            "status": "submitted"
        }
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit feedback: {str(e)}"
        )


@router.post("/strategy/outcome")
async def record_strategy_outcome(
    outcome: StrategyOutcomeModel,
    feedback_connector: FeedbackLoopConnector = Depends(get_feedback_loop_connector)
) -> Dict[str, Any]:
    """
    Record the outcome of a strategy execution
    """
    try:
        # Create feedback data
        feedback_data = {
            "feedback_id": "", # Will be generated
            "strategy_id": outcome.strategy_id,
            "instrument": outcome.instrument,
            "timeframe": outcome.timeframe,
            "source": FeedbackSource.STRATEGY_EXECUTION.value,
            "category": FeedbackCategory.PERFORMANCE_METRICS.value,
            "outcome_metrics": outcome.outcome_metrics,
            "metadata": {
                "market_regime": outcome.market_regime,
                "adaptation_id": outcome.adaptation_id,
                "version_id": outcome.version_id,
                **outcome.metadata
            },
            "timestamp": datetime.utcnow().isoformat()
        }

        # Process through the feedback loop connector
        feedback_id = await feedback_connector.process_execution_feedback(feedback_data)
        
        return {
            "feedback_id": feedback_id,
            "status": "recorded"
        }
    except Exception as e:
        logger.error(f"Error recording strategy outcome: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record strategy outcome: {str(e)}"
        )


@router.get("/statistics")
async def get_feedback_statistics(
    strategy_id: Optional[str] = None,
    model_id: Optional[str] = None,
    instrument: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    feedback_collector: TradingFeedbackCollector = Depends(get_trading_feedback_collector)
) -> Dict[str, Any]:
    """
    Get statistics about collected feedback
    """
    try:
        stats = await feedback_collector.get_statistics(
            strategy_id=strategy_id,
            model_id=model_id,
            instrument=instrument,
            start_time=start_time,
            end_time=end_time
        )
        
        return stats
    except Exception as e:
        logger.error(f"Error retrieving feedback statistics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve feedback statistics: {str(e)}"
        )


@router.get("/items", response_model=PaginatedResponse)
async def get_feedback_items(
    source: Optional[FeedbackSource] = None,
    category: Optional[FeedbackCategory] = None,
    strategy_id: Optional[str] = None,
    model_id: Optional[str] = None,
    instrument: Optional[str] = None,
    status: Optional[FeedbackStatus] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    pagination: PaginationParams = Depends(),
    feedback_collector: TradingFeedbackCollector = Depends(get_trading_feedback_collector)
) -> Dict[str, Any]:
    """
    Get paginated list of feedback items matching filters
    """
    try:
        # Get feedback items with filter
        items = await feedback_collector.get_feedback_by_filter(
            source=source,
            category=category,
            strategy_id=strategy_id,
            model_id=model_id,
            instrument=instrument,
            start_time=start_time,
            end_time=end_time,
            limit=pagination.page_size
        )
        
        # Get total count for pagination
        total_count = await feedback_collector.get_feedback_count(
            source=source,
            category=category,
            strategy_id=strategy_id,
            model_id=model_id,
            instrument=instrument,
            start_time=start_time,
            end_time=end_time
        )
        
        # Return paginated response
        return {
            "items": items,
            "total": total_count,
            "page": pagination.page,
            "page_size": pagination.page_size
        }
    except Exception as e:
        logger.error(f"Error retrieving feedback items: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve feedback items: {str(e)}"
        )


@router.get("/parameters/{strategy_id}")
async def get_parameter_performance(
    strategy_id: str,
    min_samples: int = Query(10, description="Minimum number of samples required for statistics"),
    parameter_tracking: ParameterTrackingService = Depends(get_parameter_tracking)
) -> Dict[str, Any]:
    """
    Get performance statistics for parameters of a strategy
    """
    try:
        # Get parameter performance data
        performance_data = await parameter_tracking.get_parameter_performance(
            strategy_id=strategy_id,
            min_samples=min_samples
        )
        
        # Filter out parameters with insufficient data
        parameters = []
        for param in performance_data.get("parameters", []):
            if param.get("sample_size", 0) >= min_samples:
                parameters.append(param)
        
        return {
            "strategy_id": strategy_id,
            "min_samples": min_samples,
            "parameters": parameters
        }
    except Exception as e:
        logger.error(f"Error retrieving parameter performance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve parameter performance: {str(e)}"
        )


@router.get("/strategy/versions/{strategy_id}")
async def get_strategy_version_history(
    strategy_id: str,
    mutation_service: StrategyMutationService = Depends(get_strategy_mutation)
) -> Dict[str, Any]:
    """
    Get version history for a strategy
    """
    try:
        # Get version history
        versions = await mutation_service.get_version_history(strategy_id)
        
        return {
            "strategy_id": strategy_id,
            "versions": versions,
            "version_count": len(versions)
        }
    except Exception as e:
        logger.error(f"Error retrieving strategy version history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve strategy version history: {str(e)}"
        )


@router.post("/strategy/mutate/{strategy_id}")
async def trigger_strategy_mutation(
    strategy_id: str,
    force: bool = Query(False, description="Force mutation even if conditions aren't optimal"),
    specific_parameters: Optional[List[str]] = Body(None, description="Specific parameters to mutate"),
    market_regime: Optional[str] = Query(None, description="Target market regime"),
    mutation_service: StrategyMutationService = Depends(get_strategy_mutation)
) -> Dict[str, Any]:
    """
    Trigger a mutation of a strategy
    """
    try:
        # Perform mutation
        result = await mutation_service.mutate_strategy(
            strategy_id=strategy_id,
            force=force,
            specific_parameters=specific_parameters,
            market_regime=market_regime
        )
        
        return result
    except Exception as e:
        logger.error(f"Error mutating strategy: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to mutate strategy: {str(e)}"
        )


@router.get("/strategy/mutation-effectiveness/{strategy_id}")
async def get_strategy_mutation_effectiveness(
    strategy_id: str,
    mutation_service: StrategyMutationService = Depends(get_strategy_mutation)
) -> Dict[str, Any]:
    """
    Get metrics about the effectiveness of mutations for a strategy
    """
    try:
        effectiveness = await mutation_service.get_mutation_effectiveness(strategy_id)
        return effectiveness
    except Exception as e:
        logger.error(f"Error retrieving mutation effectiveness: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve mutation effectiveness: {str(e)}"
        )


@router.post("/strategy/evaluate/{strategy_id}")
async def evaluate_strategy_versions(
    strategy_id: str,
    market_regime: Optional[str] = Query(None, description="Market regime to evaluate for"),
    mutation_service: StrategyMutationService = Depends(get_strategy_mutation)
) -> Dict[str, Any]:
    """
    Evaluate all versions of a strategy and activate the best one
    """
    try:
        result = await mutation_service.evaluate_and_select_best_version(
            strategy_id=strategy_id,
            market_regime=market_regime
        )
        
        return result
    except Exception as e:
        logger.error(f"Error evaluating strategy versions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to evaluate strategy versions: {str(e)}"
        )


@router.get("/system/status")
async def get_feedback_system_status(
    feedback_loop: FeedbackLoop = Depends(get_feedback_loop),
    feedback_connector: FeedbackLoopConnector = Depends(get_feedback_loop_connector),
    feedback_collector: TradingFeedbackCollector = Depends(get_trading_feedback_collector)
) -> FeedbackSystemStatusModel:
    """
    Get current status of the feedback system and its components
    """
    try:
        # Get status from each component
        loop_health = feedback_loop.get_health_metrics()
        connector_health = await feedback_connector.get_loop_health()
        collector_stats = await feedback_collector.get_statistics()
        
        # Determine overall status
        is_healthy = (
            loop_health.get("is_healthy", False) and
            connector_health.get("is_running", False)
        )
        status = "healthy" if is_healthy else "degraded"
        
        # Build combined status
        system_status = {
            "status": status,
            "components": {
                "feedback_loop": loop_health,
                "feedback_connector": connector_health,
                "feedback_collector": {
                    "total_collected": collector_stats.get("total_collected", 0),
                    "is_healthy": collector_stats.get("total_collected", 0) > 0
                }
            },
            "statistics": {
                "total_feedback": collector_stats.get("total_collected", 0),
                "feedback_count_by_source": collector_stats.get("by_source", {}),
                "feedback_count_by_category": collector_stats.get("by_category", {})
            },
            "last_updated": datetime.utcnow().isoformat()
        }
        
        return system_status
    except Exception as e:
        logger.error(f"Error retrieving system status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve system status: {str(e)}"
        )


@router.get("/insights/learning/{strategy_id}")
async def get_learning_insights(
    strategy_id: str,
    time_window_days: int = Query(30, description="Analysis window in days"),
    feedback_loop: FeedbackLoop = Depends(get_feedback_loop)
) -> FeedbackLearningModel:
    """
    Get insights learned from feedback for a specific strategy
    """
    try:
        # Generate insights
        insights = feedback_loop.generate_insights(strategy_id)
        
        # Get feedback statistics
        stats = feedback_loop.get_feedback_statistics(strategy_id)
        
        # Get influential parameters
        influential_params = feedback_loop.get_influential_parameters(strategy_id)
        
        # Extract success factors from insights
        success_factors = [
            insight for insight in insights
            if insight.get("type") == "success_factor"
        ]
        
        # Extract risk factors from insights
        risk_factors = [
            insight for insight in insights
            if insight.get("type") == "risk_factor"
        ]
        
        return {
            "strategy_id": strategy_id,
            "insights": insights,
            "total_feedback_count": stats.get("total", 0),
            "influential_parameters": influential_params,
            "success_factors": success_factors,
            "risk_factors": risk_factors,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate insights: {str(e)}"
        )


@router.post("/system/reset-stats")
async def reset_feedback_statistics(
    confirm: bool = Query(False, description="Confirmation required"),
    feedback_collector: TradingFeedbackCollector = Depends(get_trading_feedback_collector)
) -> Dict[str, str]:
    """
    Reset feedback statistics counters
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Confirmation required to reset statistics"
        )
        
    try:
        await feedback_collector.reset_statistics()
        
        return {"status": "statistics_reset"}
    except Exception as e:
        logger.error(f"Error resetting statistics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset statistics: {str(e)}"
        )
