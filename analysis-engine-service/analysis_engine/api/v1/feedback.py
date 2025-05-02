"""
Feedback API Routes

This module implements API routes for accessing feedback-related functionality,
providing endpoints to retrieve feedback statistics, trigger model retraining,
and manage feedback rules.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Query, HTTPException, Body

from analysis_engine.adaptive_layer.feedback_integration_service import FeedbackIntegrationService
from analysis_engine.services.service_factory import get_feedback_service

router = APIRouter(
    prefix="/feedback",
    tags=["feedback"],
    responses={404: {"description": "Not found"}},
)


@router.get("/statistics")
async def get_feedback_statistics(
    strategy_id: Optional[str] = Query(None, description="Filter by strategy ID"),
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    instrument: Optional[str] = Query(None, description="Filter by instrument"),
    start_time: Optional[datetime] = Query(None, description="Start time for filtering (ISO format)"),
    end_time: Optional[datetime] = Query(None, description="End time for filtering (ISO format)"),
    feedback_service: FeedbackIntegrationService = Depends(get_feedback_service)
):
    """
    Get feedback statistics with optional filtering.
    
    This endpoint returns statistics about collected feedback, including counts by
    source, category, and outcome. Results can be filtered by strategy, model,
    instrument, and time range.
    """
    try:
        stats = await feedback_service.get_feedback_statistics(
            strategy_id=strategy_id,
            model_id=model_id,
            instrument=instrument,
            start_time=start_time,
            end_time=end_time
        )
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving feedback statistics: {str(e)}")


@router.post("/model/retrain/{model_id}")
async def trigger_model_retraining(
    model_id: str,
    feedback_service: FeedbackIntegrationService = Depends(get_feedback_service)
):
    """
    Trigger retraining of a specific model based on collected feedback.
    
    This endpoint manually initiates the retraining process for a model,
    incorporating feedback data collected since the last training.
    """
    try:
        result = await feedback_service.trigger_model_retraining(model_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error triggering model retraining: {str(e)}")


@router.put("/rules")
async def update_feedback_rules(
    rule_updates: Dict[str, Any] = Body(..., description="Rule updates to apply"),
    feedback_service: FeedbackIntegrationService = Depends(get_feedback_service)
):
    """
    Update feedback categorization rules.
    
    This endpoint allows modification of the rules used to categorize and route feedback,
    enabling adaptation of the feedback system behavior without code changes.
    """
    try:
        results = await feedback_service.update_feedback_rules(rule_updates)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating feedback rules: {str(e)}")


@router.get("/parameters/{strategy_id}")
async def get_parameter_performance(
    strategy_id: str,
    min_samples: int = Query(10, description="Minimum number of samples required for parameter statistics"),
    feedback_service: FeedbackIntegrationService = Depends(get_feedback_service)
):
    """
    Get performance statistics for strategy parameters.
    
    This endpoint returns performance metrics for different parameter values used in a strategy,
    enabling analysis of which parameter settings yield the best results.
    """
    try:
        stats = await feedback_service.get_feedback_statistics(strategy_id=strategy_id)
        
        # Check if parameter performance data exists
        if "parameter_performance" not in stats:
            return {"message": "No parameter performance data available", "parameters": {}}
        
        # Filter parameters by minimum sample size
        filtered_parameters = {}
        for param_key, metrics in stats["parameter_performance"].items():
            total_samples = metrics["success_count"] + metrics["failure_count"]
            if total_samples >= min_samples:
                filtered_parameters[param_key] = metrics
        
        return {
            "strategy_id": strategy_id,
            "min_samples": min_samples,
            "parameters": filtered_parameters
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving parameter performance: {str(e)}")
