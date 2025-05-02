"""
Feedback System API Router

This module provides REST API endpoints for the Feedback Loop System, allowing access
to feedback insights, manual feedback submission, and feedback system management.
"""

from typing import Dict, List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field

# Import models
from core_foundations.models.feedback import (
    FeedbackCategory, FeedbackSource, FeedbackStatus, 
    FeedbackEntry, FeedbackInsight, FeedbackStatistics
)

router = APIRouter(prefix="/feedback", tags=["feedback"])


class FeedbackSubmissionModel(BaseModel):
    """Model for manual feedback submission"""
    source: FeedbackSource
    category: FeedbackCategory
    instrument_id: str = Field(..., description="The trading instrument identifier")
    strategy_id: Optional[str] = Field(None, description="The strategy that generated the signal/trade")
    model_id: Optional[str] = Field(None, description="The model that generated the prediction")
    description: str = Field(..., description="Description of the feedback")
    metadata: Dict = Field(default_factory=dict, description="Additional contextual metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the feedback was generated")


@router.get("/insights", response_model=List[FeedbackInsight])
async def get_feedback_insights(
    request: Request,
    source: Optional[FeedbackSource] = None,
    category: Optional[FeedbackCategory] = None,
    instrument_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = Query(default=100, le=1000)
):
    """
    Get insights generated from feedback data analysis
    
    These insights can be used to improve trading strategies and models
    """
    feedback_service = request.app.state.feedback_integration_service
    
    # Get insights from the feedback service
    insights = await feedback_service.get_feedback_insights(
        source=source,
        category=category,
        instrument_id=instrument_id,
        start_date=start_date,
        end_date=end_date,
        limit=limit
    )
    
    return insights


@router.get("/statistics", response_model=FeedbackStatistics)
async def get_feedback_statistics(
    request: Request,
    source: Optional[FeedbackSource] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """
    Get aggregated statistics about the collected feedback
    """
    feedback_service = request.app.state.feedback_integration_service
    
    stats = await feedback_service.get_statistics(
        source=source,
        start_date=start_date,
        end_date=end_date
    )
    
    return stats


@router.post("/submit", status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    request: Request,
    feedback: FeedbackSubmissionModel
):
    """
    Submit manual feedback to be processed by the feedback loop system
    
    This endpoint allows traders or analysts to submit manual observations or corrections
    """
    feedback_collector = request.app.state.feedback_collector
    
    try:
        feedback_id = await feedback_collector.collect(
            source=feedback.source,
            category=feedback.category,
            instrument_id=feedback.instrument_id,
            strategy_id=feedback.strategy_id,
            model_id=feedback.model_id,
            description=feedback.description,
            metadata=feedback.metadata,
            timestamp=feedback.timestamp
        )
        
        return {"feedback_id": feedback_id, "status": "submitted"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit feedback: {str(e)}"
        )


@router.post("/retrain-model/{model_id}")
async def request_model_retraining(
    request: Request,
    model_id: str,
    force: bool = Query(default=False, description="Force retraining even if not necessary")
):
    """
    Request retraining of a specific model based on collected feedback
    
    This can be triggered manually when needed, instead of waiting for automatic retraining
    """
    feedback_service = request.app.state.feedback_integration_service
    
    try:
        result = await feedback_service.manual_retrain_request(model_id, force=force)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate model retraining: {str(e)}"
        )


@router.get("/entries", response_model=List[FeedbackEntry])
async def get_feedback_entries(
    request: Request,
    source: Optional[FeedbackSource] = None,
    category: Optional[FeedbackCategory] = None,
    instrument_id: Optional[str] = None,
    strategy_id: Optional[str] = None,
    model_id: Optional[str] = None,
    status: Optional[FeedbackStatus] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0)
):
    """
    Get collected feedback entries matching the specified filters
    """
    feedback_collector = request.app.state.feedback_collector
    
    entries = await feedback_collector.get_entries(
        source=source,
        category=category,
        instrument_id=instrument_id,
        strategy_id=strategy_id,
        model_id=model_id,
        status=status,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset
    )
    
    return entries


@router.get("/system/status")
async def get_feedback_system_status(request: Request):
    """
    Get the current status of the feedback system components
    
    Returns operational status of all feedback loop components
    """
    feedback_service = request.app.state.feedback_integration_service
    
    status = await feedback_service.get_system_status()
    return status


@router.post("/system/reset-metrics")
async def reset_feedback_metrics(
    request: Request,
    confirm: bool = Query(default=False, description="Confirmation required to reset metrics")
):
    """
    Reset collected feedback metrics
    
    This will clear accumulated metrics but preserve the feedback entries
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Confirmation required to reset metrics"
        )
        
    feedback_service = request.app.state.feedback_integration_service
    
    try:
        await feedback_service.reset_metrics()
        return {"status": "success", "message": "Feedback metrics have been reset"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset metrics: {str(e)}"
        )
