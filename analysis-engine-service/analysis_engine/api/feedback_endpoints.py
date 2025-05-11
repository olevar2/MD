"""
API Endpoints for Monitoring the Feedback System

Provides endpoints to query the status, recent activity, and statistics
of the adaptive feedback loop and its components.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)
router = APIRouter(prefix='/feedback', tags=['Feedback System Monitoring'])
_RECENT_FEEDBACK_STORE: List[Dict[str, Any]] = []
_MAX_STORE_SIZE = 100

class FeedbackSystemStatus(BaseModel):
    status: str = 'Operational'
    consumer_status: str = 'Connected'
    last_event_processed_at: Optional[datetime] = None

class FeedbackEventSummary(BaseModel):
    feedback_id: str
    feedback_type: str
    source: str
    timestamp: datetime

class FeedbackStats(BaseModel):
    total_processed: int
    processed_by_type: Dict[str, int]

def add_to_recent_store(event_summary: FeedbackEventSummary):
    """Placeholder to simulate storing recent events."""
    _RECENT_FEEDBACK_STORE.insert(0, event_summary.dict())
    if len(_RECENT_FEEDBACK_STORE) > _MAX_STORE_SIZE:
        _RECENT_FEEDBACK_STORE.pop()

@router.get('/status', response_model=FeedbackSystemStatus)
async def get_feedback_system_status():
    """Get the current operational status of the feedback system."""
    logger.info('Request received for feedback system status.')
    try:
        last_event_time = _RECENT_FEEDBACK_STORE[0]['timestamp'] if _RECENT_FEEDBACK_STORE else None
        return FeedbackSystemStatus(last_event_processed_at=last_event_time)
    except Exception as e:
        logger.error(f'Error fetching feedback system status: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail='Internal server error fetching status')

@router.get('/recent', response_model=List[FeedbackEventSummary])
async def get_recent_feedback_events(limit: int=20):
    """Retrieve a list of recently processed feedback events."""
    logger.info(f'Request received for recent feedback events (limit: {limit}).')
    try:
        return _RECENT_FEEDBACK_STORE[:limit]
    except Exception as e:
        logger.error(f'Error fetching recent feedback events: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail='Internal server error fetching recent events')

@router.get('/stats', response_model=FeedbackStats)
async def get_feedback_statistics():
    """Get statistics about processed feedback events."""
    logger.info('Request received for feedback statistics.')
    try:
        total = len(_RECENT_FEEDBACK_STORE)
        by_type = {}
        for event in _RECENT_FEEDBACK_STORE:
            type = event.get('feedback_type', 'unknown')
            by_type[type] = by_type.get(type, 0) + 1
        return FeedbackStats(total_processed=total, processed_by_type=by_type)
    except Exception as e:
        logger.error(f'Error fetching feedback statistics: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail='Internal server error fetching statistics')

def record_feedback_processed(event: Any):
    """Call this function after successfully processing a feedback event."""
    try:
        summary = FeedbackEventSummary(feedback_id=str(event.feedback_id), feedback_type=event.feedback_type, source=event.source, timestamp=event.timestamp)
        add_to_recent_store(summary)
        logger.debug(f'Recorded processed feedback event summary: {summary.feedback_id}')
    except Exception as e:
        logger.warning(f'Could not record feedback event summary: {e}', exc_info=True)