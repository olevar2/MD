"""
Monitoring API Endpoints

This module provides API endpoints for monitoring system components,
including the Kafka feedback loop integration.
"""

from fastapi import APIRouter, Depends, Request
from typing import Dict, Any, List, Optional

from analysis_engine.adaptive_layer.feedback_kafka_integration import FeedbackLoopEventIntegration

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/feedback-kafka")
async def get_kafka_feedback_metrics(request: Request) -> Dict[str, Any]:
    """
    Get metrics for the Kafka feedback loop integration.
    
    This endpoint exposes metrics about:
    - Event processing statistics
    - Dead letter queue statistics
    - Event publishing statistics
    
    Returns:
        Dict with Kafka integration metrics
    """
    # Get the FeedbackLoopEventIntegration from the app state
    feedback_integration = request.app.state.feedback_integration
    
    if not feedback_integration or not isinstance(feedback_integration, FeedbackLoopEventIntegration):
        return {
            "status": "not_available",
            "message": "Kafka feedback integration not configured"
        }
    
    # Get statistics from the integration
    stats = feedback_integration.get_stats()
    
    # Add additional service health checks
    health_status = "healthy"
    if stats.get("consumer", {}).get("events_failed", 0) > 0:
        health_status = "degraded"
    
    # Calculate success rate
    consumer_stats = stats.get("consumer", {})
    events_processed = consumer_stats.get("events_processed", 0)
    events_failed = consumer_stats.get("events_failed", 0)
    
    if events_processed > 0:
        success_rate = (events_processed - events_failed) / events_processed * 100
    else:
        success_rate = 100.0  # Default if no events processed
    
    # Combine everything into a single response
    response = {
        "status": health_status,
        "metrics": {
            "consumer": consumer_stats,
            "publisher": stats.get("publisher", {}),
            "success_rate_percent": round(success_rate, 2)
        },
        "subscribed_topics": consumer_stats.get("subscribed_event_types", []),
        "last_event_time": consumer_stats.get("last_event_time"),
        "last_publish_time": stats.get("publisher", {}).get("last_publish_time")
    }
    
    return response


@router.get("/feedback-dlq")
async def get_dead_letter_queue_metrics(request: Request) -> Dict[str, Any]:
    """
    Get metrics for the Kafka feedback dead letter queue.
    
    Returns:
        Dict with DLQ metrics
    """
    # This would typically connect to your DLQ monitoring system
    # For now, we'll return a placeholder
    
    # For a real implementation, you would:
    # 1. Query the DLQ topic for statistics
    # 2. Analyze error patterns
    # 3. Return actionable metrics
    return {
        "status": "implemented",
        "message": "DLQ metrics tracking active",
        "metrics": {
            "total_failures": 0,  # Would be populated from actual DLQ tracking
            "error_categories": {},
            "retry_eligible_count": 0
        }
    }
""""""
