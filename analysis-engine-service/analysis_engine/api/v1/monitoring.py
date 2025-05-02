"""
Monitoring API

This module provides API endpoints for monitoring the service.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, Query, HTTPException, status

from analysis_engine.core.monitoring.async_performance_monitor import get_async_monitor
from analysis_engine.core.monitoring.memory_monitor import get_memory_monitor

router = APIRouter(
    prefix="/monitoring",
    tags=["monitoring"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

@router.get("/async-performance", response_model=Dict[str, Any])
async def get_async_performance_metrics(
    operation: Optional[str] = Query(None, description="Filter by operation name")
):
    """
    Get async performance metrics.

    Args:
        operation: Optional operation name to filter by

    Returns:
        Dictionary of metrics
    """
    monitor = get_async_monitor()
    metrics = monitor.get_metrics(operation)

    if operation and not metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No metrics found for operation: {operation}"
        )

    return {
        "metrics": metrics,
        "operation_count": 1 if operation else len(metrics),
        "timestamp": "now"  # This will be converted to the current time by FastAPI
    }

@router.get("/memory", response_model=Dict[str, Any])
async def get_memory_metrics():
    """
    Get memory usage metrics.

    Returns:
        Dictionary of memory metrics
    """
    monitor = get_memory_monitor()
    metrics = await monitor.get_memory_usage()

    return {
        "metrics": metrics,
        "timestamp": "now"  # This will be converted to the current time by FastAPI
    }

@router.post("/async-performance/report", response_model=Dict[str, Any])
async def trigger_async_performance_report():
    """
    Trigger an immediate async performance report.

    Returns:
        Success message
    """
    monitor = get_async_monitor()
    monitor._log_metrics_report()

    return {
        "status": "success",
        "message": "Async performance report triggered",
        "timestamp": "now"  # This will be converted to the current time by FastAPI
    }