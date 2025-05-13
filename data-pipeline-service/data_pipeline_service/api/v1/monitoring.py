"""
Monitoring API Module.

This module provides API endpoints for monitoring service performance.
"""
import logging
import os
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from data_pipeline_service.monitoring import get_query_performance_stats, get_slow_queries, update_slow_query_threshold
logger = logging.getLogger(__name__)
router = APIRouter(prefix='/monitoring', tags=['monitoring'], responses={(
    404): {'description': 'Not found'}})


from data_pipeline_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@router.get('/query-performance', response_model=Dict[str, Any])
@async_with_exception_handling
async def get_query_performance() ->Dict[str, Any]:
    """
    Get query performance statistics.

    Returns:
        Dictionary with query performance statistics
    """
    try:
        return get_query_performance_stats()
    except Exception as e:
        logger.error(f'Error getting query performance stats: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Error getting query performance stats: {str(e)}')


@router.get('/slow-queries', response_model=List[Dict[str, Any]])
@async_with_exception_handling
async def get_slow_query_list(limit: int=Query(10, ge=1, le=100)) ->List[Dict
    [str, Any]]:
    """
    Get the slowest queries.

    Args:
        limit: Maximum number of queries to return

    Returns:
        List of slow queries
    """
    try:
        return get_slow_queries(limit=limit)
    except Exception as e:
        logger.error(f'Error getting slow queries: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Error getting slow queries: {str(e)}')


@router.post('/slow-query-threshold', response_model=Dict[str, Any])
@async_with_exception_handling
async def set_slow_query_threshold(threshold: float=Query(..., gt=0)) ->Dict[
    str, Any]:
    """
    Update the slow query threshold.

    Args:
        threshold: New threshold in seconds

    Returns:
        Dictionary with result
    """
    try:
        update_slow_query_threshold(threshold)
        return {'status': 'success', 'message':
            f'Slow query threshold updated to {threshold}s'}
    except Exception as e:
        logger.error(f'Error updating slow query threshold: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Error updating slow query threshold: {str(e)}')


@router.get('/dashboard', response_class=HTMLResponse)
@async_with_exception_handling
async def get_query_dashboard() ->HTMLResponse:
    """
    Get the query performance dashboard.

    Returns:
        HTML response with the dashboard
    """
    try:
        dashboard_path = os.path.join(os.path.dirname(os.path.dirname(os.
            path.dirname(os.path.dirname(__file__)))), 'static',
            'query_dashboard.html')
        if not os.path.exists(dashboard_path):
            logger.error(f'Query dashboard file not found at {dashboard_path}')
            raise HTTPException(status_code=404, detail=
                'Dashboard file not found')
        with open(dashboard_path, 'r') as f:
            content = f.read()
        return HTMLResponse(content=content)
    except Exception as e:
        logger.error(f'Error serving query dashboard: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Error serving dashboard: {str(e)}')
