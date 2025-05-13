"""
Dashboard API Module.

This module provides API endpoints for serving dashboards.
"""
import logging
import os
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
logger = logging.getLogger(__name__)
router = APIRouter(prefix='/dashboard', tags=['dashboard'], responses={(404
    ): {'description': 'Not found'}})


from ml_integration_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@router.get('/cache', response_class=HTMLResponse)
@async_with_exception_handling
async def get_cache_dashboard() ->HTMLResponse:
    """
    Get the cache monitoring dashboard.
    
    Returns:
        HTML response with the dashboard
    """
    try:
        dashboard_path = os.path.join(os.path.dirname(os.path.dirname(os.
            path.dirname(__file__))), 'static', 'cache_dashboard.html')
        if not os.path.exists(dashboard_path):
            logger.error(f'Cache dashboard file not found at {dashboard_path}')
            raise HTTPException(status_code=404, detail=
                'Dashboard file not found')
        with open(dashboard_path, 'r') as f:
            content = f.read()
        return HTMLResponse(content=content)
    except Exception as e:
        logger.error(f'Error serving cache dashboard: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Error serving dashboard: {str(e)}')
