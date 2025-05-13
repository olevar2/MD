"""
Multi-Asset Portfolio API Endpoints.

Provides API endpoints for multi-asset portfolio management operations.
"""
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Path, Query, HTTPException, Depends, Body
from core_foundations.utils.logger import get_logger
from core.multi_asset_portfolio_manager import MultiAssetPortfolioManager
from core.position import PositionCreate
logger = get_logger('multi-asset-portfolio-api')
router = APIRouter(prefix='/api/v1/multi-asset', tags=['Multi-Asset Portfolio']
    )
portfolio_manager = MultiAssetPortfolioManager()


from core.exceptions_bridge_1 import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@router.post('/positions', summary='Create multi-asset position')
@async_with_exception_handling
async def create_multi_asset_position(position_data: Dict[str, Any]=Body(...)):
    """
    Create a new trading position with multi-asset support.
    
    Automatically applies asset-specific parameters based on the symbol's asset class.
    """
    try:
        position = portfolio_manager.create_position(position_data)
        return position
    except Exception as e:
        logger.error(f'Error creating position: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to create position: {str(e)}')


@router.get('/portfolio/{account_id}/summary', summary=
    'Get multi-asset portfolio summary')
@async_with_exception_handling
async def get_portfolio_summary(account_id: str=Path(..., description=
    'Account ID to get portfolio summary for')):
    """
    Get a detailed portfolio summary with asset class breakdown.
    
    Returns portfolio metrics grouped by asset class and cross-asset risk metrics.
    """
    try:
        summary = portfolio_manager.get_portfolio_summary(account_id)
        return summary
    except Exception as e:
        logger.error(f'Error getting portfolio summary: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to get portfolio summary: {str(e)}')


@router.get('/portfolio/{account_id}/risk', summary=
    'Calculate unified risk metrics')
@async_with_exception_handling
async def calculate_unified_risk(account_id: str=Path(..., description=
    'Account ID to calculate risk metrics for')):
    """
    Calculate unified risk metrics across all asset classes in the portfolio.
    
    Returns value at risk, correlation-adjusted risk, and concentration risk.
    """
    try:
        risk_metrics = portfolio_manager.calculate_unified_risk(account_id)
        return risk_metrics
    except Exception as e:
        logger.error(f'Error calculating risk metrics: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to calculate risk metrics: {str(e)}')


@router.get('/portfolio/{account_id}/allocation-recommendations', summary=
    'Get asset allocation recommendations')
@async_with_exception_handling
async def get_allocation_recommendations(account_id: str=Path(...,
    description='Account ID to get recommendations for')):
    """
    Get asset allocation recommendations based on current portfolio.
    
    Returns current allocation and recommended allocation with explanation.
    """
    try:
        recommendations = (portfolio_manager.
            get_asset_allocation_recommendations(account_id))
        return recommendations
    except Exception as e:
        logger.error(f'Error getting allocation recommendations: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to get allocation recommendations: {str(e)}')
