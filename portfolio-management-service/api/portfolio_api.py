"""
Portfolio API Endpoints.

Provides API endpoints for portfolio management operations.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, Path, Query, HTTPException, Depends, Body
from core_foundations.utils.logger import get_logger
from services.portfolio_service import PortfolioService
from core.position import Position, PositionCreate, PositionUpdate, PositionStatus, PositionPerformance
from core.account import AccountDetails
logger = get_logger('portfolio-api')
router = APIRouter(prefix='/api/v1/portfolio', tags=['Portfolio'])
portfolio_service = PortfolioService()


from core.exceptions_bridge_1 import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@router.post('/positions', response_model=Position)
@async_with_exception_handling
async def create_position(position: PositionCreate):
    """Create a new trading position."""
    try:
        result = portfolio_service.create_position(position)
        return result
    except Exception as e:
        logger.error(f'Error creating position: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to create position: {str(e)}')


@router.get('/positions/{position_id}', response_model=Position)
async def get_position(position_id: str=Path(..., description=
    'ID of the position to retrieve')):
    """Get a position by ID."""
    position = portfolio_service.get_position(position_id)
    if not position:
        raise HTTPException(status_code=404, detail=
            f'Position {position_id} not found')
    return position


@router.put('/positions/{position_id}', response_model=Position)
async def update_position(position_update: PositionUpdate, position_id: str
    =Path(..., description='ID of the position to update')):
    """Update a position."""
    position = portfolio_service.update_position(position_id, position_update)
    if not position:
        raise HTTPException(status_code=404, detail=
            f'Position {position_id} not found')
    return position


@router.post('/positions/{position_id}/close', response_model=Position)
async def close_position(position_id: str=Path(..., description=
    'ID of the position to close'), exit_price: float=Query(...,
    description='Exit price for the position')):
    """Close an open position."""
    position = portfolio_service.close_position(position_id, exit_price)
    if not position:
        raise HTTPException(status_code=404, detail=
            f'Position {position_id} not found or already closed')
    return position


@router.get('/accounts/{account_id}/summary', response_model=Dict[str, Any])
async def get_portfolio_summary(account_id: str=Path(..., description=
    'ID of the account')):
    """Get a comprehensive summary of the portfolio for an account."""
    summary = portfolio_service.get_portfolio_summary(account_id)
    if summary.get('status') == 'not_found':
        raise HTTPException(status_code=404, detail=
            f'Account {account_id} not found')
    return summary


@router.get('/accounts/{account_id}/historical-performance', response_model
    =Dict[str, Any])
@async_with_exception_handling
async def get_historical_performance(account_id: str=Path(..., description=
    'ID of the account'), period_days: int=Query(90, description=
    'Number of days to look back')):
    """Get historical performance data for an account."""
    try:
        return portfolio_service.get_historical_performance(account_id,
            period_days)
    except Exception as e:
        logger.error(f'Error retrieving historical performance: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to retrieve historical performance: {str(e)}')


@router.post('/accounts/{account_id}/update-prices', response_model=Dict[
    str, Any])
@async_with_exception_handling
async def update_position_prices(price_updates: Dict[str, float]=Body(...,
    description='Map of symbols to current prices'), account_id: str=Path(
    ..., description='ID of the account')):
    """Update current prices for all open positions and recalculate unrealized PnL."""
    try:
        return portfolio_service.update_position_prices(account_id,
            price_updates)
    except Exception as e:
        logger.error(f'Error updating position prices: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Failed to update prices: {str(e)}')


@router.post('/accounts/{account_id}/daily-snapshot', response_model=Dict[
    str, Any])
async def create_daily_snapshot(account_id: str=Path(..., description=
    'ID of the account')):
    """Create a daily snapshot of portfolio state."""
    snapshot = portfolio_service.create_daily_snapshot(account_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail=
            f'Failed to create snapshot for account {account_id}')
    return {'account_id': account_id, 'snapshot_id': snapshot.id,
        'timestamp': snapshot.timestamp, 'balance': snapshot.balance,
        'equity': snapshot.equity}
