"""
Positions API Module.

API endpoints for managing trading positions.
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Path, status
from core_foundations.utils.logger import get_logger
from core.position import Position, PositionCreate, PositionUpdate
from services.portfolio_service import PortfolioService
from portfolio_management_service.error import convert_to_http_exception, PortfolioManagementError, PortfolioNotFoundError, PositionNotFoundError, InsufficientBalanceError, PortfolioOperationError
logger = get_logger('positions-api')
router = APIRouter()
portfolio_service = PortfolioService()
POSITION_ID_DESC = 'Position ID'
POSITION_NOT_FOUND = 'Position not found'


from core.exceptions_bridge_1 import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@router.post('/', response_model=Position, status_code=201)
@async_with_exception_handling
async def create_position(position_data: PositionCreate) ->Position:
    """
    Create a new trading position.

    Args:
        position_data: Data for the new position

    Returns:
        Created position
    """
    try:
        position = await portfolio_service.create_position(position_data)
        logger.info(f'API: Created position for {position_data.symbol}')
        return position
    except PortfolioManagementError as e:
        http_exc = convert_to_http_exception(e)
        raise http_exc
    except Exception as e:
        logger.error(f'API: Unexpected error creating position: {str(e)}')
        raise HTTPException(status_code=status.
            HTTP_500_INTERNAL_SERVER_ERROR, detail={'error_code':
            'UNEXPECTED_ERROR', 'message': 'Failed to create position'})


@router.get('/{position_id}', response_model=Position)
@async_with_exception_handling
async def get_position(position_id: str=Path(..., description=POSITION_ID_DESC)
    ) ->Position:
    """
    Get a position by ID.

    Args:
        position_id: Position ID

    Returns:
        Position details
    """
    try:
        position = await portfolio_service.get_position(position_id)
        return position
    except PortfolioManagementError as e:
        http_exc = convert_to_http_exception(e)
        raise http_exc
    except Exception as e:
        logger.error(f'API: Unexpected error retrieving position: {str(e)}')
        raise HTTPException(status_code=status.
            HTTP_500_INTERNAL_SERVER_ERROR, detail={'error_code':
            'UNEXPECTED_ERROR', 'message': 'Failed to retrieve position'})


@router.patch('/{position_id}', response_model=Position)
@async_with_exception_handling
async def update_position(position_update: PositionUpdate, position_id: str
    =Path(..., description=POSITION_ID_DESC)) ->Position:
    """
    Update a position.

    Args:
        position_id: Position ID
        position_update: Data to update

    Returns:
        Updated position
    """
    try:
        position = await portfolio_service.update_position(position_id,
            position_update)
        logger.info(f'API: Updated position {position_id}')
        return position
    except PortfolioManagementError as e:
        http_exc = convert_to_http_exception(e)
        raise http_exc
    except Exception as e:
        logger.error(f'API: Unexpected error updating position: {str(e)}')
        raise HTTPException(status_code=status.
            HTTP_500_INTERNAL_SERVER_ERROR, detail={'error_code':
            'UNEXPECTED_ERROR', 'message': 'Failed to update position'})


@router.post('/{position_id}/close', response_model=Position)
@async_with_exception_handling
async def close_position(position_id: str=Path(..., description=
    POSITION_ID_DESC), close_price: float=Query(..., description=
    'Closing price')) ->Position:
    """
    Close a position.

    Args:
        position_id: Position ID
        close_price: Closing price

    Returns:
        Closed position
    """
    try:
        position = await portfolio_service.close_position(position_id,
            close_price)
        logger.info(
            f'API: Closed position {position_id} with realized P&L: {position.realized_pnl}'
            )
        return position
    except PortfolioManagementError as e:
        http_exc = convert_to_http_exception(e)
        raise http_exc
    except Exception as e:
        logger.error(f'API: Unexpected error closing position: {str(e)}')
        raise HTTPException(status_code=status.
            HTTP_500_INTERNAL_SERVER_ERROR, detail={'error_code':
            'UNEXPECTED_ERROR', 'message': 'Failed to close position'})


@router.get('/portfolio/{account_id}/summary', response_model=Dict[str, Any])
@async_with_exception_handling
async def get_portfolio_summary(account_id: str=Path(..., description=
    'Account ID')) ->Dict[str, Any]:
    """
    Get a comprehensive summary of the portfolio for an account.

    Args:
        account_id: Account ID

    Returns:
        Dictionary with portfolio summary information
    """
    try:
        summary = await portfolio_service.get_portfolio_summary(account_id)
        return summary
    except PortfolioManagementError as e:
        http_exc = convert_to_http_exception(e)
        raise http_exc
    except Exception as e:
        logger.error(
            f'API: Unexpected error getting portfolio summary: {str(e)}')
        raise HTTPException(status_code=status.
            HTTP_500_INTERNAL_SERVER_ERROR, detail={'error_code':
            'UNEXPECTED_ERROR', 'message': 'Failed to get portfolio summary'})


@router.get('/portfolio/{account_id}/performance', response_model=Dict[str,
    Any])
@async_with_exception_handling
async def get_historical_performance(account_id: str=Path(..., description=
    'Account ID'), period_days: int=Query(90, description=
    'Number of days to look back')) ->Dict[str, Any]:
    """
    Get historical performance data for an account.

    Args:
        account_id: Account ID
        period_days: Number of days to look back

    Returns:
        Dictionary with historical performance data
    """
    try:
        performance_data = await portfolio_service.get_historical_performance(
            account_id, period_days)
        return performance_data
    except PortfolioManagementError as e:
        http_exc = convert_to_http_exception(e)
        raise http_exc
    except Exception as e:
        logger.error(
            f'API: Unexpected error getting historical performance: {str(e)}')
        raise HTTPException(status_code=status.
            HTTP_500_INTERNAL_SERVER_ERROR, detail={'error_code':
            'UNEXPECTED_ERROR', 'message':
            'Failed to get historical performance'})
