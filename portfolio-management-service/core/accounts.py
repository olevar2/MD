"""
Accounts API Module.

API endpoints for managing trading accounts and balances.
"""
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Path, Body
from core_foundations.utils.logger import get_logger
from core.account import AccountBalance
from services.portfolio_service import PortfolioService
logger = get_logger('accounts-api')
router = APIRouter()
portfolio_service = PortfolioService()


from core.exceptions_bridge_1 import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@router.get('/{account_id}/balance', response_model=AccountBalance)
async def get_account_balance(account_id: str=Path(..., description=
    'Account ID')) ->AccountBalance:
    """
    Get the current account balance.
    
    Args:
        account_id: Account ID
        
    Returns:
        Current account balance
    """
    balance = portfolio_service.get_account_balance(account_id)
    if not balance:
        logger.warning(f'API: No balance found for account {account_id}')
        raise HTTPException(status_code=404, detail='Account balance not found'
            )
    return balance


@router.get('/{account_id}/summary', response_model=Dict[str, Any])
@async_with_exception_handling
async def get_account_summary(account_id: str=Path(..., description=
    'Account ID')) ->Dict[str, Any]:
    """
    Get summary information for an account.
    
    Args:
        account_id: Account ID
        
    Returns:
        Dictionary with account summary information
    """
    try:
        summary = portfolio_service.get_account_summary(account_id)
        return summary
    except Exception as e:
        logger.error(f'API: Unexpected error getting account summary: {str(e)}'
            )
        raise HTTPException(status_code=500, detail=
            'Failed to get account summary')


@router.post('/{account_id}/initialize', response_model=AccountBalance)
@async_with_exception_handling
async def initialize_account_balance(account_id: str=Path(..., description=
    'Account ID'), initial_balance: float=Query(..., description=
    'Initial account balance', gt=0)) ->AccountBalance:
    """
    Initialize balance for a new account.
    
    Args:
        account_id: Account ID
        initial_balance: Initial account balance
        
    Returns:
        Created account balance
    """
    try:
        balance = portfolio_service.initialize_account_balance(account_id,
            initial_balance)
        logger.info(
            f'API: Initialized balance for account {account_id}: {initial_balance}'
            )
        return balance
    except ValueError as e:
        logger.error(f'API: Failed to initialize account balance: {str(e)}')
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            f'API: Unexpected error initializing account balance: {str(e)}')
        raise HTTPException(status_code=500, detail=
            'Failed to initialize account balance')


@router.post('/{account_id}/deposit', response_model=AccountBalance)
@async_with_exception_handling
async def add_funds(account_id: str=Path(..., description='Account ID'),
    amount: float=Query(..., description='Amount to deposit', gt=0),
    description: str=Query('', description='Optional description')
    ) ->AccountBalance:
    """
    Add funds to an account.
    
    Args:
        account_id: Account ID
        amount: Amount to add
        description: Optional description
        
    Returns:
        Updated account balance
    """
    try:
        balance = portfolio_service.add_funds(account_id, amount, description)
        if not balance:
            logger.warning(f'API: No balance found for account {account_id}')
            raise HTTPException(status_code=404, detail=
                'Account balance not found')
        logger.info(f'API: Added {amount} to account {account_id}')
        return balance
    except ValueError as e:
        logger.error(f'API: Failed to add funds: {str(e)}')
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f'API: Unexpected error adding funds: {str(e)}')
        raise HTTPException(status_code=500, detail='Failed to add funds')


@router.post('/{account_id}/withdraw', response_model=AccountBalance)
@async_with_exception_handling
async def withdraw_funds(account_id: str=Path(..., description='Account ID'
    ), amount: float=Query(..., description='Amount to withdraw', gt=0),
    description: str=Query('', description='Optional description')
    ) ->AccountBalance:
    """
    Withdraw funds from an account.
    
    Args:
        account_id: Account ID
        amount: Amount to withdraw
        description: Optional description
        
    Returns:
        Updated account balance
    """
    try:
        balance = portfolio_service.withdraw_funds(account_id, amount,
            description)
        if not balance:
            logger.warning(f'API: No balance found for account {account_id}')
            raise HTTPException(status_code=404, detail=
                'Account balance not found')
        logger.info(f'API: Withdrew {amount} from account {account_id}')
        return balance
    except ValueError as e:
        logger.error(f'API: Failed to withdraw funds: {str(e)}')
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f'API: Unexpected error withdrawing funds: {str(e)}')
        raise HTTPException(status_code=500, detail='Failed to withdraw funds')


@router.get('/{account_id}/performance', response_model=Dict[str, Any])
@async_with_exception_handling
async def get_performance_metrics(account_id: str=Path(..., description=
    'Account ID'), start_date: date=Query(..., description='Start date'),
    end_date: date=Query(..., description='End date')) ->Dict[str, Any]:
    """
    Get performance metrics for an account.
    
    Args:
        account_id: Account ID
        start_date: Start date
        end_date: End date
        
    Returns:
        Dictionary with performance metrics
    """
    try:
        metrics = portfolio_service.get_performance_metrics(account_id,
            start_date, end_date)
        return metrics
    except Exception as e:
        logger.error(
            f'API: Unexpected error getting performance metrics: {str(e)}')
        raise HTTPException(status_code=500, detail=
            'Failed to get performance metrics')


@router.get('/{account_id}/history', response_model=List[Dict[str, Any]])
@async_with_exception_handling
async def get_balance_history(account_id: str=Path(..., description=
    'Account ID'), field: str=Query('equity', description=
    'Balance field to retrieve'), start_date: Optional[datetime]=Query(None,
    description='Optional start date filter'), end_date: Optional[datetime]
    =Query(None, description='Optional end date filter'), interval: str=
    Query('1h', description='Time interval for aggregation')) ->List[Dict[
    str, Any]]:
    """
    Get historical balance data for charting.
    
    Args:
        account_id: Account ID
        field: Balance field to retrieve ('equity', 'balance', etc.)
        start_date: Optional start date filter
        end_date: Optional end date filter
        interval: Time interval for aggregation ('1h', '6h', '1d')
        
    Returns:
        List of data points with timestamp and value
    """
    try:
        history = portfolio_service.get_balance_history(account_id, field,
            start_date, end_date, interval)
        return history
    except ValueError as e:
        logger.error(f'API: Failed to get balance history: {str(e)}')
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f'API: Unexpected error getting balance history: {str(e)}'
            )
        raise HTTPException(status_code=500, detail=
            'Failed to get balance history')


@router.post('/{account_id}/info', response_model=Dict[str, Any])
@async_with_exception_handling
async def create_or_update_account_info(account_id: str=Path(...,
    description='Account ID'), data: Dict[str, Any]=Body(..., description=
    'Account data')) ->Dict[str, Any]:
    """
    Create or update account information.
    
    Args:
        account_id: Account ID
        data: Account data
        
    Returns:
        Updated account information
    """
    try:
        info = portfolio_service.create_or_update_account_info(account_id, data
            )
        logger.info(f'API: Updated information for account {account_id}')
        return info
    except Exception as e:
        logger.error(f'API: Unexpected error updating account info: {str(e)}')
        raise HTTPException(status_code=500, detail=
            'Failed to update account information')
