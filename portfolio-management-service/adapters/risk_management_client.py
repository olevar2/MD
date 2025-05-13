"""
Risk Management Service Client.

This module provides a client for interacting with the Risk Management Service.
It allows the Portfolio Management Service to check risk limits before executing trades.
"""
import aiohttp
import json
import os
from typing import Dict, List, Optional, Any
import logging
from urllib.parse import urljoin
from core_foundations.utils.logger import get_logger
from common_lib.resilience import retry_with_policy, register_common_retryable_exceptions
logger = get_logger('risk-management-client')


from core.exceptions_bridge_1 import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class RiskManagementClient:
    """
    Client for the Risk Management Service.

    This client allows the Portfolio Management Service to check risk limits
    and perform risk calculations before executing trades.
    """

    def __init__(self, base_url: str=None):
        """
        Initialize the Risk Management Service client.

        Args:
            base_url: Base URL for the Risk Management Service API
        """
        self.base_url = base_url or os.environ.get('RISK_MANAGEMENT_URL',
            'http://risk-management-service:8007')
        self.api_base = urljoin(self.base_url, '/api/v1/risk/')
        self.session = None
        self.api_key = os.environ.get('RISK_MANAGEMENT_API_KEY', '')

    async def _ensure_session(self):
        """Ensure that an aiohttp session exists."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()

    @retry_with_policy(max_attempts=3, base_delay=1.0, max_delay=10.0,
        jitter=True, exceptions=register_common_retryable_exceptions(),
        service_name='portfolio-management-service', operation_name=
        'risk_management_api_request')
    @async_with_exception_handling
    async def _make_request(self, method: str, endpoint: str, params:
        Optional[Dict]=None, data: Optional[Dict]=None) ->Dict:
        """
        Make a request to the Risk Management Service API with resilience patterns.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint to call
            params: Query parameters
            data: Request body data

        Returns:
            Response data as dictionary
        """
        await self._ensure_session()
        url = urljoin(self.api_base, endpoint)
        headers = {}
        if self.api_key:
            headers['X-API-Key'] = self.api_key
        try:
            logger.debug(f'Making {method} request to {url}')
            async with self.session.request(method, url, params=params,
                json=data, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientResponseError as e:
            logger.error(f'Error calling Risk Management Service: {str(e)}')
            if 400 <= e.status < 500 and e.status != 429:
                logger.warning(f'Client error {e.status}, not retrying')
                raise ValueError(f'Client error {e.status}: {e.message}')
            raise
        except Exception as e:
            logger.error(
                f'Unexpected error calling Risk Management Service: {str(e)}')
            raise

    async def check_position_risk(self, account_id: str, position_size:
        float, symbol: str, entry_price: float) ->Dict[str, Any]:
        """
        Check if a new position would violate risk limits.

        Args:
            account_id: Account ID
            position_size: Position size
            symbol: Trading symbol
            entry_price: Entry price

        Returns:
            Risk check result
        """
        params = {'account_id': account_id, 'position_size': position_size,
            'symbol': symbol, 'entry_price': entry_price}
        return await self._make_request('POST', 'check/position', params=params
            )

    async def check_portfolio_risk(self, account_id: str) ->Dict[str, Any]:
        """
        Check overall portfolio risk against limits.

        Args:
            account_id: Account ID

        Returns:
            Portfolio risk check result
        """
        return await self._make_request('POST', f'check/portfolio/{account_id}'
            )

    async def get_account_exposure(self, account_id: str) ->float:
        """
        Get the current account exposure from the Risk Management Service.

        Args:
            account_id: Account ID

        Returns:
            Account exposure as a percentage of account balance
        """
        result = await self.check_portfolio_risk(account_id)
        return result.get('account_exposure', 0.0)

    async def get_symbol_exposure(self, account_id: str, symbol: str) ->float:
        """
        Get the current exposure for a specific symbol from the Risk Management Service.

        Args:
            account_id: Account ID
            symbol: Trading symbol

        Returns:
            Symbol exposure as a percentage of account balance
        """
        result = await self.check_portfolio_risk(account_id)
        symbol_exposures = result.get('symbol_exposures', {})
        return symbol_exposures.get(symbol, 0.0)

    async def calculate_position_size(self, account_balance: float,
        risk_per_trade_pct: float, stop_loss_pips: float, pip_value: float,
        leverage: float=1.0) ->Dict[str, float]:
        """
        Calculate position size based on account risk percentage.

        Args:
            account_balance: Account balance
            risk_per_trade_pct: Risk percentage per trade
            stop_loss_pips: Stop loss in pips
            pip_value: Pip value in account currency
            leverage: Account leverage

        Returns:
            Position size calculation result
        """
        params = {'account_balance': account_balance, 'risk_per_trade_pct':
            risk_per_trade_pct, 'stop_loss_pips': stop_loss_pips,
            'pip_value': pip_value, 'leverage': leverage}
        return await self._make_request('POST', 'calculate/position-size',
            params=params)

    async def calculate_value_at_risk(self, portfolio_value: float,
        daily_returns: List[float], confidence_level: float=0.95,
        time_horizon_days: int=1) ->Dict[str, float]:
        """
        Calculate Value at Risk (VaR) for a portfolio.

        Args:
            portfolio_value: Portfolio value
            daily_returns: Historical daily returns as percentages
            confidence_level: Confidence level (default: 0.95)
            time_horizon_days: Time horizon in days (default: 1)

        Returns:
            VaR calculation result
        """
        params = {'portfolio_value': portfolio_value, 'confidence_level':
            confidence_level, 'time_horizon_days': time_horizon_days}
        data = {'daily_returns': daily_returns}
        return await self._make_request('POST', 'calculate/var', params=
            params, data=data)

    async def calculate_drawdown_risk(self, current_balance: float,
        historical_balances: List[float], max_drawdown_limit_pct: float=20.0
        ) ->Dict[str, Any]:
        """
        Calculate drawdown risk metrics.

        Args:
            current_balance: Current account balance
            historical_balances: List of historical account balances
            max_drawdown_limit_pct: Maximum allowed drawdown percentage

        Returns:
            Drawdown risk calculation result
        """
        params = {'current_balance': current_balance,
            'max_drawdown_limit_pct': max_drawdown_limit_pct}
        data = {'historical_balances': historical_balances}
        return await self._make_request('POST', 'calculate/drawdown',
            params=params, data=data)

    async def calculate_max_trades(self, account_balance: float,
        risk_per_trade_pct: float, portfolio_heat_limit_pct: float=20.0
        ) ->Dict[str, int]:
        """
        Calculate maximum number of simultaneous trades based on risk limits.

        Args:
            account_balance: Current account balance
            risk_per_trade_pct: Risk percentage per trade
            portfolio_heat_limit_pct: Maximum total portfolio risk percentage

        Returns:
            Maximum trades calculation result
        """
        params = {'account_balance': account_balance, 'risk_per_trade_pct':
            risk_per_trade_pct, 'portfolio_heat_limit_pct':
            portfolio_heat_limit_pct}
        return await self._make_request('POST', 'calculate/max-trades',
            params=params)
