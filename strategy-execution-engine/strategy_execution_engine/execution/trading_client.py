"""
Trading Gateway Client for Forex Trading Platform

This module provides a client to interact with the trading gateway service,
allowing the strategy execution engine to send orders for execution.
"""
import requests
import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from common_lib.resilience import retry_with_policy, register_common_retryable_exceptions, RetryExhaustedException
logger = logging.getLogger(__name__)
try:
    register_common_retryable_exceptions([requests.exceptions.RequestException]
        )
except NameError:
    logger.warning(
        'requests library not found during retry registration in trading_client.'
        )


from strategy_execution_engine.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class TradingGatewayClient:
    """
    Client for interacting with the trading gateway service

    This class provides methods to send orders to the trading gateway
    and retrieve information about executions and positions.
    """

    def __init__(self, config: Dict[str, Any]=None):
        """
        Initialize the trading gateway client

        Args:
            config: Configuration dictionary for the client
        """
        self.config = config or {}
        self.base_url = self.config.get('gateway_service_url',
            'http://localhost:8004')
        self.timeout = self.config_manager.get('timeout_seconds', 5)
        self.max_attempts = self.config_manager.get('max_retries', 3) + 1
        self.api_key = self.config_manager.get('api_key')
        logger.info(
            f'Trading Gateway Client initialized with base URL: {self.base_url}'
            )

    @with_exception_handling
    def submit_order(self, order: Dict[str, Any]) ->Dict[str, Any]:
        """
        Submit an order to the trading gateway

        Args:
            order: Order details

        Returns:
            Dictionary with order submission result
        """
        endpoint = '/api/v1/orders'
        try:
            response = self._make_request('POST', endpoint, json_body=order)
            return response
        except (RetryExhaustedException, requests.exceptions.RequestException
            ) as e:
            logger.error(f'Order submission failed after retries: {e}')
            return {'success': False, 'error':
                f'Failed to submit order after multiple attempts: {str(e)}',
                'order_id': order.get('order_id')}
        except Exception as e:
            logger.error(f'Unexpected error during order submission: {e}',
                exc_info=True)
            return {'success': False, 'error':
                f'Unexpected error: {str(e)}', 'order_id': order.get(
                'order_id')}

    @with_exception_handling
    def cancel_order(self, order_id: str) ->Dict[str, Any]:
        """
        Cancel an existing order

        Args:
            order_id: ID of the order to cancel

        Returns:
            Dictionary with cancellation result
        """
        endpoint = f'/api/v1/orders/{order_id}/cancel'
        try:
            response = self._make_request('POST', endpoint)
            return response
        except (RetryExhaustedException, requests.exceptions.RequestException
            ) as e:
            logger.error(f'Order cancellation failed for {order_id}: {e}')
            return {'success': False, 'error':
                f'Failed to cancel order {order_id} after multiple attempts: {str(e)}'
                , 'order_id': order_id}
        except Exception as e:
            logger.error(
                f'Unexpected error during order cancellation for {order_id}: {e}'
                , exc_info=True)
            return {'success': False, 'error':
                f'Unexpected error: {str(e)}', 'order_id': order_id}

    @with_exception_handling
    def get_order_status(self, order_id: str) ->Dict[str, Any]:
        """
        Get the status of an order

        Args:
            order_id: ID of the order

        Returns:
            Dictionary with order status
        """
        endpoint = f'/api/v1/orders/{order_id}'
        try:
            response = self._make_request('GET', endpoint)
            return response
        except (RetryExhaustedException, requests.exceptions.RequestException
            ) as e:
            logger.error(f'Failed to get order status for {order_id}: {e}')
            return {'success': False, 'error':
                f'Failed to get status for order {order_id} after multiple attempts: {str(e)}'
                , 'order_id': order_id}
        except Exception as e:
            logger.error(
                f'Unexpected error getting order status for {order_id}: {e}',
                exc_info=True)
            return {'success': False, 'error':
                f'Unexpected error: {str(e)}', 'order_id': order_id}

    @with_exception_handling
    def get_open_positions(self) ->List[Dict[str, Any]]:
        """
        Get all open positions

        Returns:
            List of open positions
        """
        endpoint = '/api/v1/positions'
        try:
            response = self._make_request('GET', endpoint)
            if isinstance(response, dict) and isinstance(response.get(
                'positions'), list):
                return response.get('positions', [])
            else:
                logger.error(
                    f'Unexpected response format when getting open positions: {response}'
                    )
                return []
        except (RetryExhaustedException, requests.exceptions.RequestException
            ) as e:
            logger.error(f'Failed to get open positions: {e}')
            return []
        except Exception as e:
            logger.error(f'Unexpected error getting open positions: {e}',
                exc_info=True)
            return []

    @with_exception_handling
    def close_position(self, position_id: str) ->Dict[str, Any]:
        """
        Close an open position

        Args:
            position_id: ID of the position to close

        Returns:
            Dictionary with position closure result
        """
        endpoint = f'/api/v1/positions/{position_id}/close'
        try:
            response = self._make_request('POST', endpoint)
            return response
        except (RetryExhaustedException, requests.exceptions.RequestException
            ) as e:
            logger.error(f'Position closure failed for {position_id}: {e}')
            return {'success': False, 'error':
                f'Failed to close position {position_id} after multiple attempts: {str(e)}'
                , 'position_id': position_id}
        except Exception as e:
            logger.error(
                f'Unexpected error closing position {position_id}: {e}',
                exc_info=True)
            return {'success': False, 'error':
                f'Unexpected error: {str(e)}', 'position_id': position_id}

    @with_exception_handling
    def get_account_info(self) ->Dict[str, Any]:
        """
        Get account information

        Returns:
            Dictionary with account information
        """
        endpoint = '/api/v1/account'
        try:
            response = self._make_request('GET', endpoint)
            return response
        except (RetryExhaustedException, requests.exceptions.RequestException
            ) as e:
            logger.error(f'Failed to get account info: {e}')
            return {'error':
                f'Failed to get account info after multiple attempts: {str(e)}'
                }
        except Exception as e:
            logger.error(f'Unexpected error getting account info: {e}',
                exc_info=True)
            return {'error': f'Unexpected error: {str(e)}'}

    @retry_with_policy(exceptions=[requests.exceptions.RequestException])
    def _make_request(self, method: str, endpoint: str, params: Dict[str,
        Any]=None, json_body: Dict[str, Any]=None) ->Dict[str, Any]:
        """
        Make an HTTP request to the trading gateway service with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            json_body: JSON body data

        Returns:
            Response data as dictionary

        Raises:
            core_foundations.resilience.retry_policy.RetryExhaustedException: If all retries fail.
            requests.exceptions.HTTPError: For non-retryable HTTP errors (e.g., 4xx).
            Exception: For other unexpected errors.
        """
        url = f'{self.base_url}{endpoint}'
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        response = requests.request(method=method, url=url, headers=headers,
            params=params, json=json_body, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
