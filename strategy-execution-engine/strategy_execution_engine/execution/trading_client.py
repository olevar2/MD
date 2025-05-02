"""
Trading Gateway Client for Forex Trading Platform

This module provides a client to interact with the trading gateway service,
allowing the strategy execution engine to send orders for execution.
"""
import requests
import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple

# Import the centralized retry policy
from common_lib.resilience import retry_with_policy, register_common_retryable_exceptions, RetryExhaustedException

logger = logging.getLogger(__name__)

# Register common requests exceptions for retry
# This might already be done in retry_policy.py, but explicit registration here is fine
try:
    register_common_retryable_exceptions([requests.exceptions.RequestException])
except NameError: # Handle case where requests might not be installed when retry_policy is loaded
    logger.warning("requests library not found during retry registration in trading_client.")


class TradingGatewayClient:
    """
    Client for interacting with the trading gateway service

    This class provides methods to send orders to the trading gateway
    and retrieve information about executions and positions.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the trading gateway client

        Args:
            config: Configuration dictionary for the client
        """
        self.config = config or {}
        self.base_url = self.config.get("gateway_service_url", "http://localhost:8004")
        self.timeout = self.config.get("timeout_seconds", 5)
        # Use max_attempts directly in the decorator
        self.max_attempts = self.config.get("max_retries", 3) + 1 # +1 because tenacity counts retries, policy counts attempts
        # Optional API key for authentication
        self.api_key = self.config.get("api_key")

        logger.info(f"Trading Gateway Client initialized with base URL: {self.base_url}")

    def submit_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit an order to the trading gateway

        Args:
            order: Order details

        Returns:
            Dictionary with order submission result
        """
        endpoint = "/api/v1/orders"

        try:
            response = self._make_request("POST", endpoint, json_body=order)
            return response
        except (RetryExhaustedException, requests.exceptions.RequestException) as e:
            logger.error(f"Order submission failed after retries: {e}")
            return {
                "success": False,
                "error": f"Failed to submit order after multiple attempts: {str(e)}",
                "order_id": order.get("order_id")
            }
        except Exception as e: # Catch other unexpected errors
            logger.error(f"Unexpected error during order submission: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "order_id": order.get("order_id")
            }

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an existing order

        Args:
            order_id: ID of the order to cancel

        Returns:
            Dictionary with cancellation result
        """
        endpoint = f"/api/v1/orders/{order_id}/cancel"

        try:
            response = self._make_request("POST", endpoint)
            return response
        except (RetryExhaustedException, requests.exceptions.RequestException) as e:
            logger.error(f"Order cancellation failed for {order_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to cancel order {order_id} after multiple attempts: {str(e)}",
                "order_id": order_id
            }
        except Exception as e:
            logger.error(f"Unexpected error during order cancellation for {order_id}: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "order_id": order_id
            }

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get the status of an order

        Args:
            order_id: ID of the order

        Returns:
            Dictionary with order status
        """
        endpoint = f"/api/v1/orders/{order_id}"

        try:
            response = self._make_request("GET", endpoint)
            return response
        except (RetryExhaustedException, requests.exceptions.RequestException) as e:
            logger.error(f"Failed to get order status for {order_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to get status for order {order_id} after multiple attempts: {str(e)}",
                "order_id": order_id
            }
        except Exception as e:
            logger.error(f"Unexpected error getting order status for {order_id}: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "order_id": order_id
            }

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Get all open positions

        Returns:
            List of open positions
        """
        endpoint = "/api/v1/positions"

        try:
            response = self._make_request("GET", endpoint)
            # Ensure response is a dict and contains 'positions' list
            if isinstance(response, dict) and isinstance(response.get("positions"), list):
                return response.get("positions", [])
            else:
                logger.error(f"Unexpected response format when getting open positions: {response}")
                return []
        except (RetryExhaustedException, requests.exceptions.RequestException) as e:
            logger.error(f"Failed to get open positions: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting open positions: {e}", exc_info=True)
            return []

    def close_position(self, position_id: str) -> Dict[str, Any]:
        """
        Close an open position

        Args:
            position_id: ID of the position to close

        Returns:
            Dictionary with position closure result
        """
        endpoint = f"/api/v1/positions/{position_id}/close"

        try:
            response = self._make_request("POST", endpoint)
            return response
        except (RetryExhaustedException, requests.exceptions.RequestException) as e:
            logger.error(f"Position closure failed for {position_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to close position {position_id} after multiple attempts: {str(e)}",
                "position_id": position_id
            }
        except Exception as e:
            logger.error(f"Unexpected error closing position {position_id}: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "position_id": position_id
            }

    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information

        Returns:
            Dictionary with account information
        """
        endpoint = "/api/v1/account"

        try:
            response = self._make_request("GET", endpoint)
            return response
        except (RetryExhaustedException, requests.exceptions.RequestException) as e:
            logger.error(f"Failed to get account info: {e}")
            return {"error": f"Failed to get account info after multiple attempts: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error getting account info: {e}", exc_info=True)
            return {"error": f"Unexpected error: {str(e)}"}

    # Apply retry policy decorator
    @retry_with_policy(
        # Use the max_attempts from config if possible, otherwise default
        # Since self isn't available directly, we use the default from retry_policy (3)
        # If a different number of attempts is crucial, consider applying the retry
        # logic within the calling methods instead of decorating _make_request.
        # For now, we stick with the default.
        exceptions=[requests.exceptions.RequestException] # Specify exceptions to retry
    )
    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Dict[str, Any] = None,
        json_body: Dict[str, Any] = None # Renamed json to json_body
    ) -> Dict[str, Any]:
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
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}

        # Add API key if available
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Retry logic is handled by the decorator
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=json_body, # Pass the renamed variable here
            timeout=self.timeout
        )

        # Raise exception for 4XX/5XX responses
        # The retry policy will catch retryable exceptions (like connection errors, timeouts, 5xx)
        response.raise_for_status()
        return response.json()
