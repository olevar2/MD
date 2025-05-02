"""
Risk Management Client for Forex Trading Platform

This module provides a client to interact with the risk management service,
allowing the strategy execution engine to validate trades against risk parameters.
"""
import requests
import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple

# Import the centralized retry policy
from common_lib.resilience import retry_with_policy, register_common_retryable_exceptions

logger = logging.getLogger(__name__)

# Register common requests exceptions for retry
# This might already be done in retry_policy.py, but explicit registration here is fine
try:
    register_common_retryable_exceptions([requests.exceptions.RequestException])
except NameError: # Handle case where requests might not be installed when retry_policy is loaded
    logger.warning("requests library not found during retry registration in risk_client.")

class RiskManagementClient:
    """
    Client for interacting with the risk management service
    
    This class provides methods to check trades against risk parameters
    and get risk management information from the risk management service.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the risk management client
        
        Args:
            config: Configuration dictionary for the client
        """
        self.config = config or {}
        self.base_url = self.config.get("risk_service_url", "http://localhost:8003")
        self.timeout = self.config.get("timeout_seconds", 5)
        self.retries = self.config.get("max_retries", 3)
        
        # Optional API key for authentication
        self.api_key = self.config.get("api_key")
        
        logger.info(f"Risk Management Client initialized with base URL: {self.base_url}")
    
    @retry_with_policy(
        max_attempts=4, # Corresponds to initial attempt + 3 retries
        # base_delay=1.0, # Default
        # backoff_factor=2.0, # Default
        # jitter=True, # Default
        exceptions=[requests.exceptions.RequestException] # Specify exceptions to retry
    )
    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Dict[str, Any] = None,
        json_body: Dict[str, Any] = None # Renamed json to json_body to avoid shadowing built-in
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the risk management service with retry logic.

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

    def check_risk(
        self,
        symbol: str,
        direction: str,
        size: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        account_balance: Optional[float] = None,
        strategy_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check if a trade complies with risk management rules
        
        Args:
            symbol: Trading instrument symbol
            direction: Trade direction ('buy' or 'sell')
            size: Position size in lots
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            account_balance: Current account balance
            strategy_id: Strategy identifier
            
        Returns:
            Dictionary with risk check result
        """
        endpoint = "/api/v1/risk/check"
        
        data = {
            "symbol": symbol,
            "direction": direction,
            "size": size,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "account_balance": account_balance,
            "strategy_id": strategy_id
        }
        # Filter out None values before sending
        payload = {k: v for k, v in data.items() if v is not None}

        try:
            # Pass json_body correctly
            response = self._make_request("POST", endpoint, json_body=payload)
            return response
        except Exception as e: # Catch broader exceptions including RetryExhaustedException
            logger.error(f"Risk check failed: {e}")
            return {
                "is_valid": False,
                "reason": f"Risk service error: {str(e)}",
                "risk_percentage": None,
                "max_loss": None
            }
    
    def get_risk_limits(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current risk limits from the risk management service
        
        Args:
            strategy_id: Optional strategy identifier for strategy-specific limits
            
        Returns:
            Dictionary with risk limits
        """
        endpoint = "/api/v1/risk/limits"
        params = {}
        if strategy_id:
            params["strategy_id"] = strategy_id

        try:
            response = self._make_request("GET", endpoint, params=params)
            return response
        except Exception as e:
            logger.error(f"Failed to get risk limits: {e}")
            return {"error": str(e)}

    def get_portfolio_risk(self) -> Dict[str, Any]:
        """
        Get current portfolio risk metrics
        
        Returns:
            Dictionary with portfolio risk metrics
        """
        endpoint = "/api/v1/risk/portfolio"
        
        try:
            response = self._make_request("GET", endpoint)
            return response
        except Exception as e:
            logger.error(f"Failed to get portfolio risk: {e}")
            return {"error": str(e)}

    def report_trade_outcome(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        size: float,
        profit_loss: float,
        strategy_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Report trade outcome to the risk management service
        
        Args:
            trade_id: Unique trade identifier
            symbol: Trading instrument symbol
            direction: Trade direction ('buy' or 'sell')
            entry_price: Entry price
            exit_price: Exit price
            size: Position size in lots
            profit_loss: Profit/loss amount
            strategy_id: Strategy identifier
            
        Returns:
            Dictionary with response from risk service
        """
        endpoint = "/api/v1/risk/trade-outcome"
        
        data = {
            "trade_id": trade_id,
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "profit_loss": profit_loss
        }
        payload = {k: v for k, v in data.items() if v is not None}

        try:
            # Pass json_body correctly
            response = self._make_request("POST", endpoint, json_body=payload)
            return response
        except Exception as e:
            logger.error(f"Failed to report trade outcome: {e}")
            return {"error": str(e)}
