"""
Portfolio Management Service Client.

This module provides a client for interacting with the Portfolio Management Service.
"""

import aiohttp
import os
from typing import Dict, Any, Optional
from urllib.parse import urljoin

from core_foundations.utils.logger import get_logger
from common_lib.resilience import retry_with_policy, register_common_retryable_exceptions

logger = get_logger("portfolio-management-client")


class PortfolioManagementClient:
    """Client for interacting with the Portfolio Management Service."""

    def __init__(self, base_url: str = None):
        """
        Initialize the Portfolio Management Service client.
        
        Args:
            base_url: Base URL for the Portfolio Management Service API
        """
        self.base_url = base_url or os.environ.get('PORTFOLIO_MANAGEMENT_URL', 
                                                  "http://portfolio-management-service:8006")
        self.api_base = urljoin(self.base_url, "/api/v1/multi-asset/")
        self.session = None
        self.api_key = os.environ.get('PORTFOLIO_MANAGEMENT_API_KEY', '')

    async def _ensure_session(self):
        """Ensure that an aiohttp session exists."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()

    @retry_with_policy()
    async def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, data: Optional[Dict] = None) -> Dict:
        """Make an HTTP request to the Portfolio Management Service."""
        await self._ensure_session()
        
        url = urljoin(self.api_base, endpoint)
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        
        async with self.session.request(method, url, params=params, json=data, headers=headers) as response:
            response.raise_for_status()
            return await response.json()

    async def get_portfolio_summary(self, account_id: str) -> Dict[str, Any]:
        """
        Get portfolio summary from Portfolio Management Service.
        
        Args:
            account_id: Account ID to get summary for
            
        Returns:
            Portfolio summary including positions and exposures
        """
        endpoint = f"portfolio/{account_id}/summary"
        return await self._make_request("GET", endpoint)

    async def get_portfolio_risk(self, account_id: str) -> Dict[str, Any]:
        """
        Get unified risk metrics from Portfolio Management Service.
        
        Args:
            account_id: Account ID to get risk metrics for
            
        Returns:
            Risk metrics including exposures and risk levels
        """
        endpoint = f"portfolio/{account_id}/risk"
        return await self._make_request("GET", endpoint)

    async def get_total_exposure(self, account_id: str) -> float:
        """
        Get total exposure for an account.
        
        Args:
            account_id: Account ID to get exposure for
            
        Returns:
            Total exposure value
        """
        summary = await self.get_portfolio_summary(account_id)
        return float(summary.get("total_exposure", 0.0))

    async def get_symbol_exposure(self, account_id: str, symbol: str) -> float:
        """
        Get exposure for a specific symbol.
        
        Args:
            account_id: Account ID
            symbol: Symbol to get exposure for
            
        Returns:
            Symbol exposure value
        """
        summary = await self.get_portfolio_summary(account_id)
        positions = summary.get("positions", {})
        
        # Find position matching the symbol
        for position in positions:
            if position.get("symbol") == symbol:
                return float(position.get("current_value", 0.0))
        
        return 0.0
