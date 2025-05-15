"""
Adapter for the Backtesting Service.

This module provides an adapter for the Backtesting Service, implementing
the IBacktestingService interface.
"""
from typing import Dict, Any, List, Optional
import logging
import httpx
from datetime import datetime
from common_lib.interfaces.backtesting_service_interface import IBacktestingService
from common_lib.resilience.decorators import with_circuit_breaker, with_retry, with_timeout
from common_lib.resilience.factory import create_standard_resilience_config

logger = logging.getLogger(__name__)


class BacktestingAdapter(IBacktestingService):
    """Adapter for the Backtesting Service."""

    def __init__(self, base_url: str, timeout: float = 60.0):
        """
        Initialize the BacktestingAdapter.

        Args:
            base_url: The base URL of the Backtesting Service
            timeout: The timeout for HTTP requests in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.resilience_config = create_standard_resilience_config(
            service_name="backtesting-service",
            timeout_seconds=timeout
        )
        logger.info(f"Initialized BacktestingAdapter with base URL: {base_url}")

    @with_circuit_breaker("backtesting-service")
    @with_retry("backtesting-service")
    @with_timeout("backtesting-service")
    async def run_backtest(self, 
                          config: Dict[str, Any],
                          correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a backtest for a trading strategy with the specified configuration.

        Args:
            config: The backtest configuration
            correlation_id: Optional correlation ID for request tracing

        Returns:
            A dictionary containing the backtest results
        """
        url = f"{self.base_url}/api/v1/run"
        headers = {"X-Correlation-ID": correlation_id} if correlation_id else {}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=config, headers=headers)
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("backtesting-service")
    @with_retry("backtesting-service")
    @with_timeout("backtesting-service")
    async def get_backtest_results(self, 
                                  backtest_id: str,
                                  include_trades: bool = True,
                                  include_equity_curve: bool = True,
                                  correlation_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the results of a previously run backtest.

        Args:
            backtest_id: The ID of the backtest
            include_trades: Whether to include trade details in the response
            include_equity_curve: Whether to include the equity curve in the response
            correlation_id: Optional correlation ID for request tracing

        Returns:
            A dictionary containing the backtest results, or None if not found
        """
        url = f"{self.base_url}/api/v1/{backtest_id}"
        params = {
            "include_trades": str(include_trades).lower(),
            "include_equity_curve": str(include_equity_curve).lower()
        }
        headers = {"X-Correlation-ID": correlation_id} if correlation_id else {}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, params=params, headers=headers)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("backtesting-service")
    @with_retry("backtesting-service")
    @with_timeout("backtesting-service")
    async def run_walk_forward_optimization(self, 
                                          config: Dict[str, Any],
                                          correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run walk-forward optimization for a trading strategy.

        Args:
            config: The walk-forward optimization configuration
            correlation_id: Optional correlation ID for request tracing

        Returns:
            A dictionary containing the walk-forward optimization results
        """
        url = f"{self.base_url}/api/v1/walk-forward"
        headers = {"X-Correlation-ID": correlation_id} if correlation_id else {}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=config, headers=headers)
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("backtesting-service")
    @with_retry("backtesting-service")
    @with_timeout("backtesting-service")
    async def run_monte_carlo_simulation(self, 
                                        config: Dict[str, Any],
                                        correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for a previously executed backtest.

        Args:
            config: The Monte Carlo simulation configuration
            correlation_id: Optional correlation ID for request tracing

        Returns:
            A dictionary containing the Monte Carlo simulation results
        """
        url = f"{self.base_url}/api/v1/monte-carlo"
        headers = {"X-Correlation-ID": correlation_id} if correlation_id else {}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=config, headers=headers)
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("backtesting-service")
    @with_retry("backtesting-service")
    @with_timeout("backtesting-service")
    async def run_stress_test(self, 
                             config: Dict[str, Any],
                             correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run stress test for a previously executed backtest.

        Args:
            config: The stress test configuration
            correlation_id: Optional correlation ID for request tracing

        Returns:
            A dictionary containing the stress test results
        """
        url = f"{self.base_url}/api/v1/stress-test"
        headers = {"X-Correlation-ID": correlation_id} if correlation_id else {}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=config, headers=headers)
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("backtesting-service")
    @with_retry("backtesting-service")
    @with_timeout("backtesting-service")
    async def get_available_strategies(self,
                                      correlation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get a list of available strategies for backtesting.

        Args:
            correlation_id: Optional correlation ID for request tracing

        Returns:
            A list of dictionaries containing strategy information
        """
        url = f"{self.base_url}/api/v1/strategies"
        headers = {"X-Correlation-ID": correlation_id} if correlation_id else {}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("backtesting-service")
    @with_retry("backtesting-service")
    @with_timeout("backtesting-service")
    async def get_available_data_sources(self,
                                        correlation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get a list of available data sources for backtesting.

        Args:
            correlation_id: Optional correlation ID for request tracing

        Returns:
            A list of dictionaries containing data source information
        """
        url = f"{self.base_url}/api/v1/data-sources"
        headers = {"X-Correlation-ID": correlation_id} if correlation_id else {}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("backtesting-service")
    @with_retry("backtesting-service")
    @with_timeout("backtesting-service")
    async def cancel_backtest(self,
                             backtest_id: str,
                             correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancel a running backtest.

        Args:
            backtest_id: The ID of the backtest to cancel
            correlation_id: Optional correlation ID for request tracing

        Returns:
            A dictionary containing the cancellation result
        """
        url = f"{self.base_url}/api/v1/{backtest_id}/cancel"
        headers = {"X-Correlation-ID": correlation_id} if correlation_id else {}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, headers=headers)
            response.raise_for_status()
            return response.json()

    @with_circuit_breaker("backtesting-service")
    @with_retry("backtesting-service")
    @with_timeout("backtesting-service")
    async def list_backtests(self,
                            status: Optional[str] = None,
                            strategy_id: Optional[str] = None,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            limit: int = 100,
                            offset: int = 0,
                            correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        List backtests with optional filtering.

        Args:
            status: Optional status filter
            strategy_id: Optional strategy ID filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum number of results to return
            offset: Offset for pagination
            correlation_id: Optional correlation ID for request tracing

        Returns:
            A dictionary containing the list of backtests and pagination information
        """
        url = f"{self.base_url}/api/v1/list"
        params = {
            "limit": limit,
            "offset": offset
        }
        
        if status:
            params["status"] = status
        if strategy_id:
            params["strategy_id"] = strategy_id
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
            
        headers = {"X-Correlation-ID": correlation_id} if correlation_id else {}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()