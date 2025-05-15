"""
Interface for the Backtesting Service.

This module defines the interface for the Backtesting Service, which provides
backtesting capabilities for trading strategies, including strategy backtesting,
walk-forward optimization, Monte Carlo simulation, and stress testing.
"""
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from datetime import datetime


class IBacktestingService(ABC):
    """Interface for the Backtesting Service."""

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def get_available_strategies(self,
                                      correlation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get a list of available strategies for backtesting.

        Args:
            correlation_id: Optional correlation ID for request tracing

        Returns:
            A list of dictionaries containing strategy information
        """
        pass

    @abstractmethod
    async def get_available_data_sources(self,
                                        correlation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get a list of available data sources for backtesting.

        Args:
            correlation_id: Optional correlation ID for request tracing

        Returns:
            A list of dictionaries containing data source information
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass