"""
Standardized Backtesting Client

This module provides a client for interacting with the standardized Backtesting API.
"""
import logging
import aiohttp
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from analysis_engine.core.config import get_settings
from analysis_engine.core.resilience import retry_with_backoff, circuit_breaker
from analysis_engine.monitoring.structured_logging import get_structured_logger
from analysis_engine.core.exceptions_bridge import ServiceUnavailableError, ServiceTimeoutError
logger = get_structured_logger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class BacktestingClient:
    """
    Client for interacting with the standardized Backtesting API.

    This client provides methods for accessing backtesting capabilities,
    including strategy backtesting, walk-forward optimization, Monte Carlo simulation,
    and stress testing.

    It includes resilience patterns like retry with backoff and circuit breaking.
    """

    def __init__(self, base_url: Optional[str]=None, timeout: int=60):
        """
        Initialize the Backtesting client.

        Args:
            base_url: Base URL for the Backtesting API. If None, uses the URL from settings.
            timeout: Request timeout in seconds.
        """
        settings = get_settings()
        self.base_url = base_url or settings.analysis_engine_url
        self.timeout = timeout
        self.api_prefix = '/api/v1/analysis/backtesting'
        self.circuit_breaker = circuit_breaker(failure_threshold=5,
            recovery_timeout=30, name='backtesting_client')
        logger.info(
            f'Initialized Backtesting client with base URL: {self.base_url}')

    @async_with_exception_handling
    async def _make_request(self, method: str, endpoint: str, data:
        Optional[Dict]=None, params: Optional[Dict]=None) ->Dict:
        """
        Make a request to the Backtesting API with resilience patterns.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request data
            params: Query parameters

        Returns:
            Response data

        Raises:
            ServiceUnavailableError: If the service is unavailable
            ServiceTimeoutError: If the request times out
            Exception: For other errors
        """
        url = f'{self.base_url}{self.api_prefix}{endpoint}'

        @retry_with_backoff(max_retries=3, backoff_factor=1.5,
            retry_exceptions=[aiohttp.ClientError, TimeoutError])
        @self.circuit_breaker
        @async_with_exception_handling
        async def _request():
    """
     request.
    
    """

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(method=method, url=url, json
                        =data, params=params, timeout=self.timeout
                        ) as response:
                        if response.status >= 500:
                            error_text = await response.text()
                            logger.error(
                                f'Server error from Backtesting API: {error_text}'
                                )
                            raise ServiceUnavailableError(
                                f'Backtesting API server error: {response.status}'
                                )
                        if response.status >= 400:
                            error_text = await response.text()
                            logger.error(
                                f'Client error from Backtesting API: {error_text}'
                                )
                            raise Exception(
                                f'Backtesting API client error: {response.status} - {error_text}'
                                )
                        return await response.json()
            except aiohttp.ClientError as e:
                logger.error(f'Connection error to Backtesting API: {str(e)}')
                raise ServiceUnavailableError(
                    f'Failed to connect to Backtesting API: {str(e)}')
            except asyncio.TimeoutError:
                logger.error(f'Timeout connecting to Backtesting API')
                raise ServiceTimeoutError(
                    f'Timeout connecting to Backtesting API')
        return await _request()

    async def run_backtest(self, strategy_config: Dict[str, Any],
        start_date: Union[datetime, str], end_date: Union[datetime, str],
        instruments: List[str], initial_capital: float=10000.0,
        commission_model: str='fixed', commission_settings: Optional[Dict[
        str, Any]]=None, slippage_model: str='fixed', slippage_settings:
        Optional[Dict[str, Any]]=None, data_source: str='historical',
        data_parameters: Optional[Dict[str, Any]]=None) ->Dict:
        """
        Run a backtest for a trading strategy with the specified configuration.

        Args:
            strategy_config: Strategy configuration
            start_date: Start date for backtesting
            end_date: End date for backtesting
            instruments: List of instruments to backtest
            initial_capital: Initial capital for backtesting
            commission_model: Commission model (fixed, percentage)
            commission_settings: Commission settings
            slippage_model: Slippage model (fixed, percentage, variable)
            slippage_settings: Slippage settings
            data_source: Data source (historical, generated)
            data_parameters: Data source parameters

        Returns:
            Backtest results
        """
        if isinstance(start_date, datetime):
            start_date = start_date.isoformat()
        if isinstance(end_date, datetime):
            end_date = end_date.isoformat()
        request_data = {'strategy_config': strategy_config, 'start_date':
            start_date, 'end_date': end_date, 'instruments': instruments,
            'initial_capital': initial_capital, 'commission_model':
            commission_model, 'commission_settings': commission_settings or
            {}, 'slippage_model': slippage_model, 'slippage_settings': 
            slippage_settings or {}, 'data_source': data_source,
            'data_parameters': data_parameters or {}}
        logger.info(
            f"Running backtest for strategy {strategy_config_manager.get('strategy_id')}"
            )
        return await self._make_request('POST', '/run', data=request_data)

    @with_resilience('get_backtest_results')
    async def get_backtest_results(self, backtest_id: str, include_trades:
        bool=True, include_equity_curve: bool=True) ->Dict:
        """
        Get the results of a previously run backtest.

        Args:
            backtest_id: ID of the backtest to retrieve
            include_trades: Include trade details in response
            include_equity_curve: Include equity curve in response

        Returns:
            Backtest results
        """
        params = {'include_trades': include_trades, 'include_equity_curve':
            include_equity_curve}
        logger.info(f'Getting backtest results for {backtest_id}')
        return await self._make_request('GET', f'/{backtest_id}', params=params
            )

    async def run_walk_forward_optimization(self, strategy_config: Dict[str,
        Any], parameter_ranges: Dict[str, List[Any]], start_date: Union[
        datetime, str], end_date: Union[datetime, str], instruments: List[
        str], initial_capital: float=10000.0, window_size_days: int=90,
        anchor_size_days: int=30, optimization_metric: str='sharpe_ratio',
        parallel_jobs: int=1) ->Dict:
        """
        Run walk-forward optimization for a trading strategy.

        Args:
            strategy_config: Strategy configuration
            parameter_ranges: Parameter ranges for optimization
            start_date: Start date for backtesting
            end_date: End date for backtesting
            instruments: List of instruments to backtest
            initial_capital: Initial capital for backtesting
            window_size_days: Size of each window in days
            anchor_size_days: Size of each anchor (out-of-sample) period in days
            optimization_metric: Metric to optimize for
            parallel_jobs: Number of parallel jobs for optimization

        Returns:
            Walk-forward optimization results
        """
        if isinstance(start_date, datetime):
            start_date = start_date.isoformat()
        if isinstance(end_date, datetime):
            end_date = end_date.isoformat()
        request_data = {'strategy_config': strategy_config,
            'parameter_ranges': parameter_ranges, 'start_date': start_date,
            'end_date': end_date, 'instruments': instruments,
            'initial_capital': initial_capital, 'window_size_days':
            window_size_days, 'anchor_size_days': anchor_size_days,
            'optimization_metric': optimization_metric, 'parallel_jobs':
            parallel_jobs}
        logger.info(
            f"Running walk-forward optimization for strategy {strategy_config_manager.get('strategy_id')}"
            )
        return await self._make_request('POST', '/walk-forward', data=
            request_data)

    async def run_monte_carlo_simulation(self, backtest_id: str,
        num_simulations: int=1000, confidence_level: float=0.95,
        simulation_method: str='bootstrap', simulation_parameters: Optional
        [Dict[str, Any]]=None) ->Dict:
        """
        Run Monte Carlo simulation for a previously executed backtest.

        Args:
            backtest_id: ID of the backtest to simulate
            num_simulations: Number of Monte Carlo simulations to run
            confidence_level: Confidence level for results (0-1)
            simulation_method: Simulation method (bootstrap, parametric)
            simulation_parameters: Additional simulation parameters

        Returns:
            Monte Carlo simulation results
        """
        request_data = {'backtest_id': backtest_id, 'num_simulations':
            num_simulations, 'confidence_level': confidence_level,
            'simulation_method': simulation_method, 'simulation_parameters':
            simulation_parameters or {}}
        logger.info(
            f'Running Monte Carlo simulation for backtest {backtest_id}')
        return await self._make_request('POST', '/monte-carlo', data=
            request_data)

    async def run_stress_test(self, backtest_id: str, stress_scenarios:
        List[Dict[str, Any]], apply_to_parameters: Optional[List[str]]=None
        ) ->Dict:
        """
        Run stress test for a previously executed backtest.

        Args:
            backtest_id: ID of the backtest to stress test
            stress_scenarios: Stress scenarios to test
            apply_to_parameters: Parameters to apply stress to

        Returns:
            Stress test results
        """
        request_data = {'backtest_id': backtest_id, 'stress_scenarios':
            stress_scenarios, 'apply_to_parameters': apply_to_parameters}
        logger.info(f'Running stress test for backtest {backtest_id}')
        return await self._make_request('POST', '/stress-test', data=
            request_data)

    @with_resilience('get_available_strategies')
    async def get_available_strategies(self) ->List[Dict[str, Any]]:
        """
        Get a list of available strategies for backtesting.

        Returns:
            List of available strategies
        """
        logger.info('Getting available strategies')
        return await self._make_request('GET', '/strategies')

    @with_resilience('get_available_data_sources')
    async def get_available_data_sources(self) ->List[Dict[str, Any]]:
        """
        Get a list of available data sources for backtesting.

        Returns:
            List of available data sources
        """
        logger.info('Getting available data sources')
        return await self._make_request('GET', '/data-sources')
