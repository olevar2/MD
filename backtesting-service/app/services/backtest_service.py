"""
Backtest Service

This module provides the service layer for backtesting.
"""
import logging
import uuid
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from app.core.engine.backtest_engine import BacktestEngine
from app.repositories.backtest_repository import BacktestRepository
from app.models.backtest_models import (
    BacktestRequest,
    BacktestResponse,
    BacktestResult,
    BacktestStatus,
    OptimizationRequest,
    OptimizationResponse,
    OptimizationResult,
    WalkForwardTestRequest,
    WalkForwardTestResponse,
    WalkForwardTestResult,
    StrategyMetadata,
    StrategyListResponse
)

logger = logging.getLogger(__name__)

class BacktestService:
    """
    Service for backtesting.
    """
    def __init__(self, repository: Optional[BacktestRepository] = None, data_client=None):
        """
        Initialize the backtest service.

        Args:
            repository: Repository for storing and retrieving backtest results
            data_client: Client for retrieving market data
        """
        self.repository = repository or BacktestRepository()
        self.data_client = data_client
        self.backtest_engine = BacktestEngine()
        self.strategies = self._load_strategies()

    async def run_backtest(self, request: BacktestRequest) -> BacktestResponse:
        """
        Run a backtest for a trading strategy.

        Args:
            request: Backtest request

        Returns:
            BacktestResponse: Response with backtest ID and status
        """
        # Generate a unique ID for the backtest
        backtest_id = str(uuid.uuid4())

        # Create initial backtest record
        await self.repository.save_backtest(backtest_id, {
            'backtest_id': backtest_id,
            'strategy_id': request.strategy_id,
            'symbol': request.symbol,
            'timeframe': request.timeframe,
            'start_date': request.start_date.isoformat(),
            'end_date': request.end_date.isoformat(),
            'initial_balance': request.initial_balance,
            'parameters': request.parameters or {},
            'status': BacktestStatus.PENDING,
            'message': 'Backtest pending',
            'created_at': datetime.now().isoformat()
        })

        # Create response
        response = BacktestResponse(
            backtest_id=backtest_id,
            status=BacktestStatus.PENDING,
            message='Backtest pending',
            estimated_completion_time=datetime.now()
        )

        # Run backtest asynchronously
        # In a real implementation, this would be done in a background task
        # For simplicity, we'll run it synchronously here
        try:
            # Update status to running
            await self.repository.update_backtest_status(
                backtest_id,
                BacktestStatus.RUNNING,
                'Backtest running'
            )

            # Fetch market data
            data = await self._fetch_market_data(
                symbol=request.symbol,
                timeframe=request.timeframe,
                start_date=request.start_date,
                end_date=request.end_date
            )

            # Get strategy
            strategy = self._get_strategy(request.strategy_id)

            if not strategy:
                raise ValueError(f"Strategy {request.strategy_id} not found")

            # Run backtest
            parameters = request.parameters or {}
            parameters['backtest_id'] = backtest_id
            parameters['strategy_id'] = request.strategy_id

            backtest_result = await self.backtest_engine.run_backtest(
                strategy=strategy,
                data=data,
                parameters=parameters
            )

            # Save backtest result
            await self.repository.save_backtest(backtest_id, backtest_result.model_dump())

            # Update status to completed
            await self.repository.update_backtest_status(
                backtest_id,
                BacktestStatus.COMPLETED,
                'Backtest completed successfully'
            )

            logger.info(f"Backtest {backtest_id} completed successfully")
        except Exception as e:
            # Update status to failed
            await self.repository.update_backtest_status(
                backtest_id,
                BacktestStatus.FAILED,
                f"Backtest failed: {str(e)}"
            )

            logger.error(f"Backtest {backtest_id} failed: {e}")

        return response

    async def get_backtest_result(self, backtest_id: str) -> Optional[BacktestResult]:
        """
        Get the result of a backtest.

        Args:
            backtest_id: ID of the backtest

        Returns:
            BacktestResult: Backtest result or None if not found
        """
        # Get backtest from repository
        backtest_data = await self.repository.get_backtest(backtest_id)

        if not backtest_data:
            return None

        # Convert to BacktestResult
        return BacktestResult(**backtest_data)

    async def list_backtests(self, strategy_id: Optional[str] = None,
                           symbol: Optional[str] = None,
                           limit: int = 100,
                           offset: int = 0) -> List[Dict[str, Any]]:
        """
        List backtests with optional filtering.

        Args:
            strategy_id: Optional strategy ID to filter by
            symbol: Optional symbol to filter by
            limit: Maximum number of results to return
            offset: Offset for pagination

        Returns:
            List of backtests
        """
        return await self.repository.list_backtests(
            strategy_id=strategy_id,
            symbol=symbol,
            limit=limit,
            offset=offset
        )

    async def optimize_strategy(self, request: OptimizationRequest) -> OptimizationResponse:
        """
        Optimize a trading strategy.

        Args:
            request: Optimization request

        Returns:
            OptimizationResponse: Response with optimization ID and status
        """
        # Generate a unique ID for the optimization
        optimization_id = str(uuid.uuid4())

        # Create initial optimization record
        await self.repository.save_optimization(optimization_id, {
            'optimization_id': optimization_id,
            'strategy_id': request.strategy_id,
            'symbol': request.symbol,
            'timeframe': request.timeframe,
            'start_date': request.start_date.isoformat(),
            'end_date': request.end_date.isoformat(),
            'initial_balance': request.initial_balance,
            'parameters_to_optimize': request.parameters_to_optimize,
            'optimization_metric': request.optimization_metric,
            'optimization_method': request.optimization_method,
            'max_evaluations': request.max_evaluations,
            'status': 'pending',
            'message': 'Optimization pending',
            'created_at': datetime.now().isoformat()
        })

        # Create response
        response = OptimizationResponse(
            optimization_id=optimization_id,
            status='pending',
            message='Optimization pending',
            estimated_completion_time=datetime.now()
        )

        # Run optimization asynchronously
        # In a real implementation, this would be done in a background task
        # For simplicity, we'll run it synchronously here
        try:
            # Update status to running
            await self._update_optimization_status(
                optimization_id,
                'running',
                'Optimization running'
            )

            # Fetch market data
            data = await self._fetch_market_data(
                symbol=request.symbol,
                timeframe=request.timeframe,
                start_date=request.start_date,
                end_date=request.end_date
            )

            # Get strategy
            strategy = self._get_strategy(request.strategy_id)

            if not strategy:
                raise ValueError(f"Strategy {request.strategy_id} not found")

            # Run optimization
            # This is a placeholder for actual optimization
            # In a real implementation, this would use a proper optimization algorithm
            best_parameters = {}
            best_metric_value = 0.0
            all_results = []

            # Update status to completed
            await self._update_optimization_status(
                optimization_id,
                'completed',
                'Optimization completed successfully'
            )

            # Save optimization result
            optimization_result = OptimizationResult(
                optimization_id=optimization_id,
                strategy_id=request.strategy_id,
                symbol=request.symbol,
                timeframe=request.timeframe,
                start_date=request.start_date,
                end_date=request.end_date,
                optimization_metric=request.optimization_metric,
                optimization_method=request.optimization_method,
                best_parameters=best_parameters,
                best_metric_value=best_metric_value,
                evaluations=len(all_results),
                all_results=all_results
            )

            await self.repository.save_optimization(optimization_id, optimization_result.model_dump())

            logger.info(f"Optimization {optimization_id} completed successfully")
        except Exception as e:
            # Update status to failed
            await self._update_optimization_status(
                optimization_id,
                'failed',
                f"Optimization failed: {str(e)}"
            )

            logger.error(f"Optimization {optimization_id} failed: {e}")

        return response

    async def get_optimization_result(self, optimization_id: str) -> Optional[OptimizationResult]:
        """
        Get the result of an optimization.

        Args:
            optimization_id: ID of the optimization

        Returns:
            OptimizationResult: Optimization result or None if not found
        """
        # Get optimization from repository
        optimization_data = await self.repository.get_optimization(optimization_id)

        if not optimization_data:
            return None

        # Convert to OptimizationResult
        return OptimizationResult(**optimization_data)

    async def run_walk_forward_test(self, request: WalkForwardTestRequest) -> WalkForwardTestResponse:
        """
        Run a walk-forward test for a trading strategy.

        Args:
            request: Walk-forward test request

        Returns:
            WalkForwardTestResponse: Response with test ID and status
        """
        # Generate a unique ID for the test
        test_id = str(uuid.uuid4())

        # Create initial test record
        await self.repository.save_walk_forward_test(test_id, {
            'test_id': test_id,
            'strategy_id': request.strategy_id,
            'symbol': request.symbol,
            'timeframe': request.timeframe,
            'start_date': request.start_date.isoformat(),
            'end_date': request.end_date.isoformat(),
            'initial_balance': request.initial_balance,
            'parameters': request.parameters or {},
            'optimization_window': request.optimization_window,
            'test_window': request.test_window,
            'optimization_metric': request.optimization_metric,
            'parameters_to_optimize': request.parameters_to_optimize,
            'status': 'pending',
            'message': 'Walk-forward test pending',
            'created_at': datetime.now().isoformat()
        })

        # Create response
        response = WalkForwardTestResponse(
            test_id=test_id,
            status='pending',
            message='Walk-forward test pending',
            estimated_completion_time=datetime.now()
        )

        # Run walk-forward test asynchronously
        # In a real implementation, this would be done in a background task
        # For simplicity, we'll run it synchronously here
        try:
            # Update status to running
            await self._update_walk_forward_test_status(
                test_id,
                'running',
                'Walk-forward test running'
            )

            # Fetch market data
            data = await self._fetch_market_data(
                symbol=request.symbol,
                timeframe=request.timeframe,
                start_date=request.start_date,
                end_date=request.end_date
            )

            # Get strategy
            strategy = self._get_strategy(request.strategy_id)

            if not strategy:
                raise ValueError(f"Strategy {request.strategy_id} not found")

            # Run walk-forward test
            # This is a placeholder for actual walk-forward testing
            # In a real implementation, this would use a proper walk-forward testing algorithm
            windows = []
            parameters_by_window = {}
            trades = []
            equity_curve = []

            # Update status to completed
            await self._update_walk_forward_test_status(
                test_id,
                'completed',
                'Walk-forward test completed successfully'
            )

            # Save walk-forward test result
            test_result = WalkForwardTestResult(
                test_id=test_id,
                strategy_id=request.strategy_id,
                symbol=request.symbol,
                timeframe=request.timeframe,
                start_date=request.start_date,
                end_date=request.end_date,
                initial_balance=request.initial_balance,
                final_balance=request.initial_balance,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                performance_metrics={
                    'total_return': 0.0,
                    'annualized_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'average_trade': 0.0,
                    'average_winning_trade': 0.0,
                    'average_losing_trade': 0.0
                },
                trades=trades,
                equity_curve=equity_curve,
                windows=windows,
                parameters_by_window=parameters_by_window
            )

            await self.repository.save_walk_forward_test(test_id, test_result.model_dump())

            logger.info(f"Walk-forward test {test_id} completed successfully")
        except Exception as e:
            # Update status to failed
            await self._update_walk_forward_test_status(
                test_id,
                'failed',
                f"Walk-forward test failed: {str(e)}"
            )

            logger.error(f"Walk-forward test {test_id} failed: {e}")

        return response

    async def get_walk_forward_test_result(self, test_id: str) -> Optional[WalkForwardTestResult]:
        """
        Get the result of a walk-forward test.

        Args:
            test_id: ID of the walk-forward test

        Returns:
            WalkForwardTestResult: Walk-forward test result or None if not found
        """
        # Get walk-forward test from repository
        test_data = await self.repository.get_walk_forward_test(test_id)

        if not test_data:
            return None

        # Convert to WalkForwardTestResult
        return WalkForwardTestResult(**test_data)

    async def list_strategies(self) -> StrategyListResponse:
        """
        List available strategies.

        Returns:
            StrategyListResponse: List of available strategies
        """
        return StrategyListResponse(
            strategies=list(self.strategies.values()),
            count=len(self.strategies)
        )

    async def _fetch_market_data(self, symbol: str, timeframe: str,
                               start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch market data for a symbol.

        Args:
            symbol: The currency pair or symbol
            timeframe: The timeframe for the data
            start_date: The start date
            end_date: The end date

        Returns:
            DataFrame containing market data
        """
        if self.data_client:
            # Use data client to fetch data
            data = await self.data_client.get_market_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            return data
        else:
            # Generate mock data for testing
            return self._generate_mock_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )

    def _generate_mock_data(self, symbol: str, timeframe: str,
                          start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Generate mock market data for testing.

        Args:
            symbol: The currency pair or symbol
            timeframe: The timeframe for the data
            start_date: The start date
            end_date: The end date

        Returns:
            DataFrame containing mock market data
        """
        import numpy as np

        # Generate date range
        if timeframe == '1d':
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        elif timeframe == '1h':
            date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        elif timeframe == '4h':
            date_range = pd.date_range(start=start_date, end=end_date, freq='4H')
        else:
            # Default to daily
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Generate price data
        n = len(date_range)
        close = np.random.normal(loc=100, scale=1, size=n).cumsum() + 1000

        # Generate OHLCV data
        high = close * (1 + np.random.uniform(0, 0.01, size=n))
        low = close * (1 - np.random.uniform(0, 0.01, size=n))
        open_price = low + np.random.uniform(0, 1, size=n) * (high - low)
        volume = np.random.uniform(1000, 5000, size=n)

        # Create DataFrame
        data = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=date_range)

        return data

    def _load_strategies(self) -> Dict[str, StrategyMetadata]:
        """
        Load available strategies.

        Returns:
            Dict[str, StrategyMetadata]: Dictionary of available strategies
        """
        # This is a placeholder for loading strategies from a database or file
        # In a real implementation, this would load strategies from a database or file

        strategies = {}

        # Add some example strategies
        strategies['moving_average_crossover'] = StrategyMetadata(
            strategy_id='moving_average_crossover',
            name='Moving Average Crossover',
            description='A simple moving average crossover strategy',
            version='1.0.0',
            author='Forex Trading Platform',
            created_at=datetime.now(),
            updated_at=datetime.now(),
            parameters={
                'short_window': 10,
                'long_window': 50
            },
            supported_symbols=['EURUSD', 'GBPUSD', 'USDJPY'],
            supported_timeframes=['1h', '4h', '1d']
        )

        strategies['rsi_strategy'] = StrategyMetadata(
            strategy_id='rsi_strategy',
            name='RSI Strategy',
            description='A strategy based on the Relative Strength Index',
            version='1.0.0',
            author='Forex Trading Platform',
            created_at=datetime.now(),
            updated_at=datetime.now(),
            parameters={
                'rsi_period': 14,
                'overbought_threshold': 70,
                'oversold_threshold': 30
            },
            supported_symbols=['EURUSD', 'GBPUSD', 'USDJPY'],
            supported_timeframes=['1h', '4h', '1d']
        )

        return strategies

    def _get_strategy(self, strategy_id: str) -> Optional[Any]:
        """
        Get a strategy by ID.

        Args:
            strategy_id: ID of the strategy

        Returns:
            Strategy or None if not found
        """
        # This is a placeholder for getting a strategy
        # In a real implementation, this would return an actual strategy object

        if strategy_id not in self.strategies:
            return None

        # Return a mock strategy object
        class MockStrategy:
            def __init__(self, metadata):
                self.metadata = metadata

        return MockStrategy(self.strategies[strategy_id])

    async def _update_optimization_status(self, optimization_id: str, status: str, message: Optional[str] = None) -> None:
        """
        Update the status of an optimization.

        Args:
            optimization_id: ID of the optimization
            status: New status
            message: Optional message
        """
        # Get optimization from repository
        optimization = await self.repository.get_optimization(optimization_id)

        if optimization:
            # Update status
            optimization['status'] = status

            # Update message if provided
            if message:
                optimization['message'] = message

            # Save updated optimization
            await self.repository.save_optimization(optimization_id, optimization)

            logger.info(f"Updated optimization {optimization_id} status to {status}")
        else:
            logger.warning(f"Optimization {optimization_id} not found for status update")

    async def _update_walk_forward_test_status(self, test_id: str, status: str, message: Optional[str] = None) -> None:
        """
        Update the status of a walk-forward test.

        Args:
            test_id: ID of the walk-forward test
            status: New status
            message: Optional message
        """
        # Get walk-forward test from repository
        test = await self.repository.get_walk_forward_test(test_id)

        if test:
            # Update status
            test['status'] = status

            # Update message if provided
            if message:
                test['message'] = message

            # Save updated walk-forward test
            await self.repository.save_walk_forward_test(test_id, test)

            logger.info(f"Updated walk-forward test {test_id} status to {status}")
        else:
            logger.warning(f"Walk-forward test {test_id} not found for status update")
