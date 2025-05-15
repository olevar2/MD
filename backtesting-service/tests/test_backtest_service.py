import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
import pandas as pd
import numpy as np

from app.services.backtest_service import BacktestService
from app.models.backtest_models import (
    BacktestRequest,
    BacktestResponse,
    BacktestStatus,
    BacktestResult,
    OptimizationRequest,
    OptimizationResponse,
    OptimizationResult,
    WalkForwardTestRequest,
    WalkForwardTestResponse,
    WalkForwardTestResult,
    StrategyMetadata,
    StrategyListResponse
)

@pytest.fixture
def mock_repository():
    """Create a mock repository for testing."""
    repository = AsyncMock()

    # Configure the mock methods
    async def mock_save_backtest(backtest_id, data):
        return None

    async def mock_get_backtest(backtest_id):
        return {
            'backtest_id': backtest_id,
            'strategy_id': 'test_strategy_id',
            'start_date': '2023-01-01T00:00:00',
            'end_date': '2023-01-03T00:00:00',
            'initial_balance': 10000.0,
            'final_balance': 10500.0,
            'total_trades': 5,
            'winning_trades': 3,
            'losing_trades': 2,
            'performance_metrics': {
                'total_return': 0.05,
                'annualized_return': 0.2,
                'sharpe_ratio': 1.5,
                'max_drawdown': 0.02,
                'win_rate': 0.6,
                'profit_factor': 2.0,
                'average_trade': 100.0,
                'average_winning_trade': 200.0,
                'average_losing_trade': -50.0
            },
            'trades': [],
            'equity_curve': [],
            'parameters': {}
        }

    async def mock_update_backtest_status(backtest_id, status, message=None):
        return None

    async def mock_list_backtests(strategy_id=None, symbol=None, limit=100, offset=0):
        return [
            {
                'backtest_id': 'test_backtest_id_1',
                'strategy_id': strategy_id or 'test_strategy_id',
                'symbol': 'EURUSD',
                'timeframe': '1h',
                'start_date': '2023-01-01T00:00:00',
                'end_date': '2023-01-03T00:00:00',
                'initial_balance': 10000.0,
                'status': 'completed',
                'created_at': '2023-01-04T00:00:00'
            },
            {
                'backtest_id': 'test_backtest_id_2',
                'strategy_id': strategy_id or 'test_strategy_id',
                'symbol': 'GBPUSD',
                'timeframe': '1h',
                'start_date': '2023-01-01T00:00:00',
                'end_date': '2023-01-03T00:00:00',
                'initial_balance': 10000.0,
                'status': 'completed',
                'created_at': '2023-01-04T00:00:00'
            }
        ]

    # Assign the mock methods
    repository.save_backtest.side_effect = mock_save_backtest
    repository.get_backtest.side_effect = mock_get_backtest
    repository.update_backtest_status.side_effect = mock_update_backtest_status
    repository.list_backtests.side_effect = mock_list_backtests

    # Configure optimization methods
    async def mock_save_optimization(optimization_id, data):
        return None

    async def mock_get_optimization(optimization_id):
        return {
            'optimization_id': optimization_id,
            'strategy_id': 'test_strategy_id',
            'symbol': 'EURUSD',
            'timeframe': '1h',
            'start_date': '2023-01-01T00:00:00',
            'end_date': '2023-01-03T00:00:00',
            'optimization_metric': 'sharpe_ratio',
            'optimization_method': 'grid_search',
            'best_parameters': {'short_window': 10, 'long_window': 50},
            'best_metric_value': 1.5,
            'evaluations': 10,
            'all_results': []
        }

    repository.save_optimization.side_effect = mock_save_optimization
    repository.get_optimization.side_effect = mock_get_optimization

    # Configure walk-forward test methods
    async def mock_save_walk_forward_test(test_id, data):
        return None

    async def mock_get_walk_forward_test(test_id):
        return {
            'test_id': test_id,
            'strategy_id': 'test_strategy_id',
            'symbol': 'EURUSD',
            'timeframe': '1h',
            'start_date': '2023-01-01T00:00:00',
            'end_date': '2023-01-31T00:00:00',
            'initial_balance': 10000.0,
            'final_balance': 11000.0,
            'total_trades': 50,
            'winning_trades': 30,
            'losing_trades': 20,
            'performance_metrics': {
                'total_return': 0.1,
                'annualized_return': 1.2,
                'sharpe_ratio': 1.8,
                'max_drawdown': 0.05,
                'win_rate': 0.6,
                'profit_factor': 2.5,
                'average_trade': 20.0,
                'average_winning_trade': 50.0,
                'average_losing_trade': -25.0
            },
            'trades': [],
            'equity_curve': [],
            'windows': [
                {'start_date': '2023-01-01T00:00:00', 'end_date': '2023-01-10T00:00:00', 'type': 'optimization'},
                {'start_date': '2023-01-11T00:00:00', 'end_date': '2023-01-15T00:00:00', 'type': 'test'}
            ],
            'parameters_by_window': {
                '0': {'short_window': 10, 'long_window': 50}
            }
        }

    repository.save_walk_forward_test.side_effect = mock_save_walk_forward_test
    repository.get_walk_forward_test.side_effect = mock_get_walk_forward_test

    return repository

@pytest.fixture
def mock_data_client():
    """Create a mock data client for testing."""
    data_client = AsyncMock()

    # Configure the mock methods
    async def mock_get_market_data(symbol, timeframe, start_date, end_date):
        return pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [102.0, 103.0, 104.0],
            'low': [99.0, 100.0, 101.0],
            'close': [101.0, 102.0, 103.0],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range(start='2023-01-01', periods=3))

    data_client.get_market_data.side_effect = mock_get_market_data
    return data_client

@pytest.fixture
def backtest_service(mock_repository, mock_data_client):
    """Create a backtest service for testing."""
    service = BacktestService(repository=mock_repository, data_client=mock_data_client)

    # Mock the backtest engine
    async def mock_run_backtest(strategy, data, parameters):
        return BacktestResult(
            backtest_id=parameters.get('backtest_id', 'test_backtest_id'),
            strategy_id=parameters.get('strategy_id', 'test_strategy_id'),
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 3),
            initial_balance=10000.0,
            final_balance=10500.0,
            total_trades=5,
            winning_trades=3,
            losing_trades=2,
            performance_metrics={
                'total_return': 0.05,
                'annualized_return': 0.2,
                'sharpe_ratio': 1.5,
                'max_drawdown': 0.02,
                'win_rate': 0.6,
                'profit_factor': 2.0,
                'average_trade': 100.0,
                'average_winning_trade': 200.0,
                'average_losing_trade': -50.0
            },
            trades=[],
            equity_curve=[],
            parameters=parameters
        )

    service.backtest_engine.run_backtest = mock_run_backtest

    # Mock the _load_strategies method to return test strategies
    def mock_load_strategies():
        return {
            'moving_average_crossover': StrategyMetadata(
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
            ),
            'rsi_strategy': StrategyMetadata(
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
        }

    service._load_strategies = mock_load_strategies
    service.strategies = mock_load_strategies()

    return service

@pytest.mark.asyncio
async def test_run_backtest(backtest_service, mock_repository):
    """Test running a backtest."""
    # Create a backtest request
    request = BacktestRequest(
        strategy_id="moving_average_crossover",
        symbol="EURUSD",
        timeframe="1h",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 3),
        initial_balance=10000.0,
        parameters={"short_window": 10, "long_window": 50}
    )

    # Run the backtest
    with patch('uuid.uuid4', return_value='test-uuid'):
        response = await backtest_service.run_backtest(request)

    # Check that the response is correct
    assert isinstance(response, BacktestResponse)
    assert response.backtest_id is not None
    assert response.status == BacktestStatus.PENDING

    # Check that the repository was called correctly
    assert mock_repository.save_backtest.called
    assert mock_repository.update_backtest_status.called

@pytest.mark.asyncio
async def test_get_backtest_result(backtest_service, mock_repository):
    """Test getting a backtest result."""
    # Get the backtest result
    result = await backtest_service.get_backtest_result('test_backtest_id')

    # Check that the result is correct
    assert isinstance(result, BacktestResult)
    assert result.backtest_id == 'test_backtest_id'
    assert result.strategy_id == 'test_strategy_id'
    assert result.initial_balance == 10000.0
    assert result.final_balance == 10500.0
    assert result.total_trades == 5
    assert result.winning_trades == 3
    assert result.losing_trades == 2
    assert hasattr(result.performance_metrics, 'total_return')
    assert result.performance_metrics.total_return == 0.05
    assert hasattr(result.performance_metrics, 'sharpe_ratio')
    assert result.performance_metrics.sharpe_ratio == 1.5

    # Check that the repository was called correctly
    assert mock_repository.get_backtest.called
    mock_repository.get_backtest.assert_called_with('test_backtest_id')

@pytest.mark.asyncio
async def test_list_backtests(backtest_service, mock_repository):
    """Test listing backtests."""
    # List the backtests
    backtests = await backtest_service.list_backtests(strategy_id='test_strategy_id')

    # Check that the backtests are correct
    assert len(backtests) == 2
    assert backtests[0]['backtest_id'] == 'test_backtest_id_1'
    assert backtests[0]['strategy_id'] == 'test_strategy_id'
    assert backtests[0]['symbol'] == 'EURUSD'
    assert backtests[1]['backtest_id'] == 'test_backtest_id_2'
    assert backtests[1]['strategy_id'] == 'test_strategy_id'
    assert backtests[1]['symbol'] == 'GBPUSD'

    # Check that the repository was called correctly
    assert mock_repository.list_backtests.called
    mock_repository.list_backtests.assert_called_with(
        strategy_id='test_strategy_id',
        symbol=None,
        limit=100,
        offset=0
    )

@pytest.mark.asyncio
async def test_optimize_strategy(backtest_service, mock_repository):
    """Test optimizing a strategy."""
    # Create an optimization request
    request = OptimizationRequest(
        strategy_id="moving_average_crossover",
        symbol="EURUSD",
        timeframe="1h",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 3),
        initial_balance=10000.0,
        parameters_to_optimize={
            "short_window": {"min": 5, "max": 20, "step": 5},
            "long_window": {"min": 20, "max": 100, "step": 20}
        },
        optimization_metric="sharpe_ratio",
        optimization_method="grid_search",
        max_evaluations=10
    )

    # Mock the optimization process
    with patch('uuid.uuid4', return_value='test-optimization-uuid'):
        # Run the optimization
        response = await backtest_service.optimize_strategy(request)

    # Check that the response is correct
    assert isinstance(response, OptimizationResponse)
    assert response.optimization_id is not None
    assert response.status == 'pending'

    # Check that the repository was called correctly
    assert mock_repository.save_optimization.called

@pytest.mark.asyncio
async def test_get_optimization_result(backtest_service, mock_repository):
    """Test getting an optimization result."""
    # Get the optimization result
    result = await backtest_service.get_optimization_result('test_optimization_id')

    # Check that the result is correct
    assert isinstance(result, OptimizationResult)
    assert result.optimization_id == 'test_optimization_id'
    assert result.strategy_id == 'test_strategy_id'
    assert result.symbol == 'EURUSD'
    assert result.timeframe == '1h'
    assert result.optimization_metric == 'sharpe_ratio'
    assert result.optimization_method == 'grid_search'
    assert result.best_parameters == {'short_window': 10, 'long_window': 50}
    assert result.best_metric_value == 1.5

    # Check that the repository was called correctly
    assert mock_repository.get_optimization.called
    mock_repository.get_optimization.assert_called_with('test_optimization_id')

@pytest.mark.asyncio
async def test_run_walk_forward_test(backtest_service, mock_repository):
    """Test running a walk-forward test."""
    # Create a walk-forward test request
    request = WalkForwardTestRequest(
        strategy_id="moving_average_crossover",
        symbol="EURUSD",
        timeframe="1h",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 31),
        initial_balance=10000.0,
        parameters={"base_param": "value"},
        optimization_window=100,
        test_window=20,
        optimization_metric="sharpe_ratio",
        parameters_to_optimize={
            "short_window": {"min": 5, "max": 20, "step": 5},
            "long_window": {"min": 20, "max": 100, "step": 20}
        }
    )

    # Mock the walk-forward test process
    with patch('uuid.uuid4', return_value='test-walk-forward-uuid'):
        # Run the walk-forward test
        response = await backtest_service.run_walk_forward_test(request)

    # Check that the response is correct
    assert isinstance(response, WalkForwardTestResponse)
    assert response.test_id is not None
    assert response.status == 'pending'

    # Check that the repository was called correctly
    assert mock_repository.save_walk_forward_test.called

@pytest.mark.asyncio
async def test_get_walk_forward_test_result(backtest_service, mock_repository):
    """Test getting a walk-forward test result."""
    # Get the walk-forward test result
    result = await backtest_service.get_walk_forward_test_result('test_walk_forward_id')

    # Check that the result is correct
    assert isinstance(result, WalkForwardTestResult)
    assert result.test_id == 'test_walk_forward_id'
    assert result.strategy_id == 'test_strategy_id'
    assert result.symbol == 'EURUSD'
    assert result.timeframe == '1h'
    assert result.initial_balance == 10000.0
    assert result.final_balance == 11000.0
    assert result.total_trades == 50
    assert result.winning_trades == 30
    assert result.losing_trades == 20
    assert hasattr(result.performance_metrics, 'total_return')
    assert result.performance_metrics.total_return == 0.1
    assert hasattr(result.performance_metrics, 'sharpe_ratio')
    assert result.performance_metrics.sharpe_ratio == 1.8
    assert len(result.windows) == 2
    assert result.windows[0]['type'] == 'optimization'
    assert result.windows[1]['type'] == 'test'
    assert '0' in result.parameters_by_window
    assert 'short_window' in result.parameters_by_window['0']

    # Check that the repository was called correctly
    assert mock_repository.get_walk_forward_test.called
    mock_repository.get_walk_forward_test.assert_called_with('test_walk_forward_id')

@pytest.mark.asyncio
async def test_list_strategies(backtest_service):
    """Test listing available strategies."""
    # List the strategies
    result = await backtest_service.list_strategies()

    # Check that the result is correct
    assert hasattr(result, 'strategies')
    assert hasattr(result, 'count')
    assert len(result.strategies) == 2
    assert result.count == 2
    assert result.strategies[0].strategy_id == 'moving_average_crossover'
    assert result.strategies[0].name == 'Moving Average Crossover'
    assert result.strategies[1].strategy_id == 'rsi_strategy'
    assert result.strategies[1].name == 'RSI Strategy'
    assert 'short_window' in result.strategies[0].parameters
    assert 'rsi_period' in result.strategies[1].parameters