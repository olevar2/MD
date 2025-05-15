"""
Unit tests for the Backtest API routes.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from datetime import datetime

from tests.test_main import app
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

client = TestClient(app)

@pytest.fixture
def mock_backtest_service():
    """Create a mock backtest service for testing."""
    with patch('app.core.service_dependencies.backtest_service') as mock_service:
        # Configure run_backtest
        mock_service.run_backtest = AsyncMock(return_value=BacktestResponse(
            backtest_id="test-backtest-id",
            status=BacktestStatus.PENDING,
            message="Backtest submitted successfully"
        ))

        # Configure get_backtest_result
        mock_service.get_backtest_result = AsyncMock(return_value=BacktestResult(
            backtest_id="test-backtest-id",
            strategy_id="test-strategy-id",
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
            parameters={}
        ))

        # Configure list_backtests
        mock_service.list_backtests = AsyncMock(return_value=[
            {
                'backtest_id': 'test-backtest-id-1',
                'strategy_id': 'test-strategy-id',
                'symbol': 'EURUSD',
                'timeframe': '1h',
                'start_date': '2023-01-01T00:00:00',
                'end_date': '2023-01-03T00:00:00',
                'initial_balance': 10000.0,
                'status': 'completed',
                'created_at': '2023-01-04T00:00:00'
            }
        ])

        # Configure list_strategies
        mock_service.list_strategies = AsyncMock(return_value=StrategyListResponse(
            strategies=[
                StrategyMetadata(
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
            ],
            count=1
        ))

        yield mock_service

def test_run_backtest(mock_backtest_service):
    """Test the run backtest endpoint."""
    # Create a backtest request
    request_data = {
        "strategy_id": "moving_average_crossover",
        "symbol": "EURUSD",
        "timeframe": "1h",
        "start_date": "2023-01-01T00:00:00",
        "end_date": "2023-01-03T00:00:00",
        "initial_balance": 10000.0,
        "parameters": {"short_window": 10, "long_window": 50}
    }

    # Send the request
    response = client.post("/api/v1/backtests", json=request_data)

    # Check the response
    assert response.status_code == 200
    data = response.json()
    assert data["backtest_id"] == "test-backtest-id"
    assert data["status"] == "pending"
    assert "message" in data

    # Check that the service was called correctly
    mock_backtest_service.run_backtest.assert_called_once()

def test_get_backtest_result(mock_backtest_service):
    """Test the get backtest result endpoint."""
    # Send the request
    response = client.get("/api/v1/backtests/test-backtest-id")

    # Check the response
    assert response.status_code == 200
    data = response.json()
    assert data["backtest_id"] == "test-backtest-id"
    assert data["strategy_id"] == "test-strategy-id"
    assert "performance_metrics" in data
    assert data["performance_metrics"]["total_return"] == 0.05

    # Check that the service was called correctly
    mock_backtest_service.get_backtest_result.assert_called_once_with("test-backtest-id")

def test_list_backtests(mock_backtest_service):
    """Test the list backtests endpoint."""
    # Send the request
    response = client.get("/api/v1/backtests")

    # Check the response
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["backtest_id"] == "test-backtest-id-1"
    assert data[0]["strategy_id"] == "test-strategy-id"

    # Check that the service was called correctly
    mock_backtest_service.list_backtests.assert_called_once()

def test_list_strategies(mock_backtest_service):
    """Test the list strategies endpoint."""
    # Send the request
    response = client.get("/api/v1/strategies")

    # Check the response
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["strategy_id"] == "moving_average_crossover"
    assert data[0]["name"] == "Moving Average Crossover"

    # Check that the service was called correctly
    mock_backtest_service.list_strategies.assert_called_once()
