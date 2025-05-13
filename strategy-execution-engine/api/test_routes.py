"""
Tests for API routes in the Strategy Execution Engine.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
import json
from datetime import datetime

from core.main_1 import create_app
from core.strategy_loader import StrategyLoader
from core.backtester import Backtester


@pytest.fixture
def mock_strategy_loader():
    """Create a mock strategy loader."""
    loader = MagicMock(spec=StrategyLoader)
    loader.get_available_strategies.return_value = {
        "strategy1": {
            "name": "Test Strategy 1",
            "type": "custom",
            "status": "active",
            "instruments": ["EUR/USD", "GBP/USD"],
            "timeframe": "1h",
            "description": "Test strategy 1",
            "parameters": {"param1": 10, "param2": "value"}
        },
        "strategy2": {
            "name": "Test Strategy 2",
            "type": "custom",
            "status": "active",
            "instruments": ["USD/JPY"],
            "timeframe": "4h",
            "description": "Test strategy 2",
            "parameters": {"param1": 20, "param2": "value2"}
        }
    }

    # Mock get_strategy method
    strategy_mock = MagicMock()
    strategy_mock.name = "Test Strategy 1"
    strategy_mock.instruments = ["EUR/USD", "GBP/USD"]
    strategy_mock.timeframe = "1h"
    strategy_mock.description = "Test strategy 1"
    strategy_mock.parameters = {"param1": 10, "param2": "value"}
    loader.get_strategy.return_value = strategy_mock

    # Mock register_strategy method
    loader.register_strategy = AsyncMock(return_value="new_strategy_id")

    return loader


@pytest.fixture
def mock_backtester():
    """Create a mock backtester."""
    backtester = MagicMock(spec=Backtester)
    backtester.run_backtest = AsyncMock(return_value={
        "backtest_id": "test_backtest_id",
        "strategy_id": "strategy1",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "metrics": {
            "total_trades": 50,
            "winning_trades": 30,
            "losing_trades": 20,
            "win_rate": 0.6,
            "profit_factor": 1.5,
            "net_profit": 5000,
            "net_profit_pct": 50,
            "max_drawdown": 10
        },
        "trades": [
            {
                "id": "trade1",
                "position_id": "position1",
                "instrument": "EUR/USD",
                "type": "long",
                "entry_price": 1.1000,
                "entry_time": "2023-01-05T10:00:00",
                "exit_price": 1.1100,
                "exit_time": "2023-01-06T10:00:00",
                "size": 1.0,
                "profit_loss": 100,
                "profit_loss_pct": 1.0
            }
        ],
        "equity_curve": [
            {
                "timestamp": "2023-01-01T00:00:00",
                "equity": 10000
            },
            {
                "timestamp": "2023-12-31T00:00:00",
                "equity": 15000
            }
        ]
    })
    return backtester


@pytest.fixture
def client(mock_strategy_loader, mock_backtester):
    """Create a test client with mocked dependencies."""
    app = create_app()

    # Override dependencies
    app.state.strategy_loader = mock_strategy_loader
    app.state.backtester = mock_backtester

    return TestClient(app)


def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "Strategy Execution Engine is running"
    assert "version" in data
    assert "timestamp" in data


def test_list_strategies(client, mock_strategy_loader):
    """Test listing strategies."""
    response = client.get("/api/v1/strategies")
    assert response.status_code == 200
    data = response.json()
    assert "strategies" in data
    assert len(data["strategies"]) == 2

    # Verify strategy loader was called
    mock_strategy_loader.get_available_strategies.assert_called_once()

    # Verify response data
    strategies = data["strategies"]
    assert strategies[0]["name"] == "Test Strategy 1"
    assert strategies[1]["name"] == "Test Strategy 2"


def test_get_strategy(client, mock_strategy_loader):
    """Test getting a specific strategy."""
    response = client.get("/api/v1/strategies/strategy1")
    assert response.status_code == 200
    data = response.json()

    # Verify strategy loader was called
    mock_strategy_loader.get_strategy.assert_called_once_with("strategy1")

    # Verify response data
    assert data["name"] == "Test Strategy 1"
    assert data["instruments"] == ["EUR/USD", "GBP/USD"]
    assert data["timeframe"] == "1h"


def test_get_strategy_not_found(client, mock_strategy_loader):
    """Test getting a non-existent strategy."""
    # Configure mock to return None for non-existent strategy
    mock_strategy_loader.get_strategy.return_value = None

    response = client.get("/api/v1/strategies/non_existent")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "not found" in data["detail"]


def test_register_strategy(client, mock_strategy_loader):
    """Test registering a new strategy."""
    strategy_data = {
        "name": "New Strategy",
        "description": "A new test strategy",
        "instruments": ["EUR/USD", "GBP/USD"],
        "timeframe": "1h",
        "parameters": {"param1": 10, "param2": "value"},
        "code": "class NewStrategy(Strategy):
    """
    NewStrategy class that inherits from Strategy.
    
    Attributes:
        Add attributes here
    """
\n    def analyze(self, data):
    """
    Analyze.
    
    Args:
        data: Description of data
    
    """
\n        return {}\n"
    }

    response = client.post("/api/v1/strategies/register", json=strategy_data)
    assert response.status_code == 201
    data = response.json()

    # Verify strategy loader was called with correct arguments
    mock_strategy_loader.register_strategy.assert_called_once()
    call_args = mock_strategy_loader.register_strategy.call_args[1]
    assert call_args["name"] == "New Strategy"
    assert call_args["instruments"] == ["EUR/USD", "GBP/USD"]
    assert call_args["timeframe"] == "1h"

    # Verify response data
    assert data["id"] == "new_strategy_id"
    assert data["name"] == "New Strategy"
    assert data["status"] == "active"


def test_run_backtest(client, mock_backtester):
    """Test running a backtest."""
    backtest_data = {
        "strategy_id": "strategy1",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "initial_capital": 10000.0,
        "parameters": {"param1": 10, "param2": "value"}
    }

    response = client.post("/api/v1/backtest", json=backtest_data)
    assert response.status_code == 200
    data = response.json()

    # Verify backtester was called with correct arguments
    mock_backtester.run_backtest.assert_called_once()
    call_args = mock_backtester.run_backtest.call_args[1]
    assert call_args["strategy_id"] == "strategy1"
    assert call_args["start_date"] == "2023-01-01"
    assert call_args["end_date"] == "2023-12-31"
    assert call_args["initial_capital"] == 10000.0

    # Verify response data
    assert data["backtest_id"] == "test_backtest_id"
    assert data["strategy_id"] == "strategy1"
    assert data["metrics"]["total_trades"] == 50
    assert data["metrics"]["win_rate"] == 0.6
    assert len(data["trades"]) == 1
    assert len(data["equity_curve"]) == 2
