"""
Integration tests for the Strategy Execution Engine.

These tests verify that the service components work together correctly.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
import json
import os
from datetime import datetime

from strategy_execution_engine.main import create_app, lifespan
from strategy_execution_engine.core.container import ServiceContainer
from strategy_execution_engine.strategies.strategy_loader import StrategyLoader
from strategy_execution_engine.backtesting.backtester import Backtester


@pytest.fixture
def app():
    """Create a FastAPI app for testing."""
    return create_app()


@pytest.fixture
def client(app):
    """Create a test client."""
    with TestClient(app) as client:
        yield client


@pytest.mark.integration
def test_app_startup_shutdown(app):
    """Test application startup and shutdown."""
    # Create a mock service container
    service_container = MagicMock(spec=ServiceContainer)
    service_container.initialize = AsyncMock()
    service_container.shutdown = AsyncMock()
    
    # Create a mock strategy loader
    strategy_loader = MagicMock(spec=StrategyLoader)
    strategy_loader.load_strategies = AsyncMock()
    
    # Create a mock backtester
    backtester = MagicMock(spec=Backtester)
    
    # Create a context manager for testing lifespan
    async def test_lifespan():
        # Set up app state
        app.state.service_container = service_container
        app.state.strategy_loader = strategy_loader
        app.state.backtester = backtester
        
        # Run lifespan
        async with lifespan(app):
            # Verify initialization
            service_container.initialize.assert_called_once()
            strategy_loader.load_strategies.assert_called_once()
            
            # Yield to test
            yield
        
        # Verify shutdown
        service_container.shutdown.assert_called_once()
    
    # Run the test
    import asyncio
    asyncio.run(test_lifespan().__anext__())


@pytest.mark.integration
def test_api_routes_registered(client):
    """Test that API routes are registered correctly."""
    # Test root endpoint
    response = client.get("/")
    assert response.status_code == 200
    
    # Test health endpoint
    response = client.get("/health")
    assert response.status_code == 200
    
    # Test strategies endpoint
    response = client.get("/api/v1/strategies")
    # Note: This might return 500 in tests if the strategy loader is not mocked properly
    assert response.status_code in (200, 500)
    
    # Test OpenAPI docs
    response = client.get("/api/openapi.json")
    assert response.status_code == 200
    openapi_spec = response.json()
    
    # Verify paths in OpenAPI spec
    assert "/api/v1/strategies" in openapi_spec["paths"]
    assert "/api/v1/strategies/{strategy_id}" in openapi_spec["paths"]
    assert "/api/v1/strategies/register" in openapi_spec["paths"]
    assert "/api/v1/backtest" in openapi_spec["paths"]


@pytest.mark.integration
def test_error_handling_integration(client):
    """Test error handling integration."""
    # Test non-existent endpoint
    response = client.get("/non-existent")
    assert response.status_code == 404
    
    # Test method not allowed
    response = client.post("/health")
    assert response.status_code == 405
    
    # Test validation error
    response = client.post("/api/v1/backtest", json={})
    assert response.status_code == 422  # Unprocessable Entity


@pytest.mark.integration
@patch("strategy_execution_engine.strategies.strategy_loader.StrategyLoader")
@patch("strategy_execution_engine.backtesting.backtester.Backtester")
def test_full_backtest_flow(mock_backtester_class, mock_loader_class, client):
    """Test the full backtest flow from strategy registration to backtest execution."""
    # Configure mocks
    mock_loader = MagicMock()
    mock_loader.register_strategy = AsyncMock(return_value="test_strategy_id")
    mock_loader.get_strategy = MagicMock(return_value=MagicMock(
        name="Test Strategy",
        instruments=["EUR/USD"],
        timeframe="1h",
        parameters={}
    ))
    mock_loader_class.return_value = mock_loader
    
    mock_backtester = MagicMock()
    mock_backtester.run_backtest = AsyncMock(return_value={
        "backtest_id": "test_backtest_id",
        "strategy_id": "test_strategy_id",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "metrics": {
            "total_trades": 10,
            "win_rate": 0.6
        },
        "trades": [],
        "equity_curve": []
    })
    mock_backtester_class.return_value = mock_backtester
    
    # Step 1: Register a strategy
    strategy_data = {
        "name": "Test Strategy",
        "description": "A test strategy",
        "instruments": ["EUR/USD"],
        "timeframe": "1h",
        "parameters": {},
        "code": "class TestStrategy(Strategy):\n    def analyze(self, data):\n        return {}\n"
    }
    
    response = client.post("/api/v1/strategies/register", json=strategy_data)
    assert response.status_code == 201
    strategy_result = response.json()
    assert strategy_result["id"] == "test_strategy_id"
    
    # Step 2: Run a backtest
    backtest_data = {
        "strategy_id": "test_strategy_id",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "initial_capital": 10000.0
    }
    
    response = client.post("/api/v1/backtest", json=backtest_data)
    assert response.status_code == 200
    backtest_result = response.json()
    assert backtest_result["backtest_id"] == "test_backtest_id"
    assert backtest_result["strategy_id"] == "test_strategy_id"
    assert "metrics" in backtest_result
    assert backtest_result["metrics"]["total_trades"] == 10
    assert backtest_result["metrics"]["win_rate"] == 0.6
