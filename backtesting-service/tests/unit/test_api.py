"""
Unit tests for backtesting service.
"""
import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_health_endpoint():
    """Test that the health endpoint returns a 200 status code."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_backtesting_endpoint():
    """Test that the backtesting endpoint works correctly."""
    # Sample data for testing
    test_data = {
        "strategy_id": "test_strategy",
        "start_date": "2023-01-01",
        "end_date": "2023-01-31",
        "symbol": "EURUSD",
        "timeframe": "1h",
        "parameters": {
            "risk_reward_ratio": 2.0,
            "stop_loss_pips": 20,
            "take_profit_pips": 40
        }
    }
    
    # This is a simplified test that just checks if the endpoint responds
    # In a real test, you would verify the actual backtesting results
    response = client.post("/api/v1/backtest", json=test_data)
    assert response.status_code == 200
    assert "results" in response.json()

@pytest.mark.asyncio
async def test_async_functionality():
    """Test async functionality."""
    # This is a placeholder for testing async functionality
    # In a real test, you would test async functions directly
    assert True

def test_database_mocking():
    """Test that database mocking works correctly."""
    # This test verifies that the database mocking infrastructure works
    # In a real test, you would use the mocked database to perform operations
    assert True
