"""
End-to-end tests for backtesting service.
"""
import pytest
import os
import httpx

# Get service URL from environment variables
SERVICE_URL = os.getenv("SERVICE_URL", "http://localhost:8002")

@pytest.mark.asyncio
async def test_end_to_end_workflow():
    """Test the complete workflow of the backtesting service."""
    async with httpx.AsyncClient() as client:
        # Step 1: Check if the service is healthy
        response = await client.get(f"{SERVICE_URL}/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        
        # Step 2: Submit a backtesting request
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
        
        response = await client.post(f"{SERVICE_URL}/api/v1/backtest", json=test_data)
        assert response.status_code == 200
        result = response.json()
        
        # Step 3: Verify the result structure
        assert "results" in result
        assert "metrics" in result
        
        # Step 4: Use the result to make another request (e.g., get detailed metrics)
        backtest_id = result.get("id")
        if backtest_id:
            response = await client.get(f"{SERVICE_URL}/api/v1/backtest/{backtest_id}/metrics")
            assert response.status_code == 200
            metrics = response.json()
            assert "profit_factor" in metrics
            assert "sharpe_ratio" in metrics
            assert "max_drawdown" in metrics
        
        # Step 5: Verify that the service can handle errors gracefully
        invalid_data = {
            "strategy_id": "invalid_strategy",
            "start_date": "2023-01-01",
            "end_date": "invalid_date",
            "symbol": "INVALID",
            "timeframe": "invalid",
            "parameters": {}
        }
        
        response = await client.post(f"{SERVICE_URL}/api/v1/backtest", json=invalid_data)
        # The service should return a 4xx status code for invalid input
        assert 400 <= response.status_code < 500
