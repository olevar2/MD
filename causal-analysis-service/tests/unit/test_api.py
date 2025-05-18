"""
Unit tests for causal analysis service.
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

def test_causal_analysis_endpoint():
    """Test that the causal analysis endpoint works correctly."""
    # Sample data for testing
    test_data = {
        "variables": ["price", "volume", "volatility"],
        "data": [
            {"price": 100, "volume": 1000, "volatility": 0.1},
            {"price": 101, "volume": 1100, "volatility": 0.2},
            {"price": 102, "volume": 1200, "volatility": 0.15},
            {"price": 103, "volume": 1300, "volatility": 0.25},
            {"price": 104, "volume": 1400, "volatility": 0.3}
        ],
        "method": "pc"
    }
    
    # This is a simplified test that just checks if the endpoint responds
    # In a real test, you would verify the actual causal analysis results
    response = client.post("/api/v1/causal-analysis", json=test_data)
    assert response.status_code == 200
    assert "edges" in response.json()

@pytest.mark.asyncio
async def test_async_functionality():
    """Test async functionality."""
    # This is a placeholder for testing async functionality
    # In a real test, you would test async functions directly
    assert True
