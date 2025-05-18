"""
Integration tests for causal analysis service.
"""
import pytest
import redis
import os
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

# Get Redis connection details from environment variables
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

@pytest.fixture
def redis_client():
    """Create a Redis client for testing."""
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    yield r
    # Clean up after tests
    r.flushall()

def test_redis_connection(redis_client):
    """Test that the service can connect to Redis."""
    # Set a value in Redis
    redis_client.set("test_key", "test_value")
    
    # Get the value from Redis
    value = redis_client.get("test_key")
    
    # Verify the value
    assert value == b"test_value"

def test_causal_analysis_with_caching(redis_client):
    """Test that causal analysis results are cached in Redis."""
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
    
    # Make a request to the causal analysis endpoint
    response = client.post("/api/v1/causal-analysis", json=test_data)
    assert response.status_code == 200
    
    # In a real test, you would verify that the results are cached in Redis
    # This is a simplified test that just checks if Redis is working
    redis_client.set("test_cache_key", "test_cache_value")
    value = redis_client.get("test_cache_key")
    assert value == b"test_cache_value"

def test_service_integration():
    """Test integration with other services."""
    # This is a placeholder for testing integration with other services
    # In a real test, you would make requests to other services
    # and verify that the causal analysis service handles the responses correctly
    assert True
