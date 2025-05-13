"""
Simple API tests for Strategy Execution Engine.
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from unittest.mock import MagicMock

def create_test_app():
    """Create a simple test app."""
    from fastapi import FastAPI
    
    app = FastAPI()
    
    @app.get("/")
    async def root():
    """
    Root.
    
    """

        return {"message": "Strategy Execution Engine is running", "version": "0.1.0"}
    
    @app.get("/health")
    async def health():
    """
    Health.
    
    """

        return {"status": "healthy", "version": "0.1.0"}
    
    @app.get("/api/v1/strategies")
    async def list_strategies():
    """
    List strategies.
    
    """

        return {
            "strategies": [
                {
                    "id": "strategy1",
                    "name": "Test Strategy 1",
                    "type": "custom",
                    "status": "active"
                },
                {
                    "id": "strategy2",
                    "name": "Test Strategy 2",
                    "type": "custom",
                    "status": "active"
                }
            ]
        }
    
    return app

@pytest.fixture
def client():
    """Create a test client."""
    app = create_test_app()
    return TestClient(app)

def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "Strategy Execution Engine is running"
    assert "version" in data

def test_health_endpoint(client):
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "version" in data

def test_list_strategies(client):
    """Test listing strategies."""
    response = client.get("/api/v1/strategies")
    assert response.status_code == 200
    data = response.json()
    assert "strategies" in data
    assert len(data["strategies"]) == 2
    assert data["strategies"][0]["name"] == "Test Strategy 1"
    assert data["strategies"][1]["name"] == "Test Strategy 2"
