"""
Tests for middleware in Monitoring & Alerting Service.
"""
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from core.middleware import CorrelationIdMiddleware


def create_test_app():
    """Create a test FastAPI app with middleware."""
    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)
    
    @app.get("/test")
    async def test_endpoint(request: Request):
        return {
            "correlation_id": request.state.correlation_id
        }
    
    return app


def test_correlation_id_middleware_new_id():
    """Test that middleware adds a new correlation ID when none is provided."""
    app = create_test_app()
    client = TestClient(app)
    
    response = client.get("/test")
    assert response.status_code == 200
    
    # Check that correlation ID is in response headers
    assert "X-Correlation-ID" in response.headers
    correlation_id = response.headers["X-Correlation-ID"]
    
    # Check that correlation ID is in response body
    data = response.json()
    assert data["correlation_id"] == correlation_id


def test_correlation_id_middleware_existing_id():
    """Test that middleware uses existing correlation ID when provided."""
    app = create_test_app()
    client = TestClient(app)
    
    correlation_id = "test-correlation-id"
    response = client.get("/test", headers={"X-Correlation-ID": correlation_id})
    assert response.status_code == 200
    
    # Check that correlation ID is in response headers
    assert response.headers["X-Correlation-ID"] == correlation_id
    
    # Check that correlation ID is in response body
    data = response.json()
    assert data["correlation_id"] == correlation_id


def test_correlation_id_middleware_multiple_requests():
    """Test that middleware generates different IDs for different requests."""
    app = create_test_app()
    client = TestClient(app)
    
    response1 = client.get("/test")
    assert response1.status_code == 200
    correlation_id1 = response1.headers["X-Correlation-ID"]
    
    response2 = client.get("/test")
    assert response2.status_code == 200
    correlation_id2 = response2.headers["X-Correlation-ID"]
    
    # Check that correlation IDs are different
    assert correlation_id1 != correlation_id2
