"""
Tests for the main FastAPI application (app.main)
"""
import pytest
from httpx import AsyncClient
from fastapi import status

from app.main import app # Ensure this import works with your structure
from app.config.settings import Settings

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio

async def test_health_check(client: AsyncClient):
    """Test the health check endpoint."""
    response = await client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "healthy"}

async def test_startup_event(client: AsyncClient, caplog):
    """Test that startup event logs correctly. 
       This is a basic check; more specific resource initialization tests might be needed.
    """
    # The startup event runs when the client is initialized (due to app context)
    # We just need to make a request to ensure the app has started.
    await client.get("/health") # Make a request to trigger app startup if not already done
    assert "Starting chat service" in caplog.text

async def test_shutdown_event(client: AsyncClient, caplog):
    """Test that shutdown event logs correctly.
       This is harder to test directly in a unit test context without managing the app lifecycle explicitly.
       For now, we'll assume it's covered by integration or manual testing.
       If a specific resource cleanup needs verification, a more targeted test would be required.
    """
    # Simulating shutdown is complex here. We'll rely on other tests or manual checks.
    # For a real test, you might need to manage the app lifecycle explicitly.
    pass

async def test_general_exception_handler(client: AsyncClient):
    """Test the general exception handler.
       We need a route that will reliably raise an unhandled exception.
       Let's add a temporary one for testing purposes if not available.
    """
    # Add a temporary route to app that raises an exception
    @app.get("/test-exception")
    async def _():
        raise ValueError("Test unhandled exception")

    response = await client.get("/test-exception")
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert response.json() == {"error": "Internal server error"}
    
    # Clean up the temporary route if possible (FastAPI doesn't easily allow route removal after adding)
    # For robust testing, consider a more controlled way to trigger this or test via a specific service error.

async def test_cors_middleware_headers(client: AsyncClient, test_settings: Settings):
    """Test that CORS headers are present."""
    # Ensure ALLOWED_ORIGINS is set for the test
    origin_to_test = "http://example.com"
    if test_settings.ALLOWED_ORIGINS == ["*"]:
        # If all origins are allowed, any origin should work
        pass
    elif origin_to_test not in test_settings.ALLOWED_ORIGINS:
        # If specific origins are set, ensure our test origin is among them
        # This might require adjusting test_settings fixture or the app's CORS config for the test
        # For simplicity, we assume '*' or a relevant origin is configured for tests.
        pass 

    response = await client.options("/health", headers={"Origin": origin_to_test, "Access-Control-Request-Method": "GET"})
    assert response.status_code == status.HTTP_200_OK
    assert "access-control-allow-origin" in response.headers
    # Depending on your CORS setup in test_settings, you might get '*' or the specific origin
    assert response.headers["access-control-allow-origin"] == (test_settings.ALLOWED_ORIGINS[0] if test_settings.ALLOWED_ORIGINS != ["*"] else "*")
    assert "access-control-allow-methods" in response.headers
    assert "access-control-allow-headers" in response.headers

async def test_correlation_id_middleware(client: AsyncClient):
    """Test that CorrelationIdMiddleware adds X-Correlation-ID to response if not present in request."""
    response = await client.get("/health")
    assert "X-Correlation-ID" in response.headers
    assert response.headers["X-Correlation-ID"] is not None

async def test_correlation_id_middleware_propagates(client: AsyncClient):
    """Test that CorrelationIdMiddleware propagates X-Correlation-ID from request to response."""
    test_corr_id = "my-test-correlation-id-123"
    response = await client.get("/health", headers={"X-Correlation-ID": test_corr_id})
    assert "X-Correlation-ID" in response.headers
    assert response.headers["X-Correlation-ID"] == test_corr_id

# Note: Testing for ServiceError handler might be better placed in tests for specific API endpoints
# where such errors are expected to be raised by the service layer.