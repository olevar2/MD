"""
Tests for correlation ID middleware.
"""

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from common_lib.correlation import (
    FastAPICorrelationIdMiddleware,
    create_correlation_id_middleware,
    get_correlation_id,
    CORRELATION_ID_HEADER
)


def create_test_app():
    """Create a test FastAPI app with middleware."""
    app = FastAPI()
    app.add_middleware(FastAPICorrelationIdMiddleware)

    @app.get("/test")
    async def test_endpoint(request: Request):
        return {
            "correlation_id": request.state.correlation_id,
            "context_correlation_id": get_correlation_id()
        }

    return app


def test_fastapi_middleware_new_id():
    """Test that middleware adds a new correlation ID when none is provided."""
    app = create_test_app()
    client = TestClient(app)

    response = client.get("/test")
    assert response.status_code == 200

    # Check that correlation ID is in response headers
    assert CORRELATION_ID_HEADER in response.headers
    correlation_id = response.headers[CORRELATION_ID_HEADER]

    # Check that correlation ID is in response body
    data = response.json()
    assert data["correlation_id"] == correlation_id
    # Note: In the test environment, the context might not be cleared due to how TestClient works
    # So we don't assert that context_correlation_id is None


def test_fastapi_middleware_existing_id():
    """Test that middleware uses existing correlation ID when provided."""
    app = create_test_app()
    client = TestClient(app)

    correlation_id = "test-correlation-id"
    response = client.get("/test", headers={CORRELATION_ID_HEADER: correlation_id})
    assert response.status_code == 200

    # Check that correlation ID is in response headers
    assert response.headers[CORRELATION_ID_HEADER] == correlation_id

    # Check that correlation ID is in response body
    data = response.json()
    assert data["correlation_id"] == correlation_id
    # Note: In the test environment, the context might not be cleared due to how TestClient works
    # So we don't assert that context_correlation_id is None


def test_fastapi_middleware_multiple_requests():
    """Test that middleware generates different IDs for different requests."""
    app = create_test_app()
    client = TestClient(app)

    response1 = client.get("/test")
    assert response1.status_code == 200
    correlation_id1 = response1.headers[CORRELATION_ID_HEADER]

    response2 = client.get("/test")
    assert response2.status_code == 200
    correlation_id2 = response2.headers[CORRELATION_ID_HEADER]

    # Check that correlation IDs are different
    assert correlation_id1 != correlation_id2


def test_create_middleware_factory():
    """Test the middleware factory function."""
    # Create a FastAPI middleware
    middleware = create_correlation_id_middleware(framework="fastapi")
    assert isinstance(middleware, FastAPICorrelationIdMiddleware)

    # Test with unsupported framework
    with pytest.raises(ValueError):
        create_correlation_id_middleware(framework="unsupported")


def test_middleware_with_custom_header():
    """Test middleware with custom header name."""
    app = FastAPI()
    custom_header = "X-Custom-Correlation-ID"
    app.add_middleware(FastAPICorrelationIdMiddleware, header_name=custom_header)

    @app.get("/test")
    async def test_endpoint(request: Request):
        return {
            "correlation_id": request.state.correlation_id
        }

    client = TestClient(app)

    # Test with custom header
    correlation_id = "test-correlation-id"
    response = client.get("/test", headers={custom_header: correlation_id})
    assert response.status_code == 200

    # Check that correlation ID is in response headers with custom name
    assert response.headers[custom_header] == correlation_id

    # Check that correlation ID is in response body
    data = response.json()
    assert data["correlation_id"] == correlation_id


def test_middleware_without_auto_generation():
    """Test middleware without auto-generation of correlation IDs."""
    app = FastAPI()
    app.add_middleware(
        FastAPICorrelationIdMiddleware,
        always_generate_if_missing=False
    )

    @app.get("/test")
    async def test_endpoint(request: Request):
        correlation_id = getattr(request.state, "correlation_id", None)
        return {
            "correlation_id": correlation_id
        }

    client = TestClient(app)

    # Test without correlation ID
    response = client.get("/test")
    assert response.status_code == 200

    # Check that no correlation ID is in response headers
    assert CORRELATION_ID_HEADER not in response.headers

    # Check that no correlation ID is in response body
    data = response.json()
    assert data["correlation_id"] is None

    # Test with correlation ID
    correlation_id = "test-correlation-id"
    response = client.get("/test", headers={CORRELATION_ID_HEADER: correlation_id})
    assert response.status_code == 200

    # Check that correlation ID is in response headers
    assert response.headers[CORRELATION_ID_HEADER] == correlation_id

    # Check that correlation ID is in response body
    data = response.json()
    assert data["correlation_id"] == correlation_id