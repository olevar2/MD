"""
Integration tests for correlation ID propagation.

These tests verify that correlation IDs are correctly propagated across service boundaries.
"""

import asyncio
import pytest
import uuid
import logging
from unittest.mock import MagicMock, patch
from fastapi import FastAPI, Request, Depends
from fastapi.testclient import TestClient
from httpx import AsyncClient

from common_lib.correlation import (
    FastAPICorrelationIdMiddleware,
    get_correlation_id,
    set_correlation_id,
    clear_correlation_id,
    CORRELATION_ID_HEADER
)
from common_lib.clients import BaseServiceClient, ClientConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create a logger that includes correlation ID
logger = logging.getLogger("correlation_integration_test")


class CorrelationFilter(logging.Filter):
    """Logging filter that adds correlation ID to log records."""

    def filter(self, record):
        correlation_id = get_correlation_id()
        record.correlation_id = correlation_id or "no-correlation-id"
        return True


# Add the correlation filter to the logger
logger.addFilter(CorrelationFilter())


# Create a mock service A
def create_service_a():
    """Create a mock service A with correlation ID middleware."""
    app = FastAPI(title="Service A")
    app.add_middleware(FastAPICorrelationIdMiddleware)

    @app.get("/api/resource/{resource_id}")
    async def get_resource(resource_id: str, request: Request):
        """Get a resource by ID."""
        # Log the request
        logger.info(f"Service A: Received request for resource {resource_id}")

        # Get the correlation ID from the request
        correlation_id = request.state.correlation_id

        # Return the resource with the correlation ID
        return {
            "id": resource_id,
            "name": f"Resource {resource_id}",
            "service": "A",
            "correlation_id": correlation_id
        }

    @app.post("/api/resource")
    async def create_resource(request: Request):
        """Create a new resource."""
        # Log the request
        logger.info("Service A: Received request to create resource")

        # Get the correlation ID from the request
        correlation_id = request.state.correlation_id

        # Get the request body
        body = await request.json()

        # Create a new resource
        resource_id = str(uuid.uuid4())

        # Return the new resource with the correlation ID
        return {
            "id": resource_id,
            "name": body.get("name", f"Resource {resource_id}"),
            "service": "A",
            "correlation_id": correlation_id
        }

    return app


# Create a mock service B that calls service A
def create_service_b():
    """Create a mock service B with correlation ID middleware."""
    app = FastAPI(title="Service B")
    app.add_middleware(FastAPICorrelationIdMiddleware)

    # Create a client for service A
    service_a_client = BaseServiceClient(
        ClientConfig(
            base_url="http://service-a/api",
            service_name="service-a"
        )
    )

    @app.get("/api/proxy/{resource_id}")
    async def proxy_resource(resource_id: str, request: Request):
        """Proxy a request to service A."""
        # Log the request
        logger.info(f"Service B: Received request to proxy resource {resource_id}")

        # Get the correlation ID from the request
        correlation_id = request.state.correlation_id

        # Create a client with the correlation ID
        client = service_a_client.with_correlation_id(correlation_id)

        # Mock the response from service A
        mock_response = {
            "id": resource_id,
            "name": f"Resource {resource_id}",
            "service": "A",
            "correlation_id": correlation_id
        }

        # Return the response with additional information
        return {
            "proxied_resource": mock_response,
            "service": "B",
            "correlation_id": correlation_id
        }

    @app.post("/api/proxy")
    async def proxy_create_resource(request: Request):
        """Proxy a request to create a resource in service A."""
        # Log the request
        logger.info("Service B: Received request to proxy resource creation")

        # Get the correlation ID from the request
        correlation_id = request.state.correlation_id

        # Get the request body
        body = await request.json()

        # Create a client with the correlation ID
        client = service_a_client.with_correlation_id(correlation_id)

        # Mock the response from service A
        resource_id = str(uuid.uuid4())
        mock_response = {
            "id": resource_id,
            "name": body.get("name", f"Resource {resource_id}"),
            "service": "A",
            "correlation_id": correlation_id
        }

        # Return the response with additional information
        return {
            "proxied_resource": mock_response,
            "service": "B",
            "correlation_id": correlation_id
        }

    return app


@pytest.fixture
def service_a():
    """Fixture for service A."""
    return create_service_a()


@pytest.fixture
def service_b():
    """Fixture for service B."""
    return create_service_b()


@pytest.fixture
def client_a(service_a):
    """Fixture for service A client."""
    return TestClient(service_a)


@pytest.fixture
def client_b(service_b):
    """Fixture for service B client."""
    return TestClient(service_b)


def test_correlation_id_propagation_get(client_a, client_b):
    """Test correlation ID propagation for GET requests."""
    # Clear any existing correlation ID
    clear_correlation_id()

    # Create a correlation ID
    correlation_id = str(uuid.uuid4())

    # Make a request to service B with the correlation ID
    response_b = client_b.get(
        "/api/proxy/123",
        headers={CORRELATION_ID_HEADER: correlation_id}
    )

    # Check that the response has the correlation ID
    assert response_b.status_code == 200
    assert response_b.headers[CORRELATION_ID_HEADER] == correlation_id

    # Check that the response body has the correlation ID
    data_b = response_b.json()
    assert data_b["correlation_id"] == correlation_id
    assert data_b["proxied_resource"]["correlation_id"] == correlation_id

    # Make a request to service A with the correlation ID
    response_a = client_a.get(
        "/api/resource/123",
        headers={CORRELATION_ID_HEADER: correlation_id}
    )

    # Check that the response has the correlation ID
    assert response_a.status_code == 200
    assert response_a.headers[CORRELATION_ID_HEADER] == correlation_id

    # Check that the response body has the correlation ID
    data_a = response_a.json()
    assert data_a["correlation_id"] == correlation_id


def test_correlation_id_propagation_post(client_a, client_b):
    """Test correlation ID propagation for POST requests."""
    # Clear any existing correlation ID
    clear_correlation_id()

    # Create a correlation ID
    correlation_id = str(uuid.uuid4())

    # Make a request to service B with the correlation ID
    response_b = client_b.post(
        "/api/proxy",
        headers={CORRELATION_ID_HEADER: correlation_id},
        json={"name": "Test Resource"}
    )

    # Check that the response has the correlation ID
    assert response_b.status_code == 200
    assert response_b.headers[CORRELATION_ID_HEADER] == correlation_id

    # Check that the response body has the correlation ID
    data_b = response_b.json()
    assert data_b["correlation_id"] == correlation_id
    assert data_b["proxied_resource"]["correlation_id"] == correlation_id

    # Make a request to service A with the correlation ID
    response_a = client_a.post(
        "/api/resource",
        headers={CORRELATION_ID_HEADER: correlation_id},
        json={"name": "Test Resource"}
    )

    # Check that the response has the correlation ID
    assert response_a.status_code == 200
    assert response_a.headers[CORRELATION_ID_HEADER] == correlation_id

    # Check that the response body has the correlation ID
    data_a = response_a.json()
    assert data_a["correlation_id"] == correlation_id


def test_correlation_id_generation(client_a, client_b):
    """Test correlation ID generation when not provided."""
    # Clear any existing correlation ID
    clear_correlation_id()

    # Make a request to service B without a correlation ID
    response_b = client_b.get("/api/proxy/123")

    # Check that the response has a correlation ID
    assert response_b.status_code == 200
    assert CORRELATION_ID_HEADER in response_b.headers

    # Check that the response body has the correlation ID
    data_b = response_b.json()
    assert "correlation_id" in data_b
    assert data_b["correlation_id"] == response_b.headers[CORRELATION_ID_HEADER]
    assert data_b["proxied_resource"]["correlation_id"] == response_b.headers[CORRELATION_ID_HEADER]

    # Make a request to service A without a correlation ID
    response_a = client_a.get("/api/resource/123")

    # Check that the response has a correlation ID
    assert response_a.status_code == 200
    assert CORRELATION_ID_HEADER in response_a.headers

    # Check that the response body has the correlation ID
    data_a = response_a.json()
    assert "correlation_id" in data_a
    assert data_a["correlation_id"] == response_a.headers[CORRELATION_ID_HEADER]
