"""
Tests for client correlation functionality.
"""

import pytest
from unittest.mock import MagicMock, patch

from common_lib.correlation import (
    add_correlation_id_to_headers,
    with_correlation_headers,
    with_async_correlation_headers,
    ClientCorrelationMixin,
    get_correlation_id,
    set_correlation_id,
    clear_correlation_id,
    CORRELATION_ID_HEADER
)


def test_add_correlation_id_to_headers():
    """Test adding correlation ID to headers."""
    # Clear any existing correlation ID
    clear_correlation_id()

    # Test with empty headers and no correlation ID
    headers = {}
    updated_headers = add_correlation_id_to_headers(headers)
    assert CORRELATION_ID_HEADER in updated_headers
    assert isinstance(updated_headers[CORRELATION_ID_HEADER], str)

    # Test with existing headers and no correlation ID
    headers = {"Content-Type": "application/json"}
    updated_headers = add_correlation_id_to_headers(headers)
    assert CORRELATION_ID_HEADER in updated_headers
    assert "Content-Type" in updated_headers
    assert updated_headers["Content-Type"] == "application/json"

    # Test with correlation ID in context
    test_id = "test-correlation-id"
    set_correlation_id(test_id)
    updated_headers = add_correlation_id_to_headers({})
    assert updated_headers[CORRELATION_ID_HEADER] == test_id

    # Test with explicit correlation ID
    explicit_id = "explicit-correlation-id"
    updated_headers = add_correlation_id_to_headers({}, explicit_id)
    assert updated_headers[CORRELATION_ID_HEADER] == explicit_id

    # Clear the correlation ID
    clear_correlation_id()


def test_with_correlation_headers_decorator():
    """Test with_correlation_headers decorator."""
    # Clear any existing correlation ID
    clear_correlation_id()

    # Define a decorated function
    @with_correlation_headers
    def test_function(headers=None):
        return headers

    # Call the function with no headers
    headers = test_function()
    assert CORRELATION_ID_HEADER in headers
    assert isinstance(headers[CORRELATION_ID_HEADER], str)

    # Call the function with existing headers
    existing_headers = {"Content-Type": "application/json"}
    headers = test_function(headers=existing_headers)
    assert CORRELATION_ID_HEADER in headers
    assert "Content-Type" in headers
    assert headers["Content-Type"] == "application/json"

    # Call the function with correlation ID in context
    test_id = "test-correlation-id"
    set_correlation_id(test_id)
    headers = test_function()
    assert headers[CORRELATION_ID_HEADER] == test_id

    # Clear the correlation ID
    clear_correlation_id()


@pytest.mark.asyncio
async def test_with_async_correlation_headers_decorator():
    """Test with_async_correlation_headers decorator."""
    # Clear any existing correlation ID
    clear_correlation_id()

    # Define a decorated function
    @with_async_correlation_headers
    async def test_function(headers=None):
        return headers

    # Call the function with no headers
    headers = await test_function()
    assert CORRELATION_ID_HEADER in headers
    assert isinstance(headers[CORRELATION_ID_HEADER], str)

    # Call the function with existing headers
    existing_headers = {"Content-Type": "application/json"}
    headers = await test_function(headers=existing_headers)
    assert CORRELATION_ID_HEADER in headers
    assert "Content-Type" in headers
    assert headers["Content-Type"] == "application/json"

    # Call the function with correlation ID in context
    test_id = "test-correlation-id"
    set_correlation_id(test_id)
    headers = await test_function()
    assert headers[CORRELATION_ID_HEADER] == test_id

    # Clear the correlation ID
    clear_correlation_id()


def test_client_correlation_mixin():
    """Test ClientCorrelationMixin."""
    # Create a test class that implements ClientCorrelationMixin
    class TestClient(ClientCorrelationMixin):
    """
    TestClient class that inherits from ClientCorrelationMixin.
    
    Attributes:
        Add attributes here
    """

        def __init__(self, config=None):
    """
      init  .
    
    Args:
        config: Description of config
    
    """

            self.config = config or {}

        def with_correlation_id(self, correlation_id=None):
    """
    With correlation id.
    
    Args:
        correlation_id: Description of correlation_id
    
    """

            # Implementation of abstract method
            if correlation_id is None:
                correlation_id = get_correlation_id() or "default-id"

            new_config = self.config.copy()
            new_config["correlation_id"] = correlation_id
            return TestClient(new_config)

    # Test creating a client with correlation ID
    client = TestClient()
    correlation_id = "test-correlation-id"
    new_client = client.with_correlation_id(correlation_id)

    assert new_client.config["correlation_id"] == correlation_id

    # Test class method decorators
    @TestClient.add_correlation_headers
    def test_function(headers=None):
        return headers

    headers = test_function()
    assert CORRELATION_ID_HEADER in headers

    @TestClient.add_async_correlation_headers
    async def test_async_function(headers=None):
        return headers

    # We don't need to await here since we're just testing the decorator
    assert callable(test_async_function)