"""
Tests for correlation ID utility.
"""

import asyncio
import pytest
import threading
from unittest.mock import MagicMock, patch

from common_lib.correlation import (
    generate_correlation_id,
    get_correlation_id,
    set_correlation_id,
    clear_correlation_id,
    correlation_id_context,
    async_correlation_id_context,
    with_correlation_id,
    with_async_correlation_id,
    get_correlation_id_from_request,
    add_correlation_id_to_headers,
    CORRELATION_ID_HEADER
)


def test_generate_correlation_id():
    """Test that generate_correlation_id returns a unique string."""
    id1 = generate_correlation_id()
    id2 = generate_correlation_id()

    assert isinstance(id1, str)
    assert len(id1) > 0
    assert id1 != id2


def test_get_set_correlation_id():
    """Test setting and getting correlation ID."""
    # Clear any existing correlation ID
    clear_correlation_id()

    # Initially, no correlation ID should be set
    assert get_correlation_id() is None

    # Set a correlation ID
    test_id = "test-correlation-id"
    set_correlation_id(test_id)

    # Get the correlation ID
    assert get_correlation_id() == test_id

    # Clear the correlation ID
    clear_correlation_id()
    assert get_correlation_id() is None


def test_correlation_id_context():
    """Test correlation ID context manager."""
    # Clear any existing correlation ID
    clear_correlation_id()

    # Initially, no correlation ID should be set
    assert get_correlation_id() is None

    # Use context manager with explicit ID
    test_id = "test-correlation-id"
    with correlation_id_context(test_id):
        assert get_correlation_id() == test_id

    # After context, ID should be cleared
    assert get_correlation_id() is None

    # Use context manager with auto-generated ID
    with correlation_id_context():
        assert get_correlation_id() is not None
        assert isinstance(get_correlation_id(), str)

    # After context, ID should be cleared
    assert get_correlation_id() is None


def test_nested_correlation_id_context():
    """Test nested correlation ID context managers."""
    # Clear any existing correlation ID
    clear_correlation_id()

    # Use nested context managers
    outer_id = "outer-correlation-id"
    inner_id = "inner-correlation-id"

    with correlation_id_context(outer_id):
        assert get_correlation_id() == outer_id

        with correlation_id_context(inner_id):
            assert get_correlation_id() == inner_id

        # After inner context, outer ID should be restored
        assert get_correlation_id() == outer_id

    # After outer context, ID should be cleared
    assert get_correlation_id() is None


def test_with_correlation_id_decorator():
    """Test with_correlation_id decorator."""
    # Clear any existing correlation ID
    clear_correlation_id()

    # Define a decorated function
    @with_correlation_id
    def test_function():
        return get_correlation_id()

    # Call the function with no correlation ID set
    result = test_function()
    assert result is not None
    assert isinstance(result, str)

    # Set a correlation ID
    test_id = "test-correlation-id"
    set_correlation_id(test_id)

    # Call the function with correlation ID set
    result = test_function()
    assert result == test_id

    # Clear the correlation ID
    clear_correlation_id()


@pytest.mark.asyncio
async def test_async_correlation_id_context():
    """Test async correlation ID context manager."""
    # Clear any existing correlation ID
    clear_correlation_id()

    # Initially, no correlation ID should be set
    assert get_correlation_id() is None

    # Use context manager with explicit ID
    test_id = "test-correlation-id"
    async with async_correlation_id_context(test_id):
        assert get_correlation_id() == test_id

    # After context, ID should be cleared
    assert get_correlation_id() is None

    # Use context manager with auto-generated ID
    async with async_correlation_id_context():
        assert get_correlation_id() is not None
        assert isinstance(get_correlation_id(), str)

    # After context, ID should be cleared
    assert get_correlation_id() is None


@pytest.mark.asyncio
async def test_with_async_correlation_id_decorator():
    """Test with_async_correlation_id decorator."""
    # Clear any existing correlation ID
    clear_correlation_id()

    # Define a decorated function
    @with_async_correlation_id
    async def test_function():
        return get_correlation_id()

    # Call the function with no correlation ID set
    result = await test_function()
    assert result is not None
    assert isinstance(result, str)

    # Set a correlation ID
    test_id = "test-correlation-id"
    set_correlation_id(test_id)

    # Call the function with correlation ID set
    result = await test_function()
    assert result == test_id

    # Clear the correlation ID
    clear_correlation_id()


def test_get_correlation_id_from_request():
    """Test getting correlation ID from request."""
    # Create a mock request with correlation ID in state
    request = MagicMock()
    request.state.correlation_id = "state-correlation-id"
    assert get_correlation_id_from_request(request) == "state-correlation-id"

    # Create a mock request with correlation ID in headers
    request = MagicMock()
    request.state = MagicMock(spec=[])  # No correlation_id attribute
    request.headers = {CORRELATION_ID_HEADER: "header-correlation-id"}
    assert get_correlation_id_from_request(request) == "header-correlation-id"

    # Create a mock request with no correlation ID
    request = MagicMock()
    request.state = MagicMock(spec=[])  # No correlation_id attribute
    request.headers = {}
    correlation_id = get_correlation_id_from_request(request)
    assert correlation_id is not None
    assert isinstance(correlation_id, str)


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


def test_thread_isolation():
    """Test that correlation IDs are isolated between threads."""
    # Clear any existing correlation ID
    clear_correlation_id()

    # Set a correlation ID in the main thread
    main_id = "main-thread-id"
    set_correlation_id(main_id)

    # Create a thread that sets its own correlation ID
    thread_id = "thread-id"
    thread_result = {}

    def thread_func():
    """
    Thread func.
    
    """

        # Initially, no correlation ID should be set in the thread
        thread_result["initial"] = get_correlation_id()

        # Set a correlation ID in the thread
        set_correlation_id(thread_id)
        thread_result["after_set"] = get_correlation_id()

    thread = threading.Thread(target=thread_func)
    thread.start()
    thread.join()

    # Check that the thread had its own correlation ID
    assert thread_result["initial"] is None
    assert thread_result["after_set"] == thread_id

    # Check that the main thread's correlation ID is unchanged
    assert get_correlation_id() == main_id

    # Clear the correlation ID
    clear_correlation_id()