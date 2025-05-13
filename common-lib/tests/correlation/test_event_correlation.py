"""
Tests for event correlation functionality.
"""

import pytest
from unittest.mock import MagicMock, patch

from common_lib.correlation import (
    add_correlation_to_event_metadata,
    extract_correlation_id_from_event,
    with_event_correlation,
    with_async_event_correlation,
    get_correlation_id,
    set_correlation_id,
    clear_correlation_id
)


def test_add_correlation_to_event_metadata():
    """Test adding correlation ID to event metadata."""
    # Clear any existing correlation ID
    clear_correlation_id()

    # Test with empty metadata and no correlation ID
    metadata = {}
    updated_metadata = add_correlation_to_event_metadata(metadata)
    assert "correlation_id" in updated_metadata
    assert isinstance(updated_metadata["correlation_id"], str)

    # Test with existing metadata and no correlation ID
    metadata = {"source": "test"}
    updated_metadata = add_correlation_to_event_metadata(metadata)
    assert "correlation_id" in updated_metadata
    assert "source" in updated_metadata
    assert updated_metadata["source"] == "test"

    # Test with correlation ID in context
    test_id = "test-correlation-id"
    set_correlation_id(test_id)
    updated_metadata = add_correlation_to_event_metadata({})
    assert updated_metadata["correlation_id"] == test_id

    # Test with explicit correlation ID
    explicit_id = "explicit-correlation-id"
    updated_metadata = add_correlation_to_event_metadata({}, correlation_id=explicit_id)
    assert updated_metadata["correlation_id"] == explicit_id

    # Test with causation ID
    causation_id = "causation-id"
    updated_metadata = add_correlation_to_event_metadata({}, causation_id=causation_id)
    assert "causation_id" in updated_metadata
    assert updated_metadata["causation_id"] == causation_id

    # Clear the correlation ID
    clear_correlation_id()


def test_extract_correlation_id_from_event():
    """Test extracting correlation ID from event."""
    # Test with correlation ID in metadata
    correlation_id = "test-correlation-id"
    event = {
        "metadata": {
            "correlation_id": correlation_id
        }
    }
    assert extract_correlation_id_from_event(event) == correlation_id

    # Test with correlation ID at top level
    event = {
        "correlation_id": correlation_id
    }
    assert extract_correlation_id_from_event(event) == correlation_id

    # Test with no correlation ID
    event = {
        "metadata": {}
    }
    assert extract_correlation_id_from_event(event) is None

    # Test with empty event
    event = {}
    assert extract_correlation_id_from_event(event) is None


def test_with_event_correlation_decorator():
    """Test with_event_correlation decorator."""
    # Clear any existing correlation ID
    clear_correlation_id()

    # Define a decorated function
    @with_event_correlation
    def test_handler(event):
        return get_correlation_id()

    # Call the handler with correlation ID in event
    correlation_id = "test-correlation-id"
    event = {
        "metadata": {
            "correlation_id": correlation_id
        }
    }
    result = test_handler(event)
    assert result == correlation_id

    # Call the handler with no correlation ID in event
    event = {}
    result = test_handler(event)
    assert result is not None
    assert isinstance(result, str)

    # Clear the correlation ID
    clear_correlation_id()


@pytest.mark.asyncio
async def test_with_async_event_correlation_decorator():
    """Test with_async_event_correlation decorator."""
    # Clear any existing correlation ID
    clear_correlation_id()

    # Define a decorated function
    @with_async_event_correlation
    async def test_handler(event):
        return get_correlation_id()

    # Call the handler with correlation ID in event
    correlation_id = "test-correlation-id"
    event = {
        "metadata": {
            "correlation_id": correlation_id
        }
    }
    result = await test_handler(event)
    assert result == correlation_id

    # Call the handler with no correlation ID in event
    event = {}
    result = await test_handler(event)
    assert result is not None
    assert isinstance(result, str)

    # Clear the correlation ID
    clear_correlation_id()


def test_event_correlation_integration():
    """Test integration of event correlation components."""
    # Clear any existing correlation ID
    clear_correlation_id()

    # Create an event with correlation ID
    correlation_id = "test-correlation-id"
    metadata = add_correlation_to_event_metadata({}, correlation_id)
    event = {"metadata": metadata}

    # Extract correlation ID from event
    extracted_id = extract_correlation_id_from_event(event)
    assert extracted_id == correlation_id

    # Use the decorator to handle the event
    @with_event_correlation
    def test_handler(event):
        # The decorator should set the correlation ID in the context
        return get_correlation_id()

    result = test_handler(event)
    assert result == correlation_id

    # Clear the correlation ID
    clear_correlation_id()