"""
Integration tests for correlation ID propagation in event-based communication.

These tests verify that correlation IDs are correctly propagated in event-based communication.
"""

import asyncio
import pytest
import uuid
import logging
from unittest.mock import MagicMock, patch

from common_lib.correlation import (
    get_correlation_id,
    set_correlation_id,
    clear_correlation_id,
    add_correlation_to_event_metadata,
    extract_correlation_id_from_event,
    with_event_correlation,
    with_async_event_correlation
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create a logger that includes correlation ID
logger = logging.getLogger("event_correlation_test")


class CorrelationFilter(logging.Filter):
    """Logging filter that adds correlation ID to log records."""

    def filter(self, record):
        correlation_id = get_correlation_id()
        record.correlation_id = correlation_id or "no-correlation-id"
        return True


# Add the correlation filter to the logger
logger.addFilter(CorrelationFilter())


# Mock event bus
class MockEventBus:
    """Mock event bus for testing."""

    def __init__(self):
        """Initialize the mock event bus."""
        self.published_events = []
        self.handlers = {}

    def publish(self, event_type, event_data, metadata=None):
        """Publish an event to the event bus."""
        # Add correlation ID to metadata
        metadata = add_correlation_to_event_metadata(metadata or {})

        # Create the event
        event = {
            "event_type": event_type,
            "data": event_data,
            "metadata": metadata
        }

        # Add to published events
        self.published_events.append(event)

        # Call handlers
        if event_type in self.handlers:
            for handler in self.handlers[event_type]:
                handler(event)

        return event

    async def publish_async(self, event_type, event_data, metadata=None):
        """Publish an event to the event bus asynchronously."""
        # Add correlation ID to metadata
        metadata = add_correlation_to_event_metadata(metadata or {})

        # Create the event
        event = {
            "event_type": event_type,
            "data": event_data,
            "metadata": metadata
        }

        # Add to published events
        self.published_events.append(event)

        # Call handlers
        if event_type in self.handlers:
            for handler in self.handlers[event_type]:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)

        return event

    def subscribe(self, event_type, handler):
        """Subscribe to an event type."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []

        self.handlers[event_type].append(handler)

    def unsubscribe(self, event_type, handler):
        """Unsubscribe from an event type."""
        if event_type in self.handlers:
            if handler in self.handlers[event_type]:
                self.handlers[event_type].remove(handler)


# Mock event producer
class EventProducer:
    """Mock event producer for testing."""

    def __init__(self, event_bus):
        """Initialize the event producer."""
        self.event_bus = event_bus

    def create_resource(self, resource_data):
        """Create a resource and publish an event."""
        # Create a resource
        resource_id = str(uuid.uuid4())
        resource = {
            "id": resource_id,
            "name": resource_data.get("name", f"Resource {resource_id}"),
            "created_at": "2023-01-01T00:00:00Z"
        }

        # Publish an event
        event = self.event_bus.publish(
            "resource.created",
            resource
        )

        return resource

    async def update_resource(self, resource_id, resource_data):
        """Update a resource and publish an event."""
        # Update a resource
        resource = {
            "id": resource_id,
            "name": resource_data.get("name", f"Resource {resource_id}"),
            "updated_at": "2023-01-01T00:00:00Z"
        }

        # Publish an event
        event = await self.event_bus.publish_async(
            "resource.updated",
            resource
        )

        return resource


# Mock event consumer
class EventConsumer:
    """Mock event consumer for testing."""

    def __init__(self, event_bus):
        """Initialize the event consumer."""
        self.event_bus = event_bus
        self.processed_events = []

        # Subscribe to events
        self.event_bus.subscribe("resource.created", self.handle_resource_created)
        self.event_bus.subscribe("resource.updated", self.handle_resource_updated)

    @with_event_correlation
    def handle_resource_created(self, event):
        """Handle a resource.created event."""
        # Log the event
        logger.info(f"Handling resource.created event: {event['data']['id']}")

        # Get the correlation ID
        correlation_id = get_correlation_id()

        # Process the event
        processed_event = {
            "event_type": event["event_type"],
            "data": event["data"],
            "correlation_id": correlation_id,
            "processed_at": "2023-01-01T00:00:00Z"
        }

        # Add to processed events
        self.processed_events.append(processed_event)

        return processed_event

    @with_async_event_correlation
    async def handle_resource_updated(self, event):
        """Handle a resource.updated event."""
        # Log the event
        logger.info(f"Handling resource.updated event: {event['data']['id']}")

        # Get the correlation ID
        correlation_id = get_correlation_id()

        # Process the event
        processed_event = {
            "event_type": event["event_type"],
            "data": event["data"],
            "correlation_id": correlation_id,
            "processed_at": "2023-01-01T00:00:00Z"
        }

        # Add to processed events
        self.processed_events.append(processed_event)

        return processed_event


@pytest.fixture
def event_bus():
    """Fixture for the mock event bus."""
    return MockEventBus()


@pytest.fixture
def event_producer(event_bus):
    """Fixture for the mock event producer."""
    return EventProducer(event_bus)


@pytest.fixture
def event_consumer(event_bus):
    """Fixture for the mock event consumer."""
    return EventConsumer(event_bus)


def test_correlation_id_propagation_sync(event_producer, event_consumer):
    """Test correlation ID propagation in synchronous event handling."""
    # Clear any existing correlation ID
    clear_correlation_id()

    # Set a correlation ID
    correlation_id = str(uuid.uuid4())
    set_correlation_id(correlation_id)

    # Create a resource
    resource = event_producer.create_resource({"name": "Test Resource"})

    # Check that the event was processed
    assert len(event_consumer.processed_events) == 1
    processed_event = event_consumer.processed_events[0]

    # Check that the correlation ID was propagated
    assert processed_event["correlation_id"] == correlation_id
    assert processed_event["event_type"] == "resource.created"
    assert processed_event["data"]["id"] == resource["id"]


@pytest.mark.asyncio
async def test_correlation_id_propagation_async(event_producer, event_consumer):
    """Test correlation ID propagation in asynchronous event handling."""
    # Clear any existing correlation ID
    clear_correlation_id()

    # Set a correlation ID
    correlation_id = str(uuid.uuid4())
    set_correlation_id(correlation_id)

    # Update a resource
    resource_id = str(uuid.uuid4())
    resource = await event_producer.update_resource(resource_id, {"name": "Test Resource"})

    # Check that the event was processed
    assert len(event_consumer.processed_events) == 1
    processed_event = event_consumer.processed_events[0]

    # Check that the correlation ID was propagated
    assert processed_event["correlation_id"] == correlation_id
    assert processed_event["event_type"] == "resource.updated"
    assert processed_event["data"]["id"] == resource["id"]


def test_correlation_id_extraction(event_bus):
    """Test correlation ID extraction from events."""
    # Clear any existing correlation ID
    clear_correlation_id()

    # Create a correlation ID
    correlation_id = str(uuid.uuid4())

    # Create an event with the correlation ID
    metadata = add_correlation_to_event_metadata({}, correlation_id)
    event = {
        "event_type": "test.event",
        "data": {"key": "value"},
        "metadata": metadata
    }

    # Extract the correlation ID
    extracted_id = extract_correlation_id_from_event(event)

    # Check that the correlation ID was extracted
    assert extracted_id == correlation_id
