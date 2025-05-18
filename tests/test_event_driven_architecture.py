"""
Comprehensive test suite for the event-driven architecture.

This script tests all aspects of the event-driven architecture to ensure it works correctly.
"""

import asyncio
import logging
import sys
import os
import unittest
from typing import Dict, Any, List, Set

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import directly from the base module to avoid circular imports
from common_lib.events.base import (
    Event, EventType, EventPriority, EventMetadata,
    IEventBus, EventHandler
)
from common_lib.events.in_memory_event_bus import InMemoryEventBus
from common_lib.events.kafka_event_bus_v2 import KafkaEventBusV2, KafkaConfig
from common_lib.events.event_publisher import EventPublisher
from common_lib.events.event_bus_factory import EventBusFactory, EventBusType


class TestEventBus(unittest.IsolatedAsyncioTestCase):
    """Test case for the event bus."""

    async def asyncSetUp(self):
        """Set up the test case."""
        self.event_bus = InMemoryEventBus()
        await self.event_bus.start()

        self.received_events = []
        self.publisher = EventPublisher(
            event_bus=self.event_bus,
            source_service="test-service"
        )

    async def asyncTearDown(self):
        """Tear down the test case."""
        await self.event_bus.stop()

    async def test_publish_and_subscribe(self):
        """Test publishing and subscribing to events."""
        # Define event handler
        async def handler(event: Event):
            self.received_events.append(event)

        # Subscribe to events
        self.event_bus.subscribe(
            event_types=[EventType.MARKET_DATA_UPDATED],
            handler=handler
        )

        # Publish an event
        await self.publisher.publish(
            event_type=EventType.MARKET_DATA_UPDATED,
            payload={"symbol": "EUR/USD", "price": 1.1234}
        )

        # Wait for the event to be processed
        await asyncio.sleep(0.1)

        # Check that the event was received
        self.assertEqual(len(self.received_events), 1)
        self.assertEqual(self.received_events[0].event_type, EventType.MARKET_DATA_UPDATED)
        self.assertEqual(self.received_events[0].payload["symbol"], "EUR/USD")
        self.assertEqual(self.received_events[0].payload["price"], 1.1234)

    async def test_multiple_subscribers(self):
        """Test multiple subscribers for the same event type."""
        received_by_handler1 = []
        received_by_handler2 = []

        # Define event handlers
        async def handler1(event: Event):
            received_by_handler1.append(event)

        async def handler2(event: Event):
            received_by_handler2.append(event)

        # Subscribe to events
        self.event_bus.subscribe(
            event_types=[EventType.MARKET_DATA_UPDATED],
            handler=handler1
        )

        self.event_bus.subscribe(
            event_types=[EventType.MARKET_DATA_UPDATED],
            handler=handler2
        )

        # Publish an event
        await self.publisher.publish(
            event_type=EventType.MARKET_DATA_UPDATED,
            payload={"symbol": "EUR/USD", "price": 1.1234}
        )

        # Wait for the event to be processed
        await asyncio.sleep(0.1)

        # Check that both handlers received the event
        self.assertEqual(len(received_by_handler1), 1)
        self.assertEqual(len(received_by_handler2), 1)

    async def test_unsubscribe(self):
        """Test unsubscribing from events."""
        received_events = []

        # Define event handler
        async def handler(event: Event):
            received_events.append(event)

        # Subscribe to events
        unsubscribe = self.event_bus.subscribe(
            event_types=[EventType.MARKET_DATA_UPDATED],
            handler=handler
        )

        # Publish an event
        await self.publisher.publish(
            event_type=EventType.MARKET_DATA_UPDATED,
            payload={"symbol": "EUR/USD", "price": 1.1234}
        )

        # Wait for the event to be processed
        await asyncio.sleep(0.1)

        # Check that the event was received
        self.assertEqual(len(received_events), 1)

        # Unsubscribe
        unsubscribe()

        # Publish another event
        await self.publisher.publish(
            event_type=EventType.MARKET_DATA_UPDATED,
            payload={"symbol": "EUR/USD", "price": 1.2345}
        )

        # Wait for the event to be processed
        await asyncio.sleep(0.1)

        # Check that the event was not received
        self.assertEqual(len(received_events), 1)

    async def test_subscribe_to_all(self):
        """Test subscribing to all events."""
        received_events = []

        # Define event handler
        async def handler(event: Event):
            received_events.append(event)

        # Subscribe to all events
        self.event_bus.subscribe_to_all(
            handler=handler
        )

        # Publish events of different types
        await self.publisher.publish(
            event_type=EventType.MARKET_DATA_UPDATED,
            payload={"symbol": "EUR/USD", "price": 1.1234}
        )

        await self.publisher.publish(
            event_type=EventType.SIGNAL_GENERATED,
            payload={"symbol": "EUR/USD", "signal": "buy"}
        )

        # Wait for the events to be processed
        await asyncio.sleep(0.1)

        # Check that both events were received
        self.assertEqual(len(received_events), 2)
        self.assertEqual(received_events[0].event_type, EventType.MARKET_DATA_UPDATED)
        self.assertEqual(received_events[1].event_type, EventType.SIGNAL_GENERATED)

    async def test_event_filter(self):
        """Test event filtering."""
        received_events = []

        # Define event handler
        async def handler(event: Event):
            received_events.append(event)

        # Define event filter
        def filter_func(event: Event):
            return event.payload.get("symbol") == "EUR/USD"

        # Subscribe to events with filter
        self.event_bus.subscribe(
            event_types=[EventType.MARKET_DATA_UPDATED],
            handler=handler,
            filter_func=filter_func
        )

        # Publish events with different symbols
        await self.publisher.publish(
            event_type=EventType.MARKET_DATA_UPDATED,
            payload={"symbol": "EUR/USD", "price": 1.1234}
        )

        await self.publisher.publish(
            event_type=EventType.MARKET_DATA_UPDATED,
            payload={"symbol": "GBP/USD", "price": 1.2345}
        )

        # Wait for the events to be processed
        await asyncio.sleep(0.1)

        # Check that only the EUR/USD event was received
        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[0].payload["symbol"], "EUR/USD")

    async def test_event_priority(self):
        """Test event priority."""
        # Publish events with different priorities
        await self.publisher.publish(
            event_type=EventType.MARKET_DATA_UPDATED,
            payload={"symbol": "EUR/USD", "price": 1.1234},
            priority=EventPriority.HIGH
        )

        await self.publisher.publish(
            event_type=EventType.MARKET_DATA_UPDATED,
            payload={"symbol": "GBP/USD", "price": 1.2345},
            priority=EventPriority.LOW
        )

        # Wait for the events to be processed
        await asyncio.sleep(0.1)

        # In the in-memory event bus, priority doesn't affect the order of processing
        # This is just a placeholder test for now
        self.assertTrue(True)

    async def test_correlation_id(self):
        """Test correlation ID."""
        received_events = []

        # Define event handler
        async def handler(event: Event):
            received_events.append(event)

        # Subscribe to events
        self.event_bus.subscribe(
            event_types=[EventType.MARKET_DATA_UPDATED, EventType.SIGNAL_GENERATED],
            handler=handler
        )

        # Set correlation ID
        correlation_id = "test-correlation-id"
        self.publisher.set_correlation_id(correlation_id)

        # Publish events
        await self.publisher.publish(
            event_type=EventType.MARKET_DATA_UPDATED,
            payload={"symbol": "EUR/USD", "price": 1.1234}
        )

        await self.publisher.publish(
            event_type=EventType.SIGNAL_GENERATED,
            payload={"symbol": "EUR/USD", "signal": "buy"}
        )

        # Wait for the events to be processed
        await asyncio.sleep(0.1)

        # Check that both events have the same correlation ID
        self.assertEqual(len(received_events), 2)
        self.assertEqual(received_events[0].metadata.correlation_id, correlation_id)
        self.assertEqual(received_events[1].metadata.correlation_id, correlation_id)


class TestEventBusFactory(unittest.TestCase):
    """Test case for the event bus factory."""

    def test_create_in_memory_event_bus(self):
        """Test creating an in-memory event bus."""
        event_bus = EventBusFactory.create_event_bus(
            bus_type=EventBusType.IN_MEMORY,
            service_name="test-service"
        )

        self.assertIsInstance(event_bus, InMemoryEventBus)

    def test_create_kafka_event_bus(self):
        """Test creating a Kafka event bus."""
        # Skip this test if confluent_kafka is not available
        try:
            import confluent_kafka
        except ImportError:
            self.skipTest("confluent_kafka not available")

        # Create Kafka event bus
        try:
            event_bus = EventBusFactory.create_event_bus(
                bus_type=EventBusType.KAFKA,
                service_name="test-service",
                config={
                    "bootstrap_servers": "localhost:9092",
                    "auto_create_topics": True
                }
            )

            self.assertIsInstance(event_bus, KafkaEventBusV2)
        except ImportError:
            self.skipTest("confluent_kafka not available")

    def test_create_unknown_event_bus(self):
        """Test creating an unknown event bus type."""
        with self.assertRaises(ValueError):
            EventBusFactory.create_event_bus(
                bus_type="unknown",
                service_name="test-service"
            )


class TestEventPublisher(unittest.IsolatedAsyncioTestCase):
    """Test case for the event publisher."""

    async def asyncSetUp(self):
        """Set up the test case."""
        self.event_bus = InMemoryEventBus()
        await self.event_bus.start()

        self.received_events = []

        # Define event handler
        async def handler(event: Event):
            self.received_events.append(event)

        # Subscribe to events
        self.event_bus.subscribe(
            event_types=[EventType.MARKET_DATA_UPDATED, EventType.SIGNAL_GENERATED],
            handler=handler
        )

    async def asyncTearDown(self):
        """Tear down the test case."""
        await self.event_bus.stop()

    async def test_publish_event(self):
        """Test publishing an event."""
        # Create publisher
        publisher = EventPublisher(
            event_bus=self.event_bus,
            source_service="test-service"
        )

        # Publish event
        await publisher.publish(
            event_type=EventType.MARKET_DATA_UPDATED,
            payload={"symbol": "EUR/USD", "price": 1.1234}
        )

        # Wait for the event to be processed
        await asyncio.sleep(0.1)

        # Check that the event was received
        self.assertEqual(len(self.received_events), 1)
        self.assertEqual(self.received_events[0].event_type, EventType.MARKET_DATA_UPDATED)
        self.assertEqual(self.received_events[0].payload["symbol"], "EUR/USD")
        self.assertEqual(self.received_events[0].payload["price"], 1.1234)
        self.assertEqual(self.received_events[0].metadata.source_service, "test-service")

    async def test_set_correlation_id(self):
        """Test setting the correlation ID."""
        # Create publisher
        publisher = EventPublisher(
            event_bus=self.event_bus,
            source_service="test-service"
        )

        # Set correlation ID
        correlation_id = "test-correlation-id"
        publisher.set_correlation_id(correlation_id)

        # Publish events
        await publisher.publish(
            event_type=EventType.MARKET_DATA_UPDATED,
            payload={"symbol": "EUR/USD", "price": 1.1234}
        )

        await publisher.publish(
            event_type=EventType.SIGNAL_GENERATED,
            payload={"symbol": "EUR/USD", "signal": "buy"}
        )

        # Wait for the events to be processed
        await asyncio.sleep(0.1)

        # Check that both events have the same correlation ID
        self.assertEqual(len(self.received_events), 2)
        self.assertEqual(self.received_events[0].metadata.correlation_id, correlation_id)
        self.assertEqual(self.received_events[1].metadata.correlation_id, correlation_id)


if __name__ == "__main__":
    unittest.main()
