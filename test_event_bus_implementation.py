"""
Test script for the event bus implementation.

This script tests the event bus implementation to ensure it works correctly.
"""

import asyncio
import logging
import sys
import os
from typing import Dict, Any

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
from common_lib.events.event_publisher import EventPublisher


# Event handlers
async def market_data_handler(event: Event) -> None:
    """Handle market data events."""
    logger.info(f"Received market data event: {event.event_type}")
    logger.info(f"  Symbol: {event.payload.get('symbol')}")
    logger.info(f"  Price: {event.payload.get('price')}")
    logger.info(f"  Timestamp: {event.payload.get('timestamp')}")


async def trading_signal_handler(event: Event) -> None:
    """Handle trading signal events."""
    logger.info(f"Received trading signal event: {event.event_type}")
    logger.info(f"  Symbol: {event.payload.get('symbol')}")
    logger.info(f"  Signal: {event.payload.get('signal')}")
    logger.info(f"  Confidence: {event.payload.get('confidence')}")


# Example service
class TestService:
    """Test service for the event bus implementation."""

    def __init__(self, name: str):
        """Initialize the test service."""
        self.name = name
        self.event_bus = InMemoryEventBus()
        self.publisher = EventPublisher(
            event_bus=self.event_bus,
            source_service=name
        )

    async def start(self) -> None:
        """Start the service."""
        # Start the event bus
        await self.event_bus.start()

        # Subscribe to events
        self.event_bus.subscribe(
            event_types=[EventType.MARKET_DATA_UPDATED],
            handler=market_data_handler
        )

        self.event_bus.subscribe(
            event_types=[EventType.SIGNAL_GENERATED],
            handler=trading_signal_handler
        )

        logger.info(f"Service {self.name} started")

    async def stop(self) -> None:
        """Stop the service."""
        # Stop the event bus
        await self.event_bus.stop()

        logger.info(f"Service {self.name} stopped")

    async def publish_market_data(
        self,
        symbol: str,
        price: float,
        timestamp: str
    ) -> None:
        """Publish market data."""
        payload = {
            "symbol": symbol,
            "price": price,
            "timestamp": timestamp
        }

        await self.publisher.publish(
            event_type=EventType.MARKET_DATA_UPDATED,
            payload=payload,
            priority=EventPriority.HIGH
        )

    async def publish_trading_signal(
        self,
        symbol: str,
        signal: str,
        confidence: float
    ) -> None:
        """Publish trading signal."""
        payload = {
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence
        }

        await self.publisher.publish(
            event_type=EventType.SIGNAL_GENERATED,
            payload=payload
        )


# Main function
async def main():
    """Main function."""
    logger.info("Starting test...")

    # Create test service
    service = TestService("test-service")

    # Start the service
    await service.start()

    try:
        # Publish market data
        logger.info("Publishing market data...")
        await service.publish_market_data(
            symbol="EUR/USD",
            price=1.1234,
            timestamp="2023-01-01T12:00:00Z"
        )

        # Wait for handlers to process
        await asyncio.sleep(1)

        # Publish trading signal
        logger.info("Publishing trading signal...")
        await service.publish_trading_signal(
            symbol="EUR/USD",
            signal="buy",
            confidence=0.85
        )

        # Wait for handlers to process
        await asyncio.sleep(1)

    finally:
        # Stop the service
        await service.stop()

    logger.info("Test completed successfully!")


# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
