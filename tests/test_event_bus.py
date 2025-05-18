"""
Test script for the event bus implementation.
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import directly from the modules to avoid circular imports
from common_lib.events.event_schema import Event, EventType, EventPriority
from common_lib.events.event_bus_interface import IEventBus, EventHandler, EventFilter
from common_lib.events.in_memory_event_bus import InMemoryEventBus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create a simple event handler
async def event_handler(event: Event) -> None:
    logger.info(f"Received event: {event.event_type}")
    logger.info(f"Payload: {event.payload}")

# Create a simple event publisher
class SimpleEventPublisher:
    def __init__(self, event_bus: InMemoryEventBus, source: str):
        self.event_bus = event_bus
        self.source = source

    async def publish(self, event_type: str, payload: Dict[str, Any]) -> None:
        # Create an event with the given type and payload
        event = Event(event_type=event_type, payload=payload)
        # Publish the event
        await self.event_bus.publish(event)

# Example service using our implementation
class TestService:
    def __init__(self, name: str):
        self.name = name
        self.event_bus = InMemoryEventBus()
        self.publisher = SimpleEventPublisher(self.event_bus, name)

    async def start(self) -> None:
        # Start the event bus
        await self.event_bus.start()

        # Subscribe to events
        self.event_bus.subscribe(
            event_types=["market_data_updated"],
            handler=event_handler
        )

        self.event_bus.subscribe(
            event_types=["trading_signal"],
            handler=event_handler
        )

        logger.info(f"Service {self.name} started")

    async def stop(self) -> None:
        # Stop the event bus
        await self.event_bus.stop()
        logger.info(f"Service {self.name} stopped")

    async def publish_market_data(self, symbol: str, price: float) -> None:
        # Create payload
        payload = {
            "symbol": symbol,
            "price": price,
            "timestamp": "2023-01-01T12:00:00Z"
        }

        # Publish event
        await self.publisher.publish("market_data_updated", payload)

    async def publish_trading_signal(self, symbol: str, signal: str) -> None:
        # Create payload
        payload = {
            "symbol": symbol,
            "signal": signal,
            "confidence": 0.85
        }

        # Publish event
        await self.publisher.publish("trading_signal", payload)

# Main function
async def main():
    # Create service
    service = TestService("test-service")

    # Start service
    await service.start()

    try:
        # Publish events
        logger.info("Publishing market data event...")
        await service.publish_market_data("EUR/USD", 1.1234)
        await asyncio.sleep(1)

        logger.info("Publishing trading signal event...")
        await service.publish_trading_signal("EUR/USD", "buy")
        await asyncio.sleep(1)

    finally:
        # Stop service
        await service.stop()

# Run the main function
if __name__ == "__main__":
    logger.info("Starting test...")
    asyncio.run(main())
    logger.info("Test completed.")
