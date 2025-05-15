"""
Event-Driven Architecture Example

This example demonstrates how to use the event-driven architecture in the Forex Trading Platform.
"""

import asyncio
import logging
import uuid
from typing import Dict, Any

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_lib.events.event_schema import Event, EventType, EventPriority
from common_lib.events.event_bus_interface import IEventBus
from common_lib.events.event_bus_factory import EventBusFactory, EventBusType
from common_lib.events.event_publisher import EventPublisher, publish_event


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Example event handlers
async def market_data_handler(event: Event) -> None:
    """
    Handle market data events.

    Args:
        event: Market data event
    """
    logger.info(f"Received market data event: {event.event_type}")
    logger.info(f"  Symbol: {event.payload.get('symbol')}")
    logger.info(f"  Price: {event.payload.get('price')}")
    logger.info(f"  Timestamp: {event.payload.get('timestamp')}")


async def trading_signal_handler(event: Event) -> None:
    """
    Handle trading signal events.

    Args:
        event: Trading signal event
    """
    logger.info(f"Received trading signal event: {event.event_type}")
    logger.info(f"  Symbol: {event.payload.get('symbol')}")
    logger.info(f"  Signal: {event.payload.get('signal')}")
    logger.info(f"  Confidence: {event.payload.get('confidence')}")


# Example service that publishes and subscribes to events
class ExampleService:
    """
    Example service that demonstrates event-driven architecture.
    """

    def __init__(self, service_name: str):
        """
        Initialize the example service.

        Args:
            service_name: Name of the service
        """
        self.service_name = service_name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Create event bus
        self.event_bus = EventBusFactory.create_event_bus(
            bus_type=EventBusType.IN_MEMORY,
            service_name=service_name
        )

        # Create event publisher
        self.publisher = EventPublisher(
            event_bus=self.event_bus,
            source_service=service_name
        )

        # Set correlation ID for all events
        self.publisher.set_correlation_id(str(uuid.uuid4()))

    async def start(self) -> None:
        """
        Start the service.
        """
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

        self.logger.info(f"Service {self.service_name} started")

    async def stop(self) -> None:
        """
        Stop the service.
        """
        # Stop the event bus
        await self.event_bus.stop()

        self.logger.info(f"Service {self.service_name} stopped")

    async def publish_market_data(
        self,
        symbol: str,
        price: float,
        timestamp: str
    ) -> None:
        """
        Publish market data.

        Args:
            symbol: Symbol
            price: Price
            timestamp: Timestamp
        """
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
        """
        Publish trading signal.

        Args:
            symbol: Symbol
            signal: Signal (buy, sell, hold)
            confidence: Confidence level (0-1)
        """
        payload = {
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence
        }

        await self.publisher.publish(
            event_type=EventType.SIGNAL_GENERATED,
            payload=payload
        )


# Example usage
async def main():
    """
    Main function.
    """
    # Create example service
    service = ExampleService("example-service")

    # Start the service
    await service.start()

    try:
        # Publish some events
        await service.publish_market_data(
            symbol="EUR/USD",
            price=1.1234,
            timestamp="2023-01-01T12:00:00Z"
        )

        await asyncio.sleep(1)

        await service.publish_trading_signal(
            symbol="EUR/USD",
            signal="buy",
            confidence=0.85
        )

        await asyncio.sleep(1)

    finally:
        # Stop the service
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
