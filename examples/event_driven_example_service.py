"""
Example service that demonstrates how to use the event-driven architecture.

This script shows how to create a service that publishes and subscribes to events.
"""

import asyncio
import logging
import sys
import os
import signal
import random
from typing import Dict, Any, List, Set

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
from common_lib.events.event_bus_factory import EventBusFactory, EventBusType


class MarketDataService:
    """Service that publishes market data events."""

    def __init__(self, event_bus: IEventBus):
        """Initialize the market data service."""
        self.event_bus = event_bus
        self.publisher = EventPublisher(
            event_bus=event_bus,
            source_service="market-data-service"
        )
        self.running = False
        self.symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]

    async def start(self) -> None:
        """Start the service."""
        self.running = True
        logger.info("Market data service started")

    async def stop(self) -> None:
        """Stop the service."""
        self.running = False
        logger.info("Market data service stopped")

    async def publish_market_data(self) -> None:
        """Publish market data for all symbols."""
        while self.running:
            for symbol in self.symbols:
                # Generate random price
                price = round(random.uniform(1.0, 2.0), 4)

                # Publish market data event
                await self.publisher.publish(
                    event_type=EventType.MARKET_DATA_UPDATED,
                    payload={
                        "symbol": symbol,
                        "price": price,
                        "timestamp": "2023-01-01T12:00:00Z"
                    },
                    priority=EventPriority.HIGH
                )

                logger.info(f"Published market data for {symbol}: {price}")

            # Wait before publishing next batch
            await asyncio.sleep(2)


class SignalGeneratorService:
    """Service that subscribes to market data events and generates trading signals."""

    def __init__(self, event_bus: IEventBus):
        """Initialize the signal generator service."""
        self.event_bus = event_bus
        self.publisher = EventPublisher(
            event_bus=event_bus,
            source_service="signal-generator-service"
        )
        # Set a correlation ID for all events published by this service
        self.publisher.set_correlation_id(f"signal-generator-{id(self)}")
        self.last_prices: Dict[str, float] = {}

    async def start(self) -> None:
        """Start the service."""
        # Subscribe to market data events
        self.event_bus.subscribe(
            event_types=[EventType.MARKET_DATA_UPDATED],
            handler=self.handle_market_data
        )

        logger.info("Signal generator service started")

    async def stop(self) -> None:
        """Stop the service."""
        logger.info("Signal generator service stopped")

    async def handle_market_data(self, event: Event) -> None:
        """Handle market data events."""
        symbol = event.payload.get("symbol")
        price = event.payload.get("price")

        if symbol and price:
            # Get last price
            last_price = self.last_prices.get(symbol)

            # Update last price
            self.last_prices[symbol] = price

            # Generate signal if we have a previous price
            if last_price:
                # Simple strategy: buy if price increased, sell if decreased
                if price > last_price:
                    signal = "buy"
                    confidence = min(1.0, (price - last_price) * 10)
                else:
                    signal = "sell"
                    confidence = min(1.0, (last_price - price) * 10)

                # Publish signal event
                await self.publisher.publish(
                    event_type=EventType.SIGNAL_GENERATED,
                    payload={
                        "symbol": symbol,
                        "signal": signal,
                        "confidence": round(confidence, 2)
                    },
                    # Set the causation ID to the market data event ID
                    causation_id=event.metadata.event_id
                )

                # Note: The correlation ID is set at the publisher level
                # using publisher.set_correlation_id() in the constructor

                logger.info(f"Generated {signal} signal for {symbol} with confidence {confidence:.2f}")


class TradingService:
    """Service that subscribes to trading signals and executes trades."""

    def __init__(self, event_bus: IEventBus):
        """Initialize the trading service."""
        self.event_bus = event_bus
        self.publisher = EventPublisher(
            event_bus=event_bus,
            source_service="trading-service"
        )
        # Set a correlation ID for all events published by this service
        self.publisher.set_correlation_id(f"trading-service-{id(self)}")
        self.positions: Dict[str, str] = {}  # symbol -> position (buy/sell)

    async def start(self) -> None:
        """Start the service."""
        # Subscribe to trading signal events
        self.event_bus.subscribe(
            event_types=[EventType.SIGNAL_GENERATED],
            handler=self.handle_trading_signal
        )

        logger.info("Trading service started")

    async def stop(self) -> None:
        """Stop the service."""
        logger.info("Trading service stopped")

    async def handle_trading_signal(self, event: Event) -> None:
        """Handle trading signal events."""
        symbol = event.payload.get("symbol")
        signal = event.payload.get("signal")
        confidence = event.payload.get("confidence", 0.0)

        if symbol and signal and confidence > 0.5:
            # Check if we already have a position
            current_position = self.positions.get(symbol)

            # Only trade if we don't have a position or if the signal is different
            if current_position != signal:
                # Update position
                self.positions[symbol] = signal

                # Log the trade
                logger.info(f"Executed {signal} trade for {symbol} with confidence {confidence:.2f}")

                # Publish order event (in a real system, this would be more complex)
                await self.publisher.publish(
                    event_type=EventType.ORDER_CREATED,
                    payload={
                        "symbol": symbol,
                        "order_type": "market",
                        "side": signal,
                        "quantity": 1.0,
                        "status": "filled"
                    },
                    # Set the causation ID to the signal event ID
                    causation_id=event.metadata.event_id
                )

                # Note: The correlation ID is set at the publisher level
                # using publisher.set_correlation_id() in the constructor


async def main():
    """Main function."""
    # Create event bus
    event_bus = EventBusFactory.create_event_bus(
        bus_type=EventBusType.IN_MEMORY,
        service_name="example-service"
    )

    # Start event bus
    await event_bus.start()

    try:
        # Create services
        market_data_service = MarketDataService(event_bus)
        signal_generator_service = SignalGeneratorService(event_bus)
        trading_service = TradingService(event_bus)

        # Start services
        await market_data_service.start()
        await signal_generator_service.start()
        await trading_service.start()

        # Start publishing market data
        market_data_task = asyncio.create_task(market_data_service.publish_market_data())

        # Run for 30 seconds
        await asyncio.sleep(30)

        # Cancel market data task
        market_data_task.cancel()

        # Stop services
        await market_data_service.stop()
        await signal_generator_service.stop()
        await trading_service.stop()

    finally:
        # Stop event bus
        await event_bus.stop()


if __name__ == "__main__":
    try:
        # Run the main function
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application cancelled by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        logger.info("Application stopped")
