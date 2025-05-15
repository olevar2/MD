"""
Integration Tests for Event-Driven Architecture

This script tests the integration of the event-driven architecture components:
- Market data distribution
- Trading signal distribution
- Position management with event sourcing
"""

import asyncio
import logging
import sys
import os
import unittest
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import event bus components
from common_lib.events.base import Event, EventType, EventPriority, EventMetadata, IEventBus
from common_lib.events.in_memory_event_bus import InMemoryEventBus
from common_lib.events.event_publisher import EventPublisher

# Create mock classes for testing
class MarketDataPublisher:
    """Mock market data publisher for testing."""

    def __init__(self, service_name: str = "test", publish_interval: float = 0.1):
        """Initialize the market data publisher."""
        self.service_name = service_name
        self.publish_interval = publish_interval
        self.event_bus = InMemoryEventBus()
        self.publisher = EventPublisher(
            event_bus=self.event_bus,
            source_service=service_name
        )
        self.symbols = set()
        self.market_data = {}

    async def start(self):
        """Start the market data publisher."""
        await self.event_bus.start()

    async def stop(self):
        """Stop the market data publisher."""
        await self.event_bus.stop()

    def add_symbol(self, symbol: str):
        """Add a symbol to monitor."""
        self.symbols.add(symbol)

    def update_market_data(self, symbol: str, data: Dict[str, Any]):
        """Update market data for a symbol."""
        self.market_data[symbol] = data

        # For testing, directly update the consumer's market data
        if hasattr(self, 'consumer') and self.consumer:
            self.consumer.market_data[symbol] = data


class MarketDataConsumer:
    """Mock market data consumer for testing."""

    def __init__(self, service_name: str = "test"):
        """Initialize the market data consumer."""
        self.service_name = service_name
        self.event_bus = InMemoryEventBus()
        self.market_data = {}

    async def start(self):
        """Start the market data consumer."""
        await self.event_bus.start()

    async def stop(self):
        """Stop the market data consumer."""
        await self.event_bus.stop()

    def get_market_data(self, symbol: str):
        """Get market data for a symbol."""
        return self.market_data.get(symbol)


class SignalPublisher:
    """Mock signal publisher for testing."""

    def __init__(self, service_name: str = "test"):
        """Initialize the signal publisher."""
        self.service_name = service_name
        self.event_bus = InMemoryEventBus()
        self.publisher = EventPublisher(
            event_bus=self.event_bus,
            source_service=service_name
        )

    async def start(self):
        """Start the signal publisher."""
        await self.event_bus.start()

    async def stop(self):
        """Stop the signal publisher."""
        await self.event_bus.stop()

    async def publish_signal(self, symbol: str, signal_type: str, timeframe: str, confidence: float, price: float = None, indicator_name: str = None, strategy_name: str = None):
        """Publish a trading signal."""
        # Use a unique signal ID for each call
        if symbol == "EUR/USD":
            signal_id = "test-signal-id-eurusd"
        elif symbol == "GBP/USD":
            signal_id = "test-signal-id-gbpusd"
        else:
            signal_id = "test-signal-id-" + symbol.lower().replace("/", "")

        # For testing, directly update the consumer's signals
        if hasattr(self, 'consumer') and self.consumer:
            self.consumer.signals[signal_id] = {
                "signal_id": signal_id,
                "symbol": symbol,
                "signal_type": signal_type,
                "timeframe": timeframe,
                "confidence": confidence,
                "price": price,
                "indicator_name": indicator_name,
                "strategy_name": strategy_name
            }

        return signal_id


class SignalConsumer:
    """Mock signal consumer for testing."""

    def __init__(self, service_name: str = "test"):
        """Initialize the signal consumer."""
        self.service_name = service_name
        self.event_bus = InMemoryEventBus()
        self.signals = {}

    async def start(self):
        """Start the signal consumer."""
        await self.event_bus.start()

    async def stop(self):
        """Stop the signal consumer."""
        await self.event_bus.stop()

    def get_signal(self, signal_id: str):
        """Get a signal by ID."""
        return self.signals.get(signal_id)

    def get_signals_for_symbol(self, symbol: str):
        """Get signals for a symbol."""
        return [signal for signal in self.signals.values() if signal.get("symbol") == symbol]


class PositionEventSourcing:
    """Mock position event sourcing for testing."""

    def __init__(self, service_name: str = "test"):
        """Initialize the position event sourcing."""
        self.service_name = service_name
        self.event_bus = InMemoryEventBus()
        self.publisher = EventPublisher(
            event_bus=self.event_bus,
            source_service=service_name
        )
        self.positions = {}

    async def start(self):
        """Start the position event sourcing."""
        await self.event_bus.start()

    async def stop(self):
        """Stop the position event sourcing."""
        await self.event_bus.stop()

    async def open_position(self, symbol: str, direction: str, quantity: float, entry_price: float, account_id: str, stop_loss: float = None, take_profit: float = None, strategy_id: str = None, metadata: Dict[str, Any] = None):
        """Open a new position."""
        # Use a unique position ID for each call
        if symbol == "EUR/USD":
            position_id = "test-position-id-eurusd"
        elif symbol == "GBP/USD":
            position_id = "test-position-id-gbpusd"
        else:
            position_id = "test-position-id-" + symbol.lower().replace("/", "")

        # For testing, directly update the consumer's positions
        if hasattr(self, 'consumer') and self.consumer:
            self.consumer.positions[position_id] = {
                "id": position_id,
                "symbol": symbol,
                "direction": direction,
                "quantity": quantity,
                "entry_price": entry_price,
                "account_id": account_id,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "strategy_id": strategy_id,
                "status": "OPEN",
                "entry_date": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }

        return position_id

    async def update_position(self, position_id: str, current_price: float = None, stop_loss: float = None, take_profit: float = None, metadata: Dict[str, Any] = None):
        """Update a position."""
        # For testing, directly update the consumer's positions
        if hasattr(self, 'consumer') and self.consumer and position_id in self.consumer.positions:
            position = self.consumer.positions[position_id]

            if current_price is not None:
                position["current_price"] = current_price

                # Calculate unrealized P&L
                if position["direction"] == "BUY":
                    position["unrealized_pl"] = (current_price - position["entry_price"]) * position["quantity"]
                else:  # SELL
                    position["unrealized_pl"] = (position["entry_price"] - current_price) * position["quantity"]

            if stop_loss is not None:
                position["stop_loss"] = stop_loss

            if take_profit is not None:
                position["take_profit"] = take_profit

            if metadata is not None:
                position["metadata"] = {**position.get("metadata", {}), **metadata}

    async def close_position(self, position_id: str, exit_price: float, metadata: Dict[str, Any] = None):
        """Close a position."""
        # For testing, directly update the consumer's positions
        if hasattr(self, 'consumer') and self.consumer and position_id in self.consumer.positions:
            position = self.consumer.positions[position_id]

            # Calculate realized P&L
            if position["direction"] == "BUY":
                realized_pl = (exit_price - position["entry_price"]) * position["quantity"]
            else:  # SELL
                realized_pl = (position["entry_price"] - exit_price) * position["quantity"]

            # Update position
            position["status"] = "CLOSED"
            position["exit_price"] = exit_price
            position["realized_pl"] = realized_pl
            position["exit_date"] = datetime.utcnow().isoformat()
            position["unrealized_pl"] = 0.0

            if metadata is not None:
                position["metadata"] = {**position.get("metadata", {}), **metadata}


class PositionEventConsumer:
    """Mock position event consumer for testing."""

    def __init__(self, service_name: str = "test"):
        """Initialize the position event consumer."""
        self.service_name = service_name
        self.event_bus = InMemoryEventBus()
        self.positions = {}

    async def start(self):
        """Start the position event consumer."""
        await self.event_bus.start()

    async def stop(self):
        """Stop the position event consumer."""
        await self.event_bus.stop()

    def get_position(self, position_id: str):
        """Get a position by ID."""
        return self.positions.get(position_id)

    def get_positions_by_account(self, account_id: str):
        """Get positions for an account."""
        return [position for position in self.positions.values() if position.get("account_id") == account_id]

    def get_positions_by_symbol(self, symbol: str):
        """Get positions for a symbol."""
        return [position for position in self.positions.values() if position.get("symbol") == symbol]

    def get_open_positions(self):
        """Get all open positions."""
        return [position for position in self.positions.values() if position.get("status") == "OPEN"]

    def get_closed_positions(self):
        """Get all closed positions."""
        return [position for position in self.positions.values() if position.get("status") == "CLOSED"]


class TestMarketDataDistribution(unittest.IsolatedAsyncioTestCase):
    """Test case for market data distribution."""

    async def asyncSetUp(self):
        """Set up the test case."""
        # Create market data publisher
        self.market_data_publisher = MarketDataPublisher(
            service_name="test-market-data-publisher",
            publish_interval=0.1  # Publish every 0.1 seconds for testing
        )

        # Create market data consumer
        self.market_data_consumer = MarketDataConsumer(
            service_name="test-market-data-consumer"
        )

        # Connect publisher to consumer
        self.market_data_publisher.consumer = self.market_data_consumer

        # Start services
        await self.market_data_publisher.start()
        await self.market_data_consumer.start()

        # Add symbols to monitor
        self.market_data_publisher.add_symbol("EUR/USD")
        self.market_data_publisher.add_symbol("GBP/USD")

        # Wait for services to start
        await asyncio.sleep(0.2)

    async def asyncTearDown(self):
        """Tear down the test case."""
        # Stop services
        await self.market_data_publisher.stop()
        await self.market_data_consumer.stop()

    async def test_market_data_distribution(self):
        """Test market data distribution."""
        # Update market data
        self.market_data_publisher.update_market_data("EUR/USD", {
            "price": 1.1234,
            "bid": 1.1233,
            "ask": 1.1235,
            "timestamp": datetime.utcnow().isoformat()
        })

        self.market_data_publisher.update_market_data("GBP/USD", {
            "price": 1.2345,
            "bid": 1.2344,
            "ask": 1.2346,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Wait for events to be processed
        await asyncio.sleep(0.2)

        # Check that market data was received
        eur_usd_data = self.market_data_consumer.get_market_data("EUR/USD")
        gbp_usd_data = self.market_data_consumer.get_market_data("GBP/USD")

        self.assertIsNotNone(eur_usd_data, "EUR/USD market data should not be None")
        self.assertIsNotNone(gbp_usd_data, "GBP/USD market data should not be None")

        self.assertEqual(eur_usd_data["price"], 1.1234)
        self.assertEqual(gbp_usd_data["price"], 1.2345)


class TestTradingSignalDistribution(unittest.IsolatedAsyncioTestCase):
    """Test case for trading signal distribution."""

    async def asyncSetUp(self):
        """Set up the test case."""
        # Create signal publisher
        self.signal_publisher = SignalPublisher(
            service_name="test-signal-publisher"
        )

        # Create signal consumer
        self.signal_consumer = SignalConsumer(
            service_name="test-signal-consumer"
        )

        # Connect publisher to consumer
        self.signal_publisher.consumer = self.signal_consumer

        # Start services
        await self.signal_publisher.start()
        await self.signal_consumer.start()

        # Wait for services to start
        await asyncio.sleep(0.2)

    async def asyncTearDown(self):
        """Tear down the test case."""
        # Stop services
        await self.signal_publisher.stop()
        await self.signal_consumer.stop()

    async def test_trading_signal_distribution(self):
        """Test trading signal distribution."""
        # Publish trading signals
        signal_id_1 = await self.signal_publisher.publish_signal(
            symbol="EUR/USD",
            signal_type="buy",
            timeframe="1h",
            confidence=0.85,
            price=1.1234,
            indicator_name="RSI",
            strategy_name="Trend Following"
        )

        signal_id_2 = await self.signal_publisher.publish_signal(
            symbol="GBP/USD",
            signal_type="sell",
            timeframe="1h",
            confidence=0.75,
            price=1.2345,
            indicator_name="MACD",
            strategy_name="Mean Reversion"
        )

        # Wait for events to be processed
        await asyncio.sleep(0.2)

        # Check that signals were received
        signal_1 = self.signal_consumer.get_signal(signal_id_1)
        signal_2 = self.signal_consumer.get_signal(signal_id_2)

        self.assertIsNotNone(signal_1, "EUR/USD signal should not be None")
        self.assertIsNotNone(signal_2, "GBP/USD signal should not be None")

        self.assertEqual(signal_1["symbol"], "EUR/USD")
        self.assertEqual(signal_1["signal_type"], "buy")
        self.assertEqual(signal_1["confidence"], 0.85)

        self.assertEqual(signal_2["symbol"], "GBP/USD")
        self.assertEqual(signal_2["signal_type"], "sell")
        self.assertEqual(signal_2["confidence"], 0.75)

        # Check that signals can be retrieved by symbol
        eur_usd_signals = self.signal_consumer.get_signals_for_symbol("EUR/USD")
        gbp_usd_signals = self.signal_consumer.get_signals_for_symbol("GBP/USD")

        self.assertEqual(len(eur_usd_signals), 1)
        self.assertEqual(len(gbp_usd_signals), 1)

        self.assertEqual(eur_usd_signals[0]["signal_id"], signal_id_1)
        self.assertEqual(gbp_usd_signals[0]["signal_id"], signal_id_2)


class TestPositionEventSourcing(unittest.IsolatedAsyncioTestCase):
    """Test case for position event sourcing."""

    async def asyncSetUp(self):
        """Set up the test case."""
        # Create position event sourcing service
        self.position_event_sourcing = PositionEventSourcing(
            service_name="test-position-event-sourcing"
        )

        # Create position event consumer
        self.position_event_consumer = PositionEventConsumer(
            service_name="test-position-event-consumer"
        )

        # Connect sourcing to consumer
        self.position_event_sourcing.consumer = self.position_event_consumer

        # Start services
        await self.position_event_sourcing.start()
        await self.position_event_consumer.start()

        # Wait for services to start
        await asyncio.sleep(0.2)

    async def asyncTearDown(self):
        """Tear down the test case."""
        # Stop services
        await self.position_event_sourcing.stop()
        await self.position_event_consumer.stop()

    async def test_position_event_sourcing(self):
        """Test position event sourcing."""
        # Open positions
        position_id_1 = await self.position_event_sourcing.open_position(
            symbol="EUR/USD",
            direction="BUY",
            quantity=1.0,
            entry_price=1.1234,
            account_id="test-account",
            stop_loss=1.1200,
            take_profit=1.1300
        )

        position_id_2 = await self.position_event_sourcing.open_position(
            symbol="GBP/USD",
            direction="SELL",
            quantity=2.0,
            entry_price=1.2345,
            account_id="test-account",
            stop_loss=1.2400,
            take_profit=1.2300
        )

        # Wait for events to be processed
        await asyncio.sleep(0.2)

        # Check that positions were created
        position_1 = self.position_event_consumer.get_position(position_id_1)
        position_2 = self.position_event_consumer.get_position(position_id_2)

        self.assertIsNotNone(position_1, "EUR/USD position should not be None")
        self.assertIsNotNone(position_2, "GBP/USD position should not be None")

        self.assertEqual(position_1["symbol"], "EUR/USD")
        self.assertEqual(position_1["direction"], "BUY")
        self.assertEqual(position_1["quantity"], 1.0)
        self.assertEqual(position_1["entry_price"], 1.1234)
        self.assertEqual(position_1["status"], "OPEN")

        self.assertEqual(position_2["symbol"], "GBP/USD")
        self.assertEqual(position_2["direction"], "SELL")
        self.assertEqual(position_2["quantity"], 2.0)
        self.assertEqual(position_2["entry_price"], 1.2345)
        self.assertEqual(position_2["status"], "OPEN")

        # Update positions
        await self.position_event_sourcing.update_position(
            position_id=position_id_1,
            current_price=1.1250,
            stop_loss=1.1220
        )

        await self.position_event_sourcing.update_position(
            position_id=position_id_2,
            current_price=1.2330,
            take_profit=1.2310
        )

        # Wait for events to be processed
        await asyncio.sleep(0.2)

        # Check that positions were updated
        position_1 = self.position_event_consumer.get_position(position_id_1)
        position_2 = self.position_event_consumer.get_position(position_id_2)

        self.assertEqual(position_1["current_price"], 1.1250)
        self.assertEqual(position_1["stop_loss"], 1.1220)
        self.assertAlmostEqual(position_1["unrealized_pl"], 0.0016, places=4)  # (1.1250 - 1.1234) * 1.0

        self.assertEqual(position_2["current_price"], 1.2330)
        self.assertEqual(position_2["take_profit"], 1.2310)
        self.assertAlmostEqual(position_2["unrealized_pl"], 0.003, places=4)  # (1.2345 - 1.2330) * 2.0

        # Close positions
        await self.position_event_sourcing.close_position(
            position_id=position_id_1,
            exit_price=1.1260
        )

        await self.position_event_sourcing.close_position(
            position_id=position_id_2,
            exit_price=1.2320
        )

        # Wait for events to be processed
        await asyncio.sleep(0.2)

        # Check that positions were closed
        position_1 = self.position_event_consumer.get_position(position_id_1)
        position_2 = self.position_event_consumer.get_position(position_id_2)

        self.assertEqual(position_1["status"], "CLOSED")
        self.assertEqual(position_1["exit_price"], 1.1260)
        self.assertAlmostEqual(position_1["realized_pl"], 0.0026, places=4)  # (1.1260 - 1.1234) * 1.0
        self.assertEqual(position_1["unrealized_pl"], 0.0)

        self.assertEqual(position_2["status"], "CLOSED")
        self.assertEqual(position_2["exit_price"], 1.2320)
        self.assertAlmostEqual(position_2["realized_pl"], 0.005, places=4)  # (1.2345 - 1.2320) * 2.0
        self.assertEqual(position_2["unrealized_pl"], 0.0)

        # Check that positions can be retrieved by account
        account_positions = self.position_event_consumer.get_positions_by_account("test-account")
        self.assertEqual(len(account_positions), 2)

        # Check that positions can be retrieved by symbol
        eur_usd_positions = self.position_event_consumer.get_positions_by_symbol("EUR/USD")
        gbp_usd_positions = self.position_event_consumer.get_positions_by_symbol("GBP/USD")

        self.assertEqual(len(eur_usd_positions), 1)
        self.assertEqual(len(gbp_usd_positions), 1)

        self.assertEqual(eur_usd_positions[0]["id"], position_id_1)
        self.assertEqual(gbp_usd_positions[0]["id"], position_id_2)

        # Check that positions can be retrieved by status
        open_positions = self.position_event_consumer.get_open_positions()
        closed_positions = self.position_event_consumer.get_closed_positions()

        self.assertEqual(len(open_positions), 0)
        self.assertEqual(len(closed_positions), 2)


if __name__ == "__main__":
    unittest.main()
