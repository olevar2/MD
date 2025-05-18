"""
Simplified Integration Tests for Event-Driven Architecture

This script tests the integration of the event-driven architecture components using
the base event bus implementation directly.
"""

import asyncio
import logging
import sys
import os
import unittest
from datetime import datetime
from typing import Dict, Any, List, Optional, Set

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


class MarketDataPublisher:
    """Simplified market data publisher for testing."""
    
    def __init__(self, event_bus: IEventBus, source_service: str):
        """Initialize the market data publisher."""
        self.event_bus = event_bus
        self.publisher = EventPublisher(
            event_bus=event_bus,
            source_service=source_service
        )
    
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
        
        logger.info(f"Published market data for {symbol}: {price}")


class SignalPublisher:
    """Simplified signal publisher for testing."""
    
    def __init__(self, event_bus: IEventBus, source_service: str):
        """Initialize the signal publisher."""
        self.event_bus = event_bus
        self.publisher = EventPublisher(
            event_bus=event_bus,
            source_service=source_service
        )
    
    async def publish_signal(
        self,
        symbol: str,
        signal_type: str,
        confidence: float
    ) -> None:
        """Publish trading signal."""
        payload = {
            "symbol": symbol,
            "signal_type": signal_type,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.publisher.publish(
            event_type=EventType.SIGNAL_GENERATED,
            payload=payload
        )
        
        logger.info(f"Published {signal_type} signal for {symbol} with confidence {confidence}")


class PositionPublisher:
    """Simplified position publisher for testing."""
    
    def __init__(self, event_bus: IEventBus, source_service: str):
        """Initialize the position publisher."""
        self.event_bus = event_bus
        self.publisher = EventPublisher(
            event_bus=event_bus,
            source_service=source_service
        )
    
    async def open_position(
        self,
        position_id: str,
        symbol: str,
        direction: str,
        quantity: float,
        entry_price: float,
        account_id: str
    ) -> None:
        """Open a position."""
        payload = {
            "position_id": position_id,
            "symbol": symbol,
            "direction": direction,
            "quantity": quantity,
            "entry_price": entry_price,
            "account_id": account_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.publisher.publish(
            event_type=EventType.POSITION_OPENED,
            payload=payload,
            priority=EventPriority.HIGH
        )
        
        logger.info(f"Opened {direction} position for {symbol} with quantity {quantity} at {entry_price}")
    
    async def update_position(
        self,
        position_id: str,
        current_price: float
    ) -> None:
        """Update a position."""
        payload = {
            "position_id": position_id,
            "current_price": current_price,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.publisher.publish(
            event_type=EventType.POSITION_UPDATED,
            payload=payload
        )
        
        logger.info(f"Updated position {position_id} with current price {current_price}")
    
    async def close_position(
        self,
        position_id: str,
        exit_price: float,
        realized_pl: float
    ) -> None:
        """Close a position."""
        payload = {
            "position_id": position_id,
            "exit_price": exit_price,
            "realized_pl": realized_pl,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.publisher.publish(
            event_type=EventType.POSITION_CLOSED,
            payload=payload,
            priority=EventPriority.HIGH
        )
        
        logger.info(f"Closed position {position_id} with exit price {exit_price} and realized P&L {realized_pl}")


class TestEventDrivenArchitecture(unittest.IsolatedAsyncioTestCase):
    """Test case for event-driven architecture."""
    
    async def asyncSetUp(self):
        """Set up the test case."""
        # Create event bus
        self.event_bus = InMemoryEventBus()
        await self.event_bus.start()
        
        # Create publishers
        self.market_data_publisher = MarketDataPublisher(
            event_bus=self.event_bus,
            source_service="market-data-service"
        )
        
        self.signal_publisher = SignalPublisher(
            event_bus=self.event_bus,
            source_service="signal-service"
        )
        
        self.position_publisher = PositionPublisher(
            event_bus=self.event_bus,
            source_service="position-service"
        )
        
        # Create event handlers
        self.market_data_events = []
        self.signal_events = []
        self.position_events = []
        
        # Subscribe to events
        self.event_bus.subscribe(
            event_types=[EventType.MARKET_DATA_UPDATED],
            handler=self.handle_market_data
        )
        
        self.event_bus.subscribe(
            event_types=[EventType.SIGNAL_GENERATED],
            handler=self.handle_signal
        )
        
        self.event_bus.subscribe(
            event_types=[
                EventType.POSITION_OPENED,
                EventType.POSITION_UPDATED,
                EventType.POSITION_CLOSED
            ],
            handler=self.handle_position
        )
    
    async def asyncTearDown(self):
        """Tear down the test case."""
        await self.event_bus.stop()
    
    async def handle_market_data(self, event: Event) -> None:
        """Handle market data event."""
        self.market_data_events.append(event)
        logger.info(f"Received market data event: {event.payload.get('symbol')} {event.payload.get('price')}")
    
    async def handle_signal(self, event: Event) -> None:
        """Handle signal event."""
        self.signal_events.append(event)
        logger.info(f"Received signal event: {event.payload.get('symbol')} {event.payload.get('signal_type')}")
    
    async def handle_position(self, event: Event) -> None:
        """Handle position event."""
        self.position_events.append(event)
        logger.info(f"Received position event: {event.event_type} {event.payload.get('position_id')}")
    
    async def test_market_data_distribution(self):
        """Test market data distribution."""
        # Publish market data
        await self.market_data_publisher.publish_market_data(
            symbol="EUR/USD",
            price=1.1234,
            timestamp=datetime.utcnow().isoformat()
        )
        
        await self.market_data_publisher.publish_market_data(
            symbol="GBP/USD",
            price=1.2345,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Wait for events to be processed
        await asyncio.sleep(0.1)
        
        # Check that events were received
        self.assertEqual(len(self.market_data_events), 2)
        
        self.assertEqual(self.market_data_events[0].payload["symbol"], "EUR/USD")
        self.assertEqual(self.market_data_events[0].payload["price"], 1.1234)
        
        self.assertEqual(self.market_data_events[1].payload["symbol"], "GBP/USD")
        self.assertEqual(self.market_data_events[1].payload["price"], 1.2345)
    
    async def test_trading_signal_distribution(self):
        """Test trading signal distribution."""
        # Publish trading signals
        await self.signal_publisher.publish_signal(
            symbol="EUR/USD",
            signal_type="buy",
            confidence=0.85
        )
        
        await self.signal_publisher.publish_signal(
            symbol="GBP/USD",
            signal_type="sell",
            confidence=0.75
        )
        
        # Wait for events to be processed
        await asyncio.sleep(0.1)
        
        # Check that events were received
        self.assertEqual(len(self.signal_events), 2)
        
        self.assertEqual(self.signal_events[0].payload["symbol"], "EUR/USD")
        self.assertEqual(self.signal_events[0].payload["signal_type"], "buy")
        self.assertEqual(self.signal_events[0].payload["confidence"], 0.85)
        
        self.assertEqual(self.signal_events[1].payload["symbol"], "GBP/USD")
        self.assertEqual(self.signal_events[1].payload["signal_type"], "sell")
        self.assertEqual(self.signal_events[1].payload["confidence"], 0.75)
    
    async def test_position_event_sourcing(self):
        """Test position event sourcing."""
        # Open position
        position_id = "test-position-1"
        await self.position_publisher.open_position(
            position_id=position_id,
            symbol="EUR/USD",
            direction="BUY",
            quantity=1.0,
            entry_price=1.1234,
            account_id="test-account"
        )
        
        # Wait for event to be processed
        await asyncio.sleep(0.1)
        
        # Check that event was received
        self.assertEqual(len(self.position_events), 1)
        self.assertEqual(self.position_events[0].event_type, EventType.POSITION_OPENED)
        self.assertEqual(self.position_events[0].payload["position_id"], position_id)
        self.assertEqual(self.position_events[0].payload["symbol"], "EUR/USD")
        self.assertEqual(self.position_events[0].payload["direction"], "BUY")
        self.assertEqual(self.position_events[0].payload["quantity"], 1.0)
        self.assertEqual(self.position_events[0].payload["entry_price"], 1.1234)
        
        # Update position
        await self.position_publisher.update_position(
            position_id=position_id,
            current_price=1.1250
        )
        
        # Wait for event to be processed
        await asyncio.sleep(0.1)
        
        # Check that event was received
        self.assertEqual(len(self.position_events), 2)
        self.assertEqual(self.position_events[1].event_type, EventType.POSITION_UPDATED)
        self.assertEqual(self.position_events[1].payload["position_id"], position_id)
        self.assertEqual(self.position_events[1].payload["current_price"], 1.1250)
        
        # Close position
        await self.position_publisher.close_position(
            position_id=position_id,
            exit_price=1.1260,
            realized_pl=0.0026
        )
        
        # Wait for event to be processed
        await asyncio.sleep(0.1)
        
        # Check that event was received
        self.assertEqual(len(self.position_events), 3)
        self.assertEqual(self.position_events[2].event_type, EventType.POSITION_CLOSED)
        self.assertEqual(self.position_events[2].payload["position_id"], position_id)
        self.assertEqual(self.position_events[2].payload["exit_price"], 1.1260)
        self.assertEqual(self.position_events[2].payload["realized_pl"], 0.0026)


if __name__ == "__main__":
    unittest.main()
