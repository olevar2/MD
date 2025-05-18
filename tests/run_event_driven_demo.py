"""
Event-Driven Architecture Demo

This script demonstrates the complete event-driven architecture implementation
by running all components together in a realistic scenario.
"""

import asyncio
import logging
import sys
import os
import uuid
import random
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set

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
    """Market data publisher service."""
    
    def __init__(self, event_bus: IEventBus, source_service: str):
        """Initialize the market data publisher."""
        self.event_bus = event_bus
        self.publisher = EventPublisher(
            event_bus=event_bus,
            source_service=source_service
        )
        self.symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
        self.prices = {
            "EUR/USD": 1.1234,
            "GBP/USD": 1.2345,
            "USD/JPY": 110.45,
            "AUD/USD": 0.7456
        }
        self.running = False
        self.task = None
    
    async def start(self):
        """Start the market data publisher."""
        self.running = True
        self.task = asyncio.create_task(self._publish_market_data_loop())
        logger.info("Market data publisher started")
    
    async def stop(self):
        """Stop the market data publisher."""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Market data publisher stopped")
    
    async def _publish_market_data_loop(self):
        """Publish market data in a loop."""
        while self.running:
            for symbol in self.symbols:
                # Update price with small random change
                change = random.uniform(-0.0050, 0.0050)
                self.prices[symbol] += change
                self.prices[symbol] = round(self.prices[symbol], 4)
                
                # Publish market data
                await self.publish_market_data(
                    symbol=symbol,
                    price=self.prices[symbol],
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
            
            # Wait before next update
            await asyncio.sleep(1)
    
    async def publish_market_data(self, symbol: str, price: float, timestamp: str):
        """Publish market data for a symbol."""
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


class MarketDataConsumer:
    """Market data consumer service."""
    
    def __init__(self, event_bus: IEventBus, source_service: str):
        """Initialize the market data consumer."""
        self.event_bus = event_bus
        self.market_data = {}
        
        # Subscribe to market data events
        self.event_bus.subscribe(
            event_types=[EventType.MARKET_DATA_UPDATED],
            handler=self._handle_market_data
        )
        
        logger.info("Market data consumer initialized")
    
    async def _handle_market_data(self, event: Event):
        """Handle market data event."""
        payload = event.payload
        symbol = payload.get("symbol")
        price = payload.get("price")
        
        if symbol and price:
            self.market_data[symbol] = payload
            logger.info(f"Received market data for {symbol}: {price}")
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol."""
        if symbol in self.market_data:
            return self.market_data[symbol].get("price")
        return None


class SignalGenerator:
    """Trading signal generator service."""
    
    def __init__(self, event_bus: IEventBus, market_data_consumer: MarketDataConsumer, source_service: str):
        """Initialize the signal generator."""
        self.event_bus = event_bus
        self.market_data_consumer = market_data_consumer
        self.publisher = EventPublisher(
            event_bus=event_bus,
            source_service=source_service
        )
        self.symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
        self.last_prices = {}
        self.running = False
        self.task = None
    
    async def start(self):
        """Start the signal generator."""
        self.running = True
        self.task = asyncio.create_task(self._generate_signals_loop())
        logger.info("Signal generator started")
    
    async def stop(self):
        """Stop the signal generator."""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Signal generator stopped")
    
    async def _generate_signals_loop(self):
        """Generate trading signals in a loop."""
        while self.running:
            for symbol in self.symbols:
                # Get current price
                current_price = self.market_data_consumer.get_price(symbol)
                if not current_price:
                    continue
                
                # Get last price
                last_price = self.last_prices.get(symbol)
                if not last_price:
                    self.last_prices[symbol] = current_price
                    continue
                
                # Calculate price change
                price_change = current_price - last_price
                
                # Generate signal if price change is significant
                if abs(price_change) > 0.0020:
                    signal_type = "buy" if price_change > 0 else "sell"
                    confidence = min(1.0, abs(price_change) * 100)
                    
                    await self.publish_signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        confidence=confidence,
                        price=current_price
                    )
                
                # Update last price
                self.last_prices[symbol] = current_price
            
            # Wait before next check
            await asyncio.sleep(2)
    
    async def publish_signal(self, symbol: str, signal_type: str, confidence: float, price: float):
        """Publish a trading signal."""
        payload = {
            "signal_id": str(uuid.uuid4()),
            "symbol": symbol,
            "signal_type": signal_type,
            "confidence": confidence,
            "price": price,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await self.publisher.publish(
            event_type=EventType.SIGNAL_GENERATED,
            payload=payload,
            priority=EventPriority.MEDIUM
        )
        
        logger.info(f"Published {signal_type} signal for {symbol} with confidence {confidence:.2f}")


class SignalConsumer:
    """Trading signal consumer service."""
    
    def __init__(self, event_bus: IEventBus, source_service: str):
        """Initialize the signal consumer."""
        self.event_bus = event_bus
        self.signals = {}
        
        # Subscribe to trading signal events
        self.event_bus.subscribe(
            event_types=[EventType.SIGNAL_GENERATED],
            handler=self._handle_signal
        )
        
        logger.info("Signal consumer initialized")
    
    async def _handle_signal(self, event: Event):
        """Handle trading signal event."""
        payload = event.payload
        signal_id = payload.get("signal_id")
        symbol = payload.get("symbol")
        signal_type = payload.get("signal_type")
        
        if signal_id and symbol and signal_type:
            self.signals[signal_id] = payload
            logger.info(f"Received {signal_type} signal for {symbol} with confidence {payload.get('confidence'):.2f}")


class PositionManager:
    """Position manager service with event sourcing."""
    
    def __init__(self, event_bus: IEventBus, signal_consumer: SignalConsumer, market_data_consumer: MarketDataConsumer, source_service: str):
        """Initialize the position manager."""
        self.event_bus = event_bus
        self.signal_consumer = signal_consumer
        self.market_data_consumer = market_data_consumer
        self.publisher = EventPublisher(
            event_bus=event_bus,
            source_service=source_service
        )
        self.positions = {}
        self.running = False
        self.task = None
        
        # Subscribe to position events
        self.event_bus.subscribe(
            event_types=[
                EventType.POSITION_OPENED,
                EventType.POSITION_UPDATED,
                EventType.POSITION_CLOSED
            ],
            handler=self._handle_position_event
        )
    
    async def start(self):
        """Start the position manager."""
        self.running = True
        self.task = asyncio.create_task(self._manage_positions_loop())
        logger.info("Position manager started")
    
    async def stop(self):
        """Stop the position manager."""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Position manager stopped")
    
    async def _handle_position_event(self, event: Event):
        """Handle position event."""
        event_type = event.event_type
        payload = event.payload
        position_id = payload.get("position_id")
        
        if not position_id:
            return
        
        if event_type == EventType.POSITION_OPENED:
            # Create position
            self.positions[position_id] = {
                "id": position_id,
                "symbol": payload.get("symbol"),
                "direction": payload.get("direction"),
                "quantity": payload.get("quantity"),
                "entry_price": payload.get("entry_price"),
                "status": "OPEN",
                "entry_date": payload.get("timestamp")
            }
            logger.info(f"Position {position_id} opened: {payload.get('symbol')} {payload.get('direction')} at {payload.get('entry_price')}")
        
        elif event_type == EventType.POSITION_UPDATED:
            # Update position
            if position_id in self.positions:
                position = self.positions[position_id]
                
                if "current_price" in payload:
                    position["current_price"] = payload["current_price"]
                
                logger.info(f"Position {position_id} updated: current price {payload.get('current_price')}")
        
        elif event_type == EventType.POSITION_CLOSED:
            # Close position
            if position_id in self.positions:
                position = self.positions[position_id]
                position["status"] = "CLOSED"
                position["exit_price"] = payload.get("exit_price")
                position["realized_pl"] = payload.get("realized_pl")
                position["exit_date"] = payload.get("timestamp")
                
                logger.info(f"Position {position_id} closed: exit price {payload.get('exit_price')}, P&L {payload.get('realized_pl')}")
    
    async def _manage_positions_loop(self):
        """Manage positions in a loop."""
        while self.running:
            # Process new signals
            for signal_id, signal in list(self.signal_consumer.signals.items()):
                # Skip processed signals
                if signal.get("processed"):
                    continue
                
                symbol = signal.get("symbol")
                signal_type = signal.get("signal_type")
                confidence = signal.get("confidence", 0)
                price = signal.get("price")
                
                # Only process high confidence signals
                if confidence >= 0.7 and price:
                    # Check if we already have a position for this symbol
                    existing_position = None
                    for position in self.positions.values():
                        if position["symbol"] == symbol and position["status"] == "OPEN":
                            existing_position = position
                            break
                    
                    if existing_position:
                        # If signal is opposite to position direction, close position
                        if (existing_position["direction"] == "BUY" and signal_type == "sell") or \
                           (existing_position["direction"] == "SELL" and signal_type == "buy"):
                            await self.close_position(
                                position_id=existing_position["id"],
                                exit_price=price
                            )
                    else:
                        # Open new position
                        if signal_type in ["buy", "sell"]:
                            await self.open_position(
                                symbol=symbol,
                                direction=signal_type.upper(),
                                quantity=1.0,
                                entry_price=price
                            )
                
                # Mark signal as processed
                signal["processed"] = True
            
            # Update open positions
            for position_id, position in list(self.positions.items()):
                if position["status"] == "OPEN":
                    symbol = position["symbol"]
                    current_price = self.market_data_consumer.get_price(symbol)
                    
                    if current_price:
                        await self.update_position(
                            position_id=position_id,
                            current_price=current_price
                        )
            
            # Wait before next check
            await asyncio.sleep(1)
    
    async def open_position(self, symbol: str, direction: str, quantity: float, entry_price: float):
        """Open a new position."""
        position_id = str(uuid.uuid4())
        
        payload = {
            "position_id": position_id,
            "symbol": symbol,
            "direction": direction,
            "quantity": quantity,
            "entry_price": entry_price,
            "account_id": "demo-account",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await self.publisher.publish(
            event_type=EventType.POSITION_OPENED,
            payload=payload,
            priority=EventPriority.HIGH
        )
        
        return position_id
    
    async def update_position(self, position_id: str, current_price: float):
        """Update a position."""
        if position_id not in self.positions:
            return
        
        position = self.positions[position_id]
        
        # Calculate unrealized P&L
        if position["direction"] == "BUY":
            unrealized_pl = (current_price - position["entry_price"]) * position["quantity"]
        else:  # SELL
            unrealized_pl = (position["entry_price"] - current_price) * position["quantity"]
        
        payload = {
            "position_id": position_id,
            "current_price": current_price,
            "unrealized_pl": unrealized_pl,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await self.publisher.publish(
            event_type=EventType.POSITION_UPDATED,
            payload=payload,
            priority=EventPriority.MEDIUM
        )
    
    async def close_position(self, position_id: str, exit_price: float):
        """Close a position."""
        if position_id not in self.positions:
            return
        
        position = self.positions[position_id]
        
        # Calculate realized P&L
        if position["direction"] == "BUY":
            realized_pl = (exit_price - position["entry_price"]) * position["quantity"]
        else:  # SELL
            realized_pl = (position["entry_price"] - exit_price) * position["quantity"]
        
        payload = {
            "position_id": position_id,
            "exit_price": exit_price,
            "realized_pl": realized_pl,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await self.publisher.publish(
            event_type=EventType.POSITION_CLOSED,
            payload=payload,
            priority=EventPriority.HIGH
        )


async def run_demo():
    """Run the event-driven architecture demo."""
    # Create event bus
    event_bus = InMemoryEventBus()
    await event_bus.start()
    
    try:
        # Create services
        market_data_consumer = MarketDataConsumer(
            event_bus=event_bus,
            source_service="market-data-consumer"
        )
        
        market_data_publisher = MarketDataPublisher(
            event_bus=event_bus,
            source_service="market-data-publisher"
        )
        
        signal_consumer = SignalConsumer(
            event_bus=event_bus,
            source_service="signal-consumer"
        )
        
        signal_generator = SignalGenerator(
            event_bus=event_bus,
            market_data_consumer=market_data_consumer,
            source_service="signal-generator"
        )
        
        position_manager = PositionManager(
            event_bus=event_bus,
            signal_consumer=signal_consumer,
            market_data_consumer=market_data_consumer,
            source_service="position-manager"
        )
        
        # Start services
        await market_data_publisher.start()
        await signal_generator.start()
        await position_manager.start()
        
        # Run for 30 seconds
        logger.info("Running event-driven architecture demo for 30 seconds...")
        await asyncio.sleep(30)
        
        # Print summary
        logger.info("\n=== Demo Summary ===")
        logger.info(f"Positions: {len(position_manager.positions)}")
        
        open_positions = [p for p in position_manager.positions.values() if p["status"] == "OPEN"]
        closed_positions = [p for p in position_manager.positions.values() if p["status"] == "CLOSED"]
        
        logger.info(f"Open positions: {len(open_positions)}")
        logger.info(f"Closed positions: {len(closed_positions)}")
        
        total_realized_pl = sum(p.get("realized_pl", 0) for p in closed_positions)
        logger.info(f"Total realized P&L: {total_realized_pl:.4f}")
        
        # Stop services
        await position_manager.stop()
        await signal_generator.stop()
        await market_data_publisher.stop()
        
    finally:
        # Stop event bus
        await event_bus.stop()


if __name__ == "__main__":
    asyncio.run(run_demo())
