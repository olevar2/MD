"""
Comprehensive Event-Driven Architecture Demo

This script demonstrates the complete event-driven architecture implementation
by running all components together in a realistic scenario:

1. Market data flows from data-pipeline-service to feature-store-service
2. Feature data flows from feature-store-service to analysis-engine-service
3. Trading signals flow from analysis-engine-service to strategy-execution-engine
4. Orders flow from strategy-execution-engine to trading-gateway-service
5. Position updates flow from trading-gateway-service to portfolio-management-service
6. Risk assessments flow from portfolio-management-service to risk-management-service
"""

import asyncio
import logging
import sys
import os
import random
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import event bus components
from common_lib.events.base import Event, EventType, EventPriority, EventMetadata, IEventBus
from common_lib.events.in_memory_event_bus import InMemoryEventBus
from common_lib.events.event_publisher import EventPublisher


class MarketDataService:
    """Market data service that publishes market data events."""
    
    def __init__(self, event_bus: IEventBus, source_service: str):
        """Initialize the market data service."""
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
        """Start the market data service."""
        self.running = True
        self.task = asyncio.create_task(self._publish_market_data_loop())
        logger.info("Market data service started")
    
    async def stop(self):
        """Stop the market data service."""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Market data service stopped")
    
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


class FeatureStoreService:
    """Feature store service that consumes market data events and publishes feature data events."""
    
    def __init__(self, event_bus: IEventBus, source_service: str):
        """Initialize the feature store service."""
        self.event_bus = event_bus
        self.publisher = EventPublisher(
            event_bus=event_bus,
            source_service=source_service
        )
        self.market_data = {}
        
        # Subscribe to market data events
        self.event_bus.subscribe(
            event_types=[EventType.MARKET_DATA_UPDATED],
            handler=self._handle_market_data
        )
        
        logger.info("Feature store service initialized")
    
    async def _handle_market_data(self, event: Event):
        """Handle market data event."""
        payload = event.payload
        symbol = payload.get("symbol")
        price = payload.get("price")
        
        if symbol and price:
            # Store market data
            if symbol not in self.market_data:
                self.market_data[symbol] = []
            
            self.market_data[symbol].append({
                "price": price,
                "timestamp": payload.get("timestamp")
            })
            
            # Keep only the last 100 data points
            if len(self.market_data[symbol]) > 100:
                self.market_data[symbol] = self.market_data[symbol][-100:]
            
            logger.info(f"Received market data for {symbol}: {price}")
            
            # Generate and publish features if we have enough data
            if len(self.market_data[symbol]) >= 20:
                await self._generate_and_publish_features(symbol)
    
    async def _generate_and_publish_features(self, symbol: str):
        """Generate and publish features for a symbol."""
        # Get the last 20 data points
        data = self.market_data[symbol][-20:]
        
        # Calculate simple features
        prices = [d["price"] for d in data]
        avg_price = sum(prices) / len(prices)
        min_price = min(prices)
        max_price = max(prices)
        price_range = max_price - min_price
        latest_price = prices[-1]
        
        # Calculate moving averages
        ma5 = sum(prices[-5:]) / 5
        ma10 = sum(prices[-10:]) / 10
        ma20 = sum(prices[-20:]) / 20
        
        # Create feature payload
        payload = {
            "symbol": symbol,
            "timestamp": data[-1]["timestamp"],
            "features": {
                "price": latest_price,
                "avg_price": avg_price,
                "min_price": min_price,
                "max_price": max_price,
                "price_range": price_range,
                "ma5": ma5,
                "ma10": ma10,
                "ma20": ma20,
                "ma5_diff": latest_price - ma5,
                "ma10_diff": latest_price - ma10,
                "ma20_diff": latest_price - ma20
            }
        }
        
        # Publish feature data event
        await self.publisher.publish(
            event_type="feature.data.updated",
            payload=payload,
            priority=EventPriority.MEDIUM
        )
        
        logger.info(f"Published feature data for {symbol}")


class AnalysisEngineService:
    """Analysis engine service that consumes feature data events and publishes trading signal events."""
    
    def __init__(self, event_bus: IEventBus, source_service: str):
        """Initialize the analysis engine service."""
        self.event_bus = event_bus
        self.publisher = EventPublisher(
            event_bus=event_bus,
            source_service=source_service
        )
        self.feature_data = {}
        
        # Subscribe to feature data events
        self.event_bus.subscribe(
            event_types=["feature.data.updated"],
            handler=self._handle_feature_data
        )
        
        logger.info("Analysis engine service initialized")
    
    async def _handle_feature_data(self, event: Event):
        """Handle feature data event."""
        payload = event.payload
        symbol = payload.get("symbol")
        features = payload.get("features")
        
        if symbol and features:
            # Store feature data
            self.feature_data[symbol] = features
            
            logger.info(f"Received feature data for {symbol}")
            
            # Generate and publish trading signal
            await self._generate_and_publish_signal(symbol, features)
    
    async def _generate_and_publish_signal(self, symbol: str, features: Dict[str, float]):
        """Generate and publish trading signal for a symbol."""
        # Simple trading strategy based on moving averages
        ma5_diff = features.get("ma5_diff", 0)
        ma10_diff = features.get("ma10_diff", 0)
        ma20_diff = features.get("ma20_diff", 0)
        
        # Determine signal type
        signal_type = "hold"
        confidence = 0.5
        
        if ma5_diff > 0 and ma10_diff > 0 and ma20_diff > 0:
            # Price is above all moving averages - bullish
            signal_type = "buy"
            confidence = 0.7 + 0.3 * min(1.0, (ma5_diff + ma10_diff + ma20_diff) / 0.01)
            confidence = min(0.95, confidence)
        elif ma5_diff < 0 and ma10_diff < 0 and ma20_diff < 0:
            # Price is below all moving averages - bearish
            signal_type = "sell"
            confidence = 0.7 + 0.3 * min(1.0, abs(ma5_diff + ma10_diff + ma20_diff) / 0.01)
            confidence = min(0.95, confidence)
        
        # Create signal payload
        payload = {
            "signal_id": str(uuid.uuid4()),
            "symbol": symbol,
            "signal_type": signal_type,
            "timeframe": "1m",  # 1-minute timeframe
            "confidence": round(confidence, 2),
            "price": features.get("price"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "indicator_name": "Moving Average Crossover",
            "strategy_name": "Trend Following"
        }
        
        # Only publish if not a hold signal or if confidence is high
        if signal_type != "hold" or confidence > 0.7:
            # Publish trading signal event
            await self.publisher.publish(
                event_type=EventType.SIGNAL_GENERATED,
                payload=payload,
                priority=EventPriority.MEDIUM
            )
            
            logger.info(f"Published {signal_type} signal for {symbol} with confidence {confidence:.2f}")


class StrategyExecutionService:
    """Strategy execution service that consumes trading signal events and publishes order events."""
    
    def __init__(self, event_bus: IEventBus, source_service: str):
        """Initialize the strategy execution service."""
        self.event_bus = event_bus
        self.publisher = EventPublisher(
            event_bus=event_bus,
            source_service=source_service
        )
        self.signals = {}
        self.positions = {}
        
        # Subscribe to trading signal events
        self.event_bus.subscribe(
            event_types=[EventType.SIGNAL_GENERATED],
            handler=self._handle_signal
        )
        
        logger.info("Strategy execution service initialized")
    
    async def _handle_signal(self, event: Event):
        """Handle trading signal event."""
        payload = event.payload
        signal_id = payload.get("signal_id")
        symbol = payload.get("symbol")
        signal_type = payload.get("signal_type")
        confidence = payload.get("confidence", 0)
        
        if signal_id and symbol and signal_type:
            # Store signal
            self.signals[signal_id] = payload
            
            logger.info(f"Received {signal_type} signal for {symbol} with confidence {confidence}")
            
            # Execute strategy if confidence is high enough
            if confidence >= 0.8:
                await self._execute_strategy(payload)
    
    async def _execute_strategy(self, signal: Dict[str, Any]):
        """Execute strategy based on a trading signal."""
        symbol = signal.get("symbol")
        signal_type = signal.get("signal_type")
        price = signal.get("price")
        
        # Check if we already have a position for this symbol
        position = self.positions.get(symbol)
        
        if position:
            # If signal is opposite to position direction, close position
            if (position["direction"] == "BUY" and signal_type == "sell") or \
               (position["direction"] == "SELL" and signal_type == "buy"):
                # Create order payload
                payload = {
                    "order_id": str(uuid.uuid4()),
                    "symbol": symbol,
                    "order_type": "MARKET",
                    "side": "SELL" if position["direction"] == "BUY" else "BUY",
                    "quantity": position["quantity"],
                    "price": price,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "position_id": position["id"],
                    "account_id": "demo-account",
                    "status": "NEW"
                }
                
                # Publish order event
                await self.publisher.publish(
                    event_type=EventType.ORDER_CREATED,
                    payload=payload,
                    priority=EventPriority.HIGH
                )
                
                logger.info(f"Published order to close position for {symbol}")
                
                # Remove position
                del self.positions[symbol]
        
        elif signal_type in ["buy", "sell"]:
            # Create new position
            position_id = str(uuid.uuid4())
            
            # Store position
            self.positions[symbol] = {
                "id": position_id,
                "symbol": symbol,
                "direction": signal_type.upper(),
                "quantity": 1.0,
                "entry_price": price
            }
            
            # Create order payload
            payload = {
                "order_id": str(uuid.uuid4()),
                "symbol": symbol,
                "order_type": "MARKET",
                "side": signal_type.upper(),
                "quantity": 1.0,
                "price": price,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "position_id": position_id,
                "account_id": "demo-account",
                "status": "NEW"
            }
            
            # Publish order event
            await self.publisher.publish(
                event_type=EventType.ORDER_CREATED,
                payload=payload,
                priority=EventPriority.HIGH
            )
            
            logger.info(f"Published order to open position for {symbol}")


async def run_demo():
    """Run the comprehensive event-driven architecture demo."""
    # Create event bus
    event_bus = InMemoryEventBus()
    await event_bus.start()
    
    try:
        # Create services
        market_data_service = MarketDataService(
            event_bus=event_bus,
            source_service="market-data-service"
        )
        
        feature_store_service = FeatureStoreService(
            event_bus=event_bus,
            source_service="feature-store-service"
        )
        
        analysis_engine_service = AnalysisEngineService(
            event_bus=event_bus,
            source_service="analysis-engine-service"
        )
        
        strategy_execution_service = StrategyExecutionService(
            event_bus=event_bus,
            source_service="strategy-execution-service"
        )
        
        # Start services
        await market_data_service.start()
        
        # Run for 60 seconds
        logger.info("Running comprehensive event-driven architecture demo for 60 seconds...")
        await asyncio.sleep(60)
        
        # Stop services
        await market_data_service.stop()
        
    finally:
        # Stop event bus
        await event_bus.stop()


if __name__ == "__main__":
    asyncio.run(run_demo())
