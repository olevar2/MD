"""
Market Data Publisher Service

This service is responsible for publishing market data events to the event bus.
It converts the traditional request-response pattern for market data into an event-driven pattern.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Set

from common_lib.events.base import Event, EventType, EventPriority, EventMetadata
from common_lib.events.event_publisher import EventPublisher
from common_lib.events.event_bus_factory import EventBusFactory, EventBusType
from common_lib.exceptions import ServiceError

logger = logging.getLogger(__name__)


class MarketDataPublisher:
    """
    Service for publishing market data events.
    
    This service converts market data from various sources into events and publishes them
    to the event bus for consumption by other services.
    """
    
    def __init__(
        self,
        service_name: str = "market-data-publisher",
        publish_interval: float = 1.0,  # Default to publishing every 1 second
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the market data publisher.
        
        Args:
            service_name: Name of the service
            publish_interval: Interval in seconds between publishing market data
            config: Configuration options
        """
        self.service_name = service_name
        self.publish_interval = publish_interval
        self.config = config or {}
        
        # Initialize event bus
        self.event_bus = EventBusFactory.create_event_bus(
            bus_type=EventBusType.IN_MEMORY,  # Use in-memory for development, Kafka for production
            service_name=service_name
        )
        
        # Initialize event publisher
        self.publisher = EventPublisher(
            event_bus=self.event_bus,
            source_service=service_name
        )
        
        # Set a correlation ID for all events published by this service
        self.publisher.set_correlation_id(f"market-data-{int(time.time())}")
        
        # Track symbols being monitored
        self.monitored_symbols: Set[str] = set()
        
        # Market data cache
        self.market_data_cache: Dict[str, Dict[str, Any]] = {}
        
        # Running flag
        self.running = False
        
        # Publishing task
        self.publishing_task = None
    
    async def start(self) -> None:
        """
        Start the market data publisher.
        """
        if self.running:
            logger.warning("Market data publisher is already running")
            return
        
        # Start the event bus
        await self.event_bus.start()
        
        # Set running flag
        self.running = True
        
        # Start publishing task
        self.publishing_task = asyncio.create_task(self._publish_market_data_loop())
        
        logger.info(f"Market data publisher started with interval {self.publish_interval}s")
    
    async def stop(self) -> None:
        """
        Stop the market data publisher.
        """
        if not self.running:
            logger.warning("Market data publisher is not running")
            return
        
        # Set running flag
        self.running = False
        
        # Cancel publishing task
        if self.publishing_task:
            self.publishing_task.cancel()
            try:
                await self.publishing_task
            except asyncio.CancelledError:
                pass
        
        # Stop the event bus
        await self.event_bus.stop()
        
        logger.info("Market data publisher stopped")
    
    def add_symbol(self, symbol: str) -> None:
        """
        Add a symbol to monitor.
        
        Args:
            symbol: Symbol to monitor
        """
        self.monitored_symbols.add(symbol)
        logger.info(f"Added symbol {symbol} to monitoring list")
    
    def remove_symbol(self, symbol: str) -> None:
        """
        Remove a symbol from monitoring.
        
        Args:
            symbol: Symbol to remove
        """
        if symbol in self.monitored_symbols:
            self.monitored_symbols.remove(symbol)
            logger.info(f"Removed symbol {symbol} from monitoring list")
    
    def update_market_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """
        Update market data for a symbol.
        
        Args:
            symbol: Symbol to update
            data: Market data
        """
        self.market_data_cache[symbol] = data
    
    async def _publish_market_data_loop(self) -> None:
        """
        Main loop for publishing market data.
        """
        while self.running:
            try:
                # Publish market data for all monitored symbols
                for symbol in self.monitored_symbols:
                    if symbol in self.market_data_cache:
                        await self._publish_market_data(symbol, self.market_data_cache[symbol])
                
                # Wait for next publishing interval
                await asyncio.sleep(self.publish_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in market data publishing loop: {str(e)}")
                await asyncio.sleep(1)  # Wait a bit before retrying
    
    async def _publish_market_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """
        Publish market data for a symbol.
        
        Args:
            symbol: Symbol to publish
            data: Market data
        """
        try:
            # Create payload
            payload = {
                "symbol": symbol,
                "price": data.get("price"),
                "bid": data.get("bid"),
                "ask": data.get("ask"),
                "timestamp": data.get("timestamp") or datetime.utcnow().isoformat(),
                "volume": data.get("volume"),
                "high": data.get("high"),
                "low": data.get("low"),
                "open": data.get("open"),
                "close": data.get("close")
            }
            
            # Publish event
            await self.publisher.publish(
                event_type=EventType.MARKET_DATA_UPDATED,
                payload=payload,
                priority=EventPriority.HIGH
            )
            
            logger.debug(f"Published market data for {symbol}: {payload['price']}")
            
        except Exception as e:
            logger.error(f"Error publishing market data for {symbol}: {str(e)}")


# Singleton instance
_market_data_publisher = None


def get_market_data_publisher(
    service_name: str = "market-data-publisher",
    publish_interval: float = 1.0,
    config: Optional[Dict[str, Any]] = None
) -> MarketDataPublisher:
    """
    Get the singleton market data publisher instance.
    
    Args:
        service_name: Name of the service
        publish_interval: Interval in seconds between publishing market data
        config: Configuration options
        
    Returns:
        Market data publisher instance
    """
    global _market_data_publisher
    
    if _market_data_publisher is None:
        _market_data_publisher = MarketDataPublisher(
            service_name=service_name,
            publish_interval=publish_interval,
            config=config
        )
    
    return _market_data_publisher
