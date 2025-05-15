"""
Market Data Consumer Service

This service is responsible for consuming market data events from the event bus.
It provides an interface for other services to access market data.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Awaitable

from common_lib.events.base import Event, EventType, EventPriority, EventMetadata, IEventBus
from common_lib.events.event_bus_factory import EventBusFactory, EventBusType
from common_lib.exceptions import ServiceError

logger = logging.getLogger(__name__)


class MarketDataConsumer:
    """
    Service for consuming market data events.
    
    This service subscribes to market data events from the event bus and provides
    an interface for other services to access the latest market data.
    """
    
    def __init__(
        self,
        service_name: str = "market-data-consumer",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the market data consumer.
        
        Args:
            service_name: Name of the service
            config: Configuration options
        """
        self.service_name = service_name
        self.config = config or {}
        
        # Initialize event bus
        self.event_bus = EventBusFactory.create_event_bus(
            bus_type=EventBusType.IN_MEMORY,  # Use in-memory for development, Kafka for production
            service_name=service_name
        )
        
        # Market data cache
        self.market_data_cache: Dict[str, Dict[str, Any]] = {}
        
        # Last update timestamps
        self.last_update: Dict[str, float] = {}
        
        # Cache expiry in seconds
        self.cache_expiry = self.config.get("cache_expiry", 60.0)
        
        # Callbacks for market data updates
        self.update_callbacks: List[Callable[[str, Dict[str, Any]], Awaitable[None]]] = []
        
        # Running flag
        self.running = False
    
    async def start(self) -> None:
        """
        Start the market data consumer.
        """
        if self.running:
            logger.warning("Market data consumer is already running")
            return
        
        # Start the event bus
        await self.event_bus.start()
        
        # Subscribe to market data events
        self.event_bus.subscribe(
            event_types=[EventType.MARKET_DATA_UPDATED],
            handler=self._handle_market_data_event
        )
        
        # Set running flag
        self.running = True
        
        logger.info("Market data consumer started")
    
    async def stop(self) -> None:
        """
        Stop the market data consumer.
        """
        if not self.running:
            logger.warning("Market data consumer is not running")
            return
        
        # Set running flag
        self.running = False
        
        # Stop the event bus
        await self.event_bus.stop()
        
        logger.info("Market data consumer stopped")
    
    async def _handle_market_data_event(self, event: Event) -> None:
        """
        Handle market data event.
        
        Args:
            event: Market data event
        """
        try:
            # Extract payload
            payload = event.payload
            symbol = payload.get("symbol")
            
            if not symbol:
                logger.warning("Received market data event without symbol")
                return
            
            # Update cache
            self.market_data_cache[symbol] = payload
            self.last_update[symbol] = time.time()
            
            # Call update callbacks
            for callback in self.update_callbacks:
                try:
                    await callback(symbol, payload)
                except Exception as e:
                    logger.error(f"Error in market data update callback: {str(e)}")
            
            logger.debug(f"Updated market data for {symbol}: {payload.get('price')}")
            
        except Exception as e:
            logger.error(f"Error handling market data event: {str(e)}")
    
    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get market data for a symbol.
        
        Args:
            symbol: Symbol to get market data for
            
        Returns:
            Market data or None if not available
        """
        # Check if we have data for this symbol
        if symbol not in self.market_data_cache:
            return None
        
        # Check if data is expired
        now = time.time()
        if now - self.last_update.get(symbol, 0) > self.cache_expiry:
            logger.warning(f"Market data for {symbol} is expired")
            return None
        
        return self.market_data_cache[symbol]
    
    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get price for a symbol.
        
        Args:
            symbol: Symbol to get price for
            
        Returns:
            Price or None if not available
        """
        market_data = self.get_market_data(symbol)
        if not market_data:
            return None
        
        return market_data.get("price")
    
    def register_update_callback(
        self,
        callback: Callable[[str, Dict[str, Any]], Awaitable[None]]
    ) -> None:
        """
        Register a callback for market data updates.
        
        Args:
            callback: Callback function that takes symbol and market data
        """
        self.update_callbacks.append(callback)
    
    def unregister_update_callback(
        self,
        callback: Callable[[str, Dict[str, Any]], Awaitable[None]]
    ) -> None:
        """
        Unregister a callback for market data updates.
        
        Args:
            callback: Callback function to unregister
        """
        if callback in self.update_callbacks:
            self.update_callbacks.remove(callback)


# Singleton instance
_market_data_consumer = None


def get_market_data_consumer(
    service_name: str = "market-data-consumer",
    config: Optional[Dict[str, Any]] = None
) -> MarketDataConsumer:
    """
    Get the singleton market data consumer instance.
    
    Args:
        service_name: Name of the service
        config: Configuration options
        
    Returns:
        Market data consumer instance
    """
    global _market_data_consumer
    
    if _market_data_consumer is None:
        _market_data_consumer = MarketDataConsumer(
            service_name=service_name,
            config=config
        )
    
    return _market_data_consumer
