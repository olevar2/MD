"""
Trading Signal Consumer Service

This service is responsible for consuming trading signal events from the event bus.
It provides an interface for other services to access trading signals.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable, Awaitable, Set

from common_lib.events.base import Event, EventType, EventPriority, EventMetadata, IEventBus
from common_lib.events.event_bus_factory import EventBusFactory, EventBusType
from common_lib.exceptions import ServiceError

logger = logging.getLogger(__name__)


class SignalConsumer:
    """
    Service for consuming trading signal events.
    
    This service subscribes to trading signal events from the event bus and provides
    an interface for other services to access the latest trading signals.
    """
    
    def __init__(
        self,
        service_name: str = "signal-consumer",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the signal consumer.
        
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
        
        # Signal cache
        self.signal_cache: Dict[str, Dict[str, Any]] = {}  # signal_id -> signal
        
        # Symbol to signals mapping
        self.symbol_signals: Dict[str, Set[str]] = {}  # symbol -> set of signal_ids
        
        # Callbacks for signal updates
        self.update_callbacks: List[Callable[[Dict[str, Any]], Awaitable[None]]] = []
        
        # Running flag
        self.running = False
    
    async def start(self) -> None:
        """
        Start the signal consumer.
        """
        if self.running:
            logger.warning("Signal consumer is already running")
            return
        
        # Start the event bus
        await self.event_bus.start()
        
        # Subscribe to trading signal events
        self.event_bus.subscribe(
            event_types=[EventType.SIGNAL_GENERATED],
            handler=self._handle_signal_event
        )
        
        # Set running flag
        self.running = True
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_expired_signals())
        
        logger.info("Signal consumer started")
    
    async def stop(self) -> None:
        """
        Stop the signal consumer.
        """
        if not self.running:
            logger.warning("Signal consumer is not running")
            return
        
        # Set running flag
        self.running = False
        
        # Stop the event bus
        await self.event_bus.stop()
        
        logger.info("Signal consumer stopped")
    
    async def _handle_signal_event(self, event: Event) -> None:
        """
        Handle trading signal event.
        
        Args:
            event: Trading signal event
        """
        try:
            # Extract payload
            payload = event.payload
            signal_id = payload.get("signal_id")
            symbol = payload.get("symbol")
            
            if not signal_id or not symbol:
                logger.warning("Received signal event without signal_id or symbol")
                return
            
            # Update cache
            self.signal_cache[signal_id] = payload
            
            # Update symbol to signals mapping
            if symbol not in self.symbol_signals:
                self.symbol_signals[symbol] = set()
            self.symbol_signals[symbol].add(signal_id)
            
            # Call update callbacks
            for callback in self.update_callbacks:
                try:
                    await callback(payload)
                except Exception as e:
                    logger.error(f"Error in signal update callback: {str(e)}")
            
            logger.info(f"Received {payload.get('signal_type')} signal for {symbol} with confidence {payload.get('confidence'):.2f}")
            
        except Exception as e:
            logger.error(f"Error handling signal event: {str(e)}")
    
    async def _cleanup_expired_signals(self) -> None:
        """
        Cleanup expired signals.
        """
        while self.running:
            try:
                now = datetime.now(timezone.utc).isoformat()
                expired_signal_ids = []
                
                # Find expired signals
                for signal_id, signal in self.signal_cache.items():
                    expiry_time = signal.get("expiry_time")
                    if expiry_time and expiry_time < now:
                        expired_signal_ids.append(signal_id)
                
                # Remove expired signals
                for signal_id in expired_signal_ids:
                    signal = self.signal_cache.pop(signal_id, None)
                    if signal:
                        symbol = signal.get("symbol")
                        if symbol and symbol in self.symbol_signals:
                            self.symbol_signals[symbol].discard(signal_id)
                            if not self.symbol_signals[symbol]:
                                del self.symbol_signals[symbol]
                
                if expired_signal_ids:
                    logger.info(f"Cleaned up {len(expired_signal_ids)} expired signals")
                
                # Wait before next cleanup
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in signal cleanup: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying
    
    def get_signal(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a signal by ID.
        
        Args:
            signal_id: Signal ID
            
        Returns:
            Signal or None if not found
        """
        return self.signal_cache.get(signal_id)
    
    def get_signals_for_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get all signals for a symbol.
        
        Args:
            symbol: Symbol to get signals for
            
        Returns:
            List of signals
        """
        signal_ids = self.symbol_signals.get(symbol, set())
        return [self.signal_cache[signal_id] for signal_id in signal_ids if signal_id in self.signal_cache]
    
    def register_update_callback(
        self,
        callback: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> None:
        """
        Register a callback for signal updates.
        
        Args:
            callback: Callback function that takes a signal
        """
        self.update_callbacks.append(callback)
    
    def unregister_update_callback(
        self,
        callback: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> None:
        """
        Unregister a callback for signal updates.
        
        Args:
            callback: Callback function to unregister
        """
        if callback in self.update_callbacks:
            self.update_callbacks.remove(callback)


# Singleton instance
_signal_consumer = None


def get_signal_consumer(
    service_name: str = "signal-consumer",
    config: Optional[Dict[str, Any]] = None
) -> SignalConsumer:
    """
    Get the singleton signal consumer instance.
    
    Args:
        service_name: Name of the service
        config: Configuration options
        
    Returns:
        Signal consumer instance
    """
    global _signal_consumer
    
    if _signal_consumer is None:
        _signal_consumer = SignalConsumer(
            service_name=service_name,
            config=config
        )
    
    return _signal_consumer
