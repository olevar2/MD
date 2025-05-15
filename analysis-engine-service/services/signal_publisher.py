"""
Trading Signal Publisher Service

This service is responsible for publishing trading signal events to the event bus.
It converts the traditional request-response pattern for trading signals into an event-driven pattern.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Set

from common_lib.events.base import Event, EventType, EventPriority, EventMetadata
from common_lib.events.event_publisher import EventPublisher
from common_lib.events.event_bus_factory import EventBusFactory, EventBusType
from common_lib.exceptions import ServiceError

logger = logging.getLogger(__name__)


class SignalPublisher:
    """
    Service for publishing trading signal events.
    
    This service converts trading signals from various sources into events and publishes them
    to the event bus for consumption by other services.
    """
    
    def __init__(
        self,
        service_name: str = "signal-publisher",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the signal publisher.
        
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
        
        # Initialize event publisher
        self.publisher = EventPublisher(
            event_bus=self.event_bus,
            source_service=service_name
        )
        
        # Set a correlation ID for all events published by this service
        self.publisher.set_correlation_id(f"signal-publisher-{int(time.time())}")
        
        # Running flag
        self.running = False
    
    async def start(self) -> None:
        """
        Start the signal publisher.
        """
        if self.running:
            logger.warning("Signal publisher is already running")
            return
        
        # Start the event bus
        await self.event_bus.start()
        
        # Set running flag
        self.running = True
        
        logger.info("Signal publisher started")
    
    async def stop(self) -> None:
        """
        Stop the signal publisher.
        """
        if not self.running:
            logger.warning("Signal publisher is not running")
            return
        
        # Set running flag
        self.running = False
        
        # Stop the event bus
        await self.event_bus.stop()
        
        logger.info("Signal publisher stopped")
    
    async def publish_signal(
        self,
        symbol: str,
        signal_type: str,
        timeframe: str,
        confidence: float,
        price: Optional[float] = None,
        indicator_name: Optional[str] = None,
        strategy_name: Optional[str] = None,
        expiry_time: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Publish a trading signal.
        
        Args:
            symbol: Trading symbol
            signal_type: Signal type (buy, sell, hold)
            timeframe: Timeframe (e.g., 1m, 5m, 1h, 1d)
            confidence: Confidence level (0-1)
            price: Current price
            indicator_name: Name of the indicator that generated the signal
            strategy_name: Name of the strategy that generated the signal
            expiry_time: Expiry time for the signal
            metadata: Additional metadata
            
        Returns:
            Signal ID
        """
        try:
            # Generate signal ID
            signal_id = str(uuid.uuid4())
            
            # Create payload
            payload = {
                "signal_id": signal_id,
                "symbol": symbol,
                "signal_type": signal_type,
                "timeframe": timeframe,
                "confidence": confidence,
                "price": price,
                "indicator_name": indicator_name,
                "strategy_name": strategy_name,
                "timestamp": datetime.utcnow().isoformat(),
                "expiry_time": expiry_time,
                "metadata": metadata or {}
            }
            
            # Determine priority based on confidence
            priority = EventPriority.MEDIUM
            if confidence >= 0.8:
                priority = EventPriority.HIGH
            elif confidence <= 0.3:
                priority = EventPriority.LOW
            
            # Publish event
            await self.publisher.publish(
                event_type=EventType.SIGNAL_GENERATED,
                payload=payload,
                priority=priority
            )
            
            logger.info(f"Published {signal_type} signal for {symbol} with confidence {confidence:.2f}")
            
            return signal_id
            
        except Exception as e:
            logger.error(f"Error publishing signal: {str(e)}")
            raise ServiceError(f"Failed to publish signal: {str(e)}")


# Singleton instance
_signal_publisher = None


def get_signal_publisher(
    service_name: str = "signal-publisher",
    config: Optional[Dict[str, Any]] = None
) -> SignalPublisher:
    """
    Get the singleton signal publisher instance.
    
    Args:
        service_name: Name of the service
        config: Configuration options
        
    Returns:
        Signal publisher instance
    """
    global _signal_publisher
    
    if _signal_publisher is None:
        _signal_publisher = SignalPublisher(
            service_name=service_name,
            config=config
        )
    
    return _signal_publisher
