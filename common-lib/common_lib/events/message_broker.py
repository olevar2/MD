"""
Message Broker Module

This module provides adapters for message brokers like RabbitMQ and Kafka.
"""

import json
import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Set, Awaitable, Union, TypeVar

import aio_pika
from aio_pika import ExchangeType, Message, connect_robust
from aio_pika.abc import AbstractIncomingMessage

from common_lib.events.event import Event, EventRegistry
from common_lib.config.config_manager import ConfigManager


class MessageBroker(ABC):
    """
    Abstract base class for message brokers.
    
    This class defines the interface for message brokers.
    """
    
    @abstractmethod
    async def connect(self) -> None:
        """
        Connect to the message broker.
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from the message broker.
        """
        pass
    
    @abstractmethod
    async def publish(self, event: Event) -> None:
        """
        Publish an event to the message broker.
        
        Args:
            event: Event to publish
        """
        pass
    
    @abstractmethod
    async def subscribe(
        self,
        event_type: str,
        callback: Callable[[Event], Awaitable[None]]
    ) -> None:
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Type of events to subscribe to
            callback: Callback to call when an event of this type is received
        """
        pass
    
    @abstractmethod
    async def unsubscribe(self, event_type: str) -> None:
        """
        Unsubscribe from events of a specific type.
        
        Args:
            event_type: Type of events to unsubscribe from
        """
        pass


class RabbitMQBroker(MessageBroker):
    """
    RabbitMQ message broker.
    
    This class provides an implementation of the message broker interface for RabbitMQ.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5672,
        username: str = "guest",
        password: str = "guest",
        exchange_name: str = "events",
        queue_name: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the RabbitMQ broker.
        
        Args:
            host: RabbitMQ host
            port: RabbitMQ port
            username: RabbitMQ username
            password: RabbitMQ password
            exchange_name: Name of the exchange to use
            queue_name: Name of the queue to use (if None, a random name is generated)
            logger: Logger to use (if None, creates a new logger)
        """
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.exchange_name = exchange_name
        self.queue_name = queue_name
        
        self.connection = None
        self.channel = None
        self.exchange = None
        self.queue = None
        self.consumer_tags = {}
    
    async def connect(self) -> None:
        """
        Connect to RabbitMQ.
        """
        try:
            # Create connection
            self.connection = await connect_robust(
                host=self.host,
                port=self.port,
                login=self.username,
                password=self.password
            )
            
            # Create channel
            self.channel = await self.connection.channel()
            
            # Create exchange
            self.exchange = await self.channel.declare_exchange(
                name=self.exchange_name,
                type=ExchangeType.TOPIC,
                durable=True
            )
            
            # Create queue
            self.queue = await self.channel.declare_queue(
                name=self.queue_name or "",
                durable=True,
                auto_delete=self.queue_name is None
            )
            
            self.logger.info(f"Connected to RabbitMQ: {self.host}:{self.port}")
        except Exception as e:
            self.logger.error(f"Error connecting to RabbitMQ: {str(e)}")
            raise
    
    async def disconnect(self) -> None:
        """
        Disconnect from RabbitMQ.
        """
        try:
            # Close connection
            if self.connection:
                await self.connection.close()
                self.connection = None
                self.channel = None
                self.exchange = None
                self.queue = None
                self.consumer_tags = {}
                
                self.logger.info("Disconnected from RabbitMQ")
        except Exception as e:
            self.logger.error(f"Error disconnecting from RabbitMQ: {str(e)}")
            raise
    
    async def publish(self, event: Event) -> None:
        """
        Publish an event to RabbitMQ.
        
        Args:
            event: Event to publish
        """
        try:
            # Check if connected
            if not self.exchange:
                await self.connect()
            
            # Get routing key
            routing_key = event.get_routing_key()
            
            # Create message
            message = Message(
                body=event.to_json().encode(),
                content_type="application/json",
                headers={
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "source": event.source,
                    "correlation_id": event.correlation_id,
                    "causation_id": event.causation_id
                }
            )
            
            # Publish message
            await self.exchange.publish(
                message=message,
                routing_key=routing_key
            )
            
            self.logger.debug(f"Published event: {event.event_type} (routing key: {routing_key})")
        except Exception as e:
            self.logger.error(f"Error publishing event: {str(e)}")
            raise
    
    async def subscribe(
        self,
        event_type: str,
        callback: Callable[[Event], Awaitable[None]]
    ) -> None:
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Type of events to subscribe to
            callback: Callback to call when an event of this type is received
        """
        try:
            # Check if connected
            if not self.queue:
                await self.connect()
            
            # Bind queue to exchange
            await self.queue.bind(
                exchange=self.exchange,
                routing_key=event_type
            )
            
            # Create message handler
            async def message_handler(message: AbstractIncomingMessage):
                async with message.process():
                    try:
                        # Parse event
                        event_data = json.loads(message.body.decode())
                        event_class = EventRegistry.get(event_data["event_type"])
                        
                        if event_class:
                            event = event_class.from_dict(event_data)
                            
                            # Call callback
                            await callback(event)
                        else:
                            self.logger.warning(
                                f"Received event of unknown type: {event_data['event_type']}"
                            )
                    except Exception as e:
                        self.logger.error(f"Error processing message: {str(e)}")
            
            # Start consuming
            consumer_tag = await self.queue.consume(message_handler)
            self.consumer_tags[event_type] = consumer_tag
            
            self.logger.debug(f"Subscribed to event type: {event_type}")
        except Exception as e:
            self.logger.error(f"Error subscribing to event type {event_type}: {str(e)}")
            raise
    
    async def unsubscribe(self, event_type: str) -> None:
        """
        Unsubscribe from events of a specific type.
        
        Args:
            event_type: Type of events to unsubscribe from
        """
        try:
            # Check if connected
            if not self.channel or event_type not in self.consumer_tags:
                return
            
            # Cancel consumer
            await self.channel.basic_cancel(self.consumer_tags[event_type])
            del self.consumer_tags[event_type]
            
            # Unbind queue from exchange
            await self.queue.unbind(
                exchange=self.exchange,
                routing_key=event_type
            )
            
            self.logger.debug(f"Unsubscribed from event type: {event_type}")
        except Exception as e:
            self.logger.error(f"Error unsubscribing from event type {event_type}: {str(e)}")
            raise


class MessageBrokerFactory:
    """
    Factory for creating message brokers.
    
    This class provides a factory for creating message brokers.
    """
    
    @staticmethod
    def create_broker(
        broker_type: str = "rabbitmq",
        config_manager: Optional[ConfigManager] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ) -> MessageBroker:
        """
        Create a message broker.
        
        Args:
            broker_type: Type of message broker to create
            config_manager: Configuration manager
            logger: Logger to use (if None, creates a new logger)
            **kwargs: Additional arguments for the message broker
            
        Returns:
            Message broker
            
        Raises:
            ValueError: If the broker type is not supported
        """
        logger = logger or logging.getLogger(f"{__name__}.MessageBrokerFactory")
        
        if broker_type.lower() == "rabbitmq":
            # Get configuration
            if config_manager:
                try:
                    # Get message broker configuration
                    service_specific = config_manager.get_service_specific_config()
                    if hasattr(service_specific, "message_broker"):
                        broker_config = getattr(service_specific, "message_broker")
                        
                        # Update kwargs with configuration
                        for key, value in broker_config.dict().items():
                            if key not in kwargs:
                                kwargs[key] = value
                except Exception as e:
                    logger.warning(f"Error getting message broker configuration: {str(e)}")
            
            # Create RabbitMQ broker
            return RabbitMQBroker(logger=logger, **kwargs)
        else:
            raise ValueError(f"Unsupported message broker type: {broker_type}")