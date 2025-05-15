"""
Event bus implementation for handling asynchronous events
"""
from typing import Dict, Any, Callable, Awaitable, List
from abc import ABC, abstractmethod
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
import json
import logging
from ..config.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

class EventBus(ABC):
    """Abstract base class for event bus implementations."""
    
    @abstractmethod
    async def publish(self, topic: str, event: Dict[str, Any], key: str = None) -> None:
        """Publish an event to a topic.
        
        Args:
            topic: Topic name
            event: Event data
            key: Optional event key
        """
        pass
    
    @abstractmethod
    async def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """Subscribe to a topic.
        
        Args:
            topic: Topic name
            handler: Event handler function
        """
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """Start the event bus."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the event bus."""
        pass

class KafkaEventBus(EventBus):
    """Kafka implementation of event bus."""
    
    def __init__(self):
        """Initialize Kafka event bus."""
        self.producer = AIOKafkaProducer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda v: v.encode('utf-8') if v else None
        )
        self.consumers: List[AIOKafkaConsumer] = []
        self.handlers: Dict[str, List[Callable[[Dict[str, Any]], Awaitable[None]]]] = {}
    
    async def publish(self, topic: str, event: Dict[str, Any], key: str = None) -> None:
        """Publish event to Kafka topic.
        
        Args:
            topic: Topic name
            event: Event data
            key: Optional event key
        """
        try:
            await self.producer.send_and_wait(topic, event, key=key)
            logger.debug(f"Published event to topic {topic}: {event}")
        except Exception as e:
            logger.error(f"Failed to publish event to topic {topic}: {e}")
            raise
    
    async def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """Subscribe to Kafka topic.
        
        Args:
            topic: Topic name
            handler: Event handler function
        """
        if topic not in self.handlers:
            self.handlers[topic] = []
            consumer = AIOKafkaConsumer(
                topic,
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
                group_id=settings.KAFKA_CONSUMER_GROUP,
                value_deserializer=lambda v: json.loads(v.decode('utf-8'))
            )
            self.consumers.append(consumer)
        
        self.handlers[topic].append(handler)
    
    async def start(self) -> None:
        """Start Kafka producer and consumers."""
        await self.producer.start()
        
        for consumer in self.consumers:
            await consumer.start()
            topic = list(consumer.subscription())[0]
            handlers = self.handlers[topic]
            
            async for msg in consumer:
                try:
                    for handler in handlers:
                        await handler(msg.value)
                except Exception as e:
                    logger.error(f"Error processing message from topic {topic}: {e}")
    
    async def stop(self) -> None:
        """Stop Kafka producer and consumers."""
        await self.producer.stop()
        for consumer in self.consumers:
            await consumer.stop()

class InMemoryEventBus(EventBus):
    """In-memory implementation of event bus for development."""
    
    def __init__(self):
        """Initialize in-memory event bus."""
        self.handlers: Dict[str, List[Callable[[Dict[str, Any]], Awaitable[None]]]] = {}
    
    async def publish(self, topic: str, event: Dict[str, Any], key: str = None) -> None:
        """Publish event to in-memory handlers.
        
        Args:
            topic: Topic name
            event: Event data
            key: Optional event key (ignored in in-memory implementation)
        """
        if topic in self.handlers:
            for handler in self.handlers[topic]:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Error processing event for topic {topic}: {e}")
    
    async def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """Subscribe to in-memory topic.
        
        Args:
            topic: Topic name
            handler: Event handler function
        """
        if topic not in self.handlers:
            self.handlers[topic] = []
        self.handlers[topic].append(handler)
    
    async def start(self) -> None:
        """Start in-memory event bus (no-op)."""
        pass
    
    async def stop(self) -> None:
        """Stop in-memory event bus (no-op)."""
        pass

def get_event_bus() -> EventBus:
    """Get event bus instance based on configuration.
    
    Returns:
        EventBus implementation
    """
    if settings.EVENT_BUS_TYPE.lower() == "kafka":
        return KafkaEventBus()
    return InMemoryEventBus()