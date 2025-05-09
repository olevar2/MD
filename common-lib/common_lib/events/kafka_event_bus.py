"""
Kafka Event Bus Implementation

This module provides a Kafka-based implementation of the event bus interface.
"""

import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Union, cast

# Import confluent_kafka, the preferred Kafka client for Python
try:
    from confluent_kafka import Consumer, KafkaError, KafkaException, Producer
    from confluent_kafka.admin import AdminClient, NewTopic
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    # Define stub classes for type hints to work without the dependency
    class Producer:
        pass
    class Consumer:
        pass
    class AdminClient:
        pass
    class NewTopic:
        pass
    class KafkaError:
        pass
    class KafkaException(Exception):
        pass

from .event_bus import EventBus, EventFilter, EventHandler
from .event_schema import Event, EventType

# Configure logger
logger = logging.getLogger(__name__)


class KafkaConfig:
    """Configuration for Kafka event bus."""
    
    def __init__(
        self,
        bootstrap_servers: str,
        service_name: str,
        group_id: Optional[str] = None,
        client_id: Optional[str] = None,
        auto_create_topics: bool = True,
        producer_config: Optional[Dict[str, Any]] = None,
        consumer_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Kafka configuration.
        
        Args:
            bootstrap_servers: Comma-separated list of Kafka broker addresses
            service_name: Name of the service using this event bus
            group_id: Consumer group ID (defaults to service_name)
            client_id: Client ID (defaults to service_name)
            auto_create_topics: Whether to automatically create topics
            producer_config: Additional Kafka producer configuration
            consumer_config: Additional Kafka consumer configuration
        """
        self.bootstrap_servers = bootstrap_servers
        self.service_name = service_name
        self.group_id = group_id or service_name
        self.client_id = client_id or service_name
        self.auto_create_topics = auto_create_topics
        self.producer_config = producer_config or {}
        self.consumer_config = consumer_config or {}


class TopicNamingStrategy:
    """Strategy for naming Kafka topics."""
    
    @staticmethod
    def event_type_to_topic(event_type: Union[EventType, str]) -> str:
        """
        Convert an event type to a Kafka topic name.
        
        Args:
            event_type: The event type
            
        Returns:
            The Kafka topic name
        """
        # Use the event type value directly as the topic name
        if isinstance(event_type, EventType):
            return f"forex.{event_type.value}"
        return f"forex.{event_type}"
    
    @staticmethod
    def topic_to_event_type(topic: str) -> str:
        """
        Convert a Kafka topic name to an event type.
        
        Args:
            topic: The Kafka topic name
            
        Returns:
            The event type
        """
        # Remove the "forex." prefix if present
        if topic.startswith("forex."):
            return topic[6:]
        return topic


class KafkaEventBus(EventBus):
    """
    Kafka-based implementation of the event bus.
    
    This class implements the EventBus interface using Kafka as the underlying message broker.
    """
    
    def __init__(self, config: KafkaConfig):
        """
        Initialize the Kafka event bus.
        
        Args:
            config: Kafka configuration
        """
        if not KAFKA_AVAILABLE:
            raise ImportError(
                "confluent_kafka is required for the Kafka event bus. "
                "Install it with 'pip install confluent-kafka'"
            )
        
        self.config = config
        self._logger = logging.getLogger(__name__)
        
        # Initialize Kafka producer
        producer_config = {
            'bootstrap.servers': config.bootstrap_servers,
            'client.id': f"{config.client_id}-producer",
            'acks': 'all',  # Wait for all replicas to acknowledge
            **config.producer_config
        }
        self._producer = Producer(producer_config)
        
        # Consumer will be initialized when subscribe is called
        self._consumer = None
        self._consumer_thread = None
        
        # Track event handlers and subscriptions
        self._event_handlers: Dict[str, List[Dict[str, Any]]] = {}
        self._subscribed_topics: Set[str] = set()
        self._running = False
        self._polling_interval = 0.1  # seconds
        
        # Create admin client for topic management
        self._admin_client = None
        if self.config.auto_create_topics:
            self._admin_client = AdminClient({'bootstrap.servers': config.bootstrap_servers})
    
    def publish(self, event: Event) -> None:
        """
        Publish an event to the event bus.
        
        Args:
            event: The event to publish
        """
        # Convert event to JSON string
        try:
            event_json = event.json()
        except Exception as e:
            self._logger.error(f"Failed to serialize event: {e}")
            return
        
        # Determine target topic
        topic = TopicNamingStrategy.event_type_to_topic(event.event_type)
        
        # Ensure the topic exists
        if self.config.auto_create_topics:
            self._ensure_topic_exists(topic)
        
        # Publish to Kafka
        try:
            self._producer.produce(
                topic=topic,
                value=event_json.encode('utf-8'),
                key=event.metadata.event_id.encode('utf-8'),
                on_delivery=self._delivery_callback
            )
            # Poll to trigger delivery reports
            self._producer.poll(0)
            
        except Exception as e:
            self._logger.error(f"Failed to publish event to {topic}: {e}")
    
    def subscribe(
        self,
        event_types: List[Union[EventType, str]],
        handler: EventHandler,
        filter_func: Optional[EventFilter] = None
    ) -> None:
        """
        Subscribe to events of the specified types.
        
        Args:
            event_types: List of event types to subscribe to
            handler: Function to handle events
            filter_func: Optional function to filter events
        """
        # Initialize consumer if not already done
        if self._consumer is None:
            self._initialize_consumer()
        
        # Convert event types to topics
        topics = [TopicNamingStrategy.event_type_to_topic(et) for et in event_types]
        
        # Register handler for each event type
        for event_type in event_types:
            event_type_str = event_type.value if isinstance(event_type, EventType) else event_type
            
            if event_type_str not in self._event_handlers:
                self._event_handlers[event_type_str] = []
            
            self._event_handlers[event_type_str].append({
                'handler': handler,
                'filter': filter_func
            })
        
        # Subscribe to topics
        new_topics = [t for t in topics if t not in self._subscribed_topics]
        if new_topics:
            # Ensure topics exist
            if self.config.auto_create_topics:
                for topic in new_topics:
                    self._ensure_topic_exists(topic)
            
            # Update subscribed topics
            self._subscribed_topics.update(new_topics)
            
            # Update consumer subscription
            if self._consumer:
                self._consumer.subscribe(list(self._subscribed_topics))
                self._logger.debug(f"Subscribed to topics: {', '.join(new_topics)}")
    
    def unsubscribe(
        self,
        event_types: List[Union[EventType, str]],
        handler: EventHandler
    ) -> None:
        """
        Unsubscribe from events of the specified types.
        
        Args:
            event_types: List of event types to unsubscribe from
            handler: Function to unsubscribe
        """
        # Remove handler for each event type
        for event_type in event_types:
            event_type_str = event_type.value if isinstance(event_type, EventType) else event_type
            
            if event_type_str in self._event_handlers:
                self._event_handlers[event_type_str] = [
                    h for h in self._event_handlers[event_type_str]
                    if h['handler'] != handler
                ]
                
                # If no handlers left for this event type, remove it
                if not self._event_handlers[event_type_str]:
                    del self._event_handlers[event_type_str]
        
        # Recalculate subscribed topics
        required_topics = set()
        for event_type in self._event_handlers.keys():
            topic = TopicNamingStrategy.event_type_to_topic(event_type)
            required_topics.add(topic)
        
        # Update subscription if needed
        removed_topics = self._subscribed_topics - required_topics
        if removed_topics and self._consumer:
            self._subscribed_topics = required_topics
            self._consumer.subscribe(list(self._subscribed_topics))
            self._logger.debug(f"Unsubscribed from topics: {', '.join(removed_topics)}")
    
    def start_consuming(self, blocking: bool = True) -> None:
        """
        Start consuming events.
        
        Args:
            blocking: Whether to block the current thread
        """
        if not self._consumer:
            self._initialize_consumer()
        
        if self._running:
            return
        
        self._running = True
        
        if blocking:
            self._consume_loop()
        else:
            self._consumer_thread = threading.Thread(target=self._consume_loop)
            self._consumer_thread.daemon = True
            self._consumer_thread.start()
    
    def stop_consuming(self) -> None:
        """Stop consuming events."""
        self._running = False
        
        if self._consumer_thread and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=5.0)
            if self._consumer_thread.is_alive():
                self._logger.warning("Consumer thread did not terminate gracefully")
        
        if self._consumer:
            self._consumer.close()
            self._consumer = None
    
    def _initialize_consumer(self) -> None:
        """Initialize the Kafka consumer."""
        consumer_config = {
            'bootstrap.servers': self.config.bootstrap_servers,
            'group.id': self.config.group_id,
            'client.id': f"{self.config.client_id}-consumer",
            'auto.offset.reset': 'latest',  # Start from latest messages
            'enable.auto.commit': True,     # Auto-commit offsets
            **self.config.consumer_config
        }
        
        self._consumer = Consumer(consumer_config)
        self._logger.debug("Initialized Kafka consumer")
    
    def _consume_loop(self) -> None:
        """Main loop for consuming messages."""
        if not self._consumer:
            self._logger.error("Cannot start consume loop: consumer not initialized")
            return
        
        self._logger.info("Starting Kafka consumer loop")
        
        while self._running:
            try:
                # Poll for messages
                msg = self._consumer.poll(self._polling_interval)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # End of partition, not an error
                        continue
                    else:
                        self._logger.error(f"Kafka consumer error: {msg.error()}")
                        continue
                
                # Process the message
                self._process_message(msg)
                
            except Exception as e:
                self._logger.error(f"Error in Kafka consumer loop: {e}", exc_info=True)
                time.sleep(1)  # Avoid tight loop in case of persistent errors
        
        self._logger.info("Kafka consumer loop stopped")
    
    def _process_message(self, msg: Any) -> None:
        """
        Process a Kafka message.
        
        Args:
            msg: The Kafka message
        """
        try:
            # Parse the message value
            value = msg.value().decode('utf-8')
            event_data = json.loads(value)
            
            # Create an Event object
            event = Event.parse_obj(event_data)
            
            # Determine the event type
            event_type_str = event.event_type.value if isinstance(event.event_type, EventType) else event.event_type
            
            # Find handlers for this event type
            handlers = self._event_handlers.get(event_type_str, [])
            
            # Call handlers
            for handler_info in handlers:
                handler = handler_info['handler']
                filter_func = handler_info.get('filter')
                
                # Apply filter if provided
                if filter_func and not filter_func(event):
                    continue
                
                try:
                    handler(event)
                except Exception as e:
                    self._logger.error(f"Error in event handler: {e}", exc_info=True)
            
        except json.JSONDecodeError as e:
            self._logger.error(f"Failed to parse message as JSON: {e}")
        except Exception as e:
            self._logger.error(f"Error processing message: {e}", exc_info=True)
    
    def _delivery_callback(self, err: Any, msg: Any) -> None:
        """
        Callback for message delivery reports.
        
        Args:
            err: Error information
            msg: Message information
        """
        if err is not None:
            self._logger.error(f"Message delivery failed: {err}")
        else:
            self._logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")
    
    def _ensure_topic_exists(self, topic: str) -> None:
        """
        Ensure a topic exists, creating it if necessary.
        
        Args:
            topic: The topic name
        """
        if not self._admin_client:
            return
        
        try:
            # Check if topic exists
            metadata = self._admin_client.list_topics(timeout=5)
            if topic in metadata.topics:
                return
            
            # Create the topic
            new_topics = [NewTopic(
                topic,
                num_partitions=3,
                replication_factor=1,
                config={'retention.ms': '604800000'}  # 7 days retention
            )]
            
            result = self._admin_client.create_topics(new_topics)
            
            # Wait for the operation to complete
            for topic, future in result.items():
                try:
                    future.result()
                    self._logger.info(f"Created topic: {topic}")
                except Exception as e:
                    # Topic might already exist
                    if "already exists" in str(e):
                        self._logger.debug(f"Topic already exists: {topic}")
                    else:
                        self._logger.error(f"Failed to create topic {topic}: {e}")
            
        except Exception as e:
            self._logger.error(f"Error ensuring topic exists: {e}")


# Helper functions for creating and publishing events
def publish_event(
    event_bus: KafkaEventBus,
    event_type: Union[EventType, str],
    data: Dict[str, Any],
    source_service: str,
    correlation_id: Optional[str] = None,
    causation_id: Optional[str] = None,
    target_services: Optional[List[str]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> Event:
    """
    Create and publish an event in one operation.
    
    Args:
        event_bus: The event bus to publish to
        event_type: Type of the event
        data: Event payload data
        source_service: Service that created the event
        correlation_id: Optional ID to correlate related events
        causation_id: Optional ID of the event that caused this event
        target_services: Optional list of specific target services
        additional_metadata: Optional additional metadata
        
    Returns:
        The created event
    """
    from .event_schema import create_event
    
    event = create_event(
        event_type=event_type,
        data=data,
        source_service=source_service,
        correlation_id=correlation_id,
        causation_id=causation_id,
        target_services=target_services,
        additional_metadata=additional_metadata
    )
    
    event_bus.publish(event)
    
    return event
