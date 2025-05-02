"""
Kafka Event Bus Implementation for Forex Trading Platform

This module provides a Kafka-based implementation of the event bus
for publishing and subscribing to events across the Forex trading platform.
It handles the details of Kafka topic management, message serialization,
and consumer group management.
"""

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union, cast

# Import confluent_kafka, the preferred Kafka client for Python
try:
    from confluent_kafka import Consumer, KafkaError, KafkaException, Producer
    from confluent_kafka.admin import AdminClient, NewTopic
except ImportError:
    raise ImportError(
        "confluent_kafka is required for the Kafka event bus. "
        "Install it with 'pip install confluent-kafka'"
    )

from .event_schema import Event, EventType, create_event

# Configure logger
logger = logging.getLogger(__name__)

# Type definitions
EventHandler = Callable[[Event], None]
EventFilter = Callable[[Event], bool]


class TopicNamingStrategy:
    """
    Strategy for naming Kafka topics based on event types.
    This provides a consistent way to map event types to Kafka topics.
    """
    
    @staticmethod
    def event_type_to_topic(event_type: Union[str, EventType]) -> str:
        """
        Convert an event type to a Kafka topic name.
        
        Args:
            event_type: The event type to convert
            
        Returns:
            str: The corresponding Kafka topic name
        """
        # Convert enum to string if needed
        if isinstance(event_type, EventType):
            event_type = event_type.value
            
        # Replace dots with hyphens for Kafka topic naming conventions
        # Prefix with 'forex-' to namespace all our topics
        return f"forex-{event_type.replace('.', '-')}"
    
    @staticmethod
    def get_service_consumer_group_id(service_name: str) -> str:
        """
        Generate a consumer group ID for a service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            str: The consumer group ID for the service
        """
        return f"{service_name}-consumer-group"
        
    @staticmethod
    def get_all_event_topics() -> List[str]:
        """
        Get a list of all topic names based on defined event types.
        
        Returns:
            List[str]: All topic names derived from EventType enum
        """
        return [
            TopicNamingStrategy.event_type_to_topic(event_type)
            for event_type in EventType
        ]


class KafkaEventBus:
    """
    Kafka-based implementation of the event bus for the Forex trading platform.
    
    This class handles the details of Kafka integration, including:
    - Publishing events to appropriate topics
    - Subscribing to events with filtering
    - Managing topic creation
    - Handling serialization/deserialization of events
    """
    
    def __init__(
        self, 
        bootstrap_servers: str,
        service_name: str,
        auto_create_topics: bool = True,
        producer_config: Optional[Dict[str, Any]] = None,
        consumer_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Kafka event bus.
        
        Args:
            bootstrap_servers: Comma-separated list of Kafka broker addresses
            service_name: Name of the service using this event bus
            auto_create_topics: Whether to automatically create topics 
            producer_config: Additional Kafka producer configuration
            consumer_config: Additional Kafka consumer configuration
        """
        self.bootstrap_servers = bootstrap_servers
        self.service_name = service_name
        self.auto_create_topics = auto_create_topics
        
        # Default producer configuration
        self._producer_config = {
            'bootstrap.servers': bootstrap_servers,
            'client.id': f"{service_name}-producer",
            'acks': 'all',  # Wait for all replicas
            'retries': 5,    # Retry on transient errors
            'retry.backoff.ms': 200,  # Time between retries
            'linger.ms': 5,  # Small delay to batch messages
        }
        
        # Update with custom config if provided
        if producer_config:
            self._producer_config.update(producer_config)
            
        # Initialize producer
        self._producer = Producer(self._producer_config)
        
        # Default consumer configuration
        self._consumer_config = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': TopicNamingStrategy.get_service_consumer_group_id(service_name),
            'auto.offset.reset': 'latest',  # Start from latest messages by default
            'enable.auto.commit': True,
            'max.poll.interval.ms': 300000,  # 5 minutes
        }
        
        # Update with custom config if provided
        if consumer_config:
            self._consumer_config.update(consumer_config)
            
        # Consumer will be initialized when subscribe is called
        self._consumer = None
        
        # Track event handlers and subscriptions
        self._event_handlers: Dict[EventType, List[EventHandler]] = {}
        self._subscribed_topics: Set[str] = set()
        self._running = False
        self._polling_interval = 0.1  # seconds
        
        # Create admin client for topic management
        self._admin_client = None
        if self.auto_create_topics:
            self._admin_client = AdminClient({'bootstrap.servers': bootstrap_servers})
            self._ensure_topics_exist()
    
    def _ensure_topics_exist(self) -> None:
        """Create topics for all event types if they don't exist."""
        if not self._admin_client:
            return
            
        topics_to_create = []
        for event_type in EventType:
            topic_name = TopicNamingStrategy.event_type_to_topic(event_type)
            # Simple topic configuration with reasonable defaults
            topics_to_create.append(NewTopic(
                topic_name,
                num_partitions=6,         # Multiple partitions for parallelism
                replication_factor=3,     # Recommended for production (adjust based on cluster)
                config={
                    'retention.ms': 604800000,  # 7 days retention
                    'cleanup.policy': 'delete',
                    'compression.type': 'lz4',  # Good balance of speed and compression
                }
            ))
            
        try:
            futures = self._admin_client.create_topics(topics_to_create)
            
            # Wait for topic creation to complete
            for topic, future in futures.items():
                try:
                    future.result()  # Wait for creation to complete
                    logger.info(f"Created topic: {topic}")
                except KafkaException as e:
                    # Topic may already exist, which is fine
                    if "already exists" in str(e):
                        logger.debug(f"Topic already exists: {topic}")
                    else:
                        logger.error(f"Failed to create topic {topic}: {e}")
                        
        except Exception as e:
            logger.error(f"Error creating topics: {e}")
    
    def _delivery_callback(self, err, msg) -> None:
        """Callback called after message delivery attempt."""
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            # Message delivered successfully
            logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")
    
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
            logger.error(f"Failed to serialize event: {e}")
            return
            
        # Determine target topic
        topic = TopicNamingStrategy.event_type_to_topic(event.event_type)
        
        # Publish to Kafka
        try:
            self._producer.produce(
                topic=topic,
                value=event_json.encode('utf-8'),
                key=str(event.event_id).encode('utf-8') if event.event_id else None,
                on_delivery=self._delivery_callback
            )
            # Poll to trigger delivery reports
            self._producer.poll(0)
            
        except Exception as e:
            logger.error(f"Failed to publish event to {topic}: {e}")
            
    def flush(self, timeout: float = 10.0) -> None:
        """
        Wait for all messages to be delivered.
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        self._producer.flush(timeout=timeout)
    
    def subscribe(
        self,
        event_types: List[EventType],
        handler: EventHandler,
        event_filter: Optional[EventFilter] = None
    ) -> None:
        """
        Subscribe to events of specific types.
        
        Args:
            event_types: List of event types to subscribe to
            handler: Function to call when an event is received
            event_filter: Optional function to filter events
        """
        # Register handler for each event type
        for event_type in event_types:
            if event_type not in self._event_handlers:
                self._event_handlers[event_type] = []
            
            # Add handler with optional filter
            self._event_handlers[event_type].append(
                (lambda e: handler(e)) if event_filter is None 
                else (lambda e: handler(e) if event_filter(e) else None)
            )
            
            # Add topic to subscribed topics set
            topic = TopicNamingStrategy.event_type_to_topic(event_type)
            self._subscribed_topics.add(topic)
        
        # If consumer is already running, update subscriptions
        if self._consumer:
            self._consumer.subscribe(list(self._subscribed_topics))
            
    def start_consuming(self, blocking: bool = False) -> None:
        """
        Start consuming events from subscribed topics.
        
        Args:
            blocking: Whether to block and process events continuously
        """
        # Don't do anything if no subscriptions
        if not self._subscribed_topics:
            logger.warning("No topics subscribed, not starting consumer")
            return
            
        # Initialize consumer if not already done
        if self._consumer is None:
            self._consumer = Consumer(self._consumer_config)
            self._consumer.subscribe(list(self._subscribed_topics))
            
        self._running = True
        
        if blocking:
            # Run in the current thread
            try:
                self._consume_events()
            except KeyboardInterrupt:
                logger.info("Consumer interrupted, stopping")
            finally:
                self.stop_consuming()
        else:
            # Start in a separate thread
            import threading
            self._consumer_thread = threading.Thread(
                target=self._consume_events, 
                daemon=True
            )
            self._consumer_thread.start()
            
    def _consume_events(self) -> None:
        """Continuously consume events from Kafka."""
        while self._running:
            try:
                # Poll for messages
                msg = self._consumer.poll(timeout=self._polling_interval)
                
                if msg is None:
                    # No message received
                    continue
                    
                if msg.error():
                    # Handle errors
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # End of partition event, not an error
                        logger.debug(f"Reached end of partition {msg.partition()}")
                    else:
                        # Log other errors
                        logger.error(f"Consumer error: {msg.error()}")
                    continue
                    
                # Process the message
                self._process_message(msg)
                
            except Exception as e:
                logger.error(f"Error while consuming events: {e}")
                # Add a small delay to prevent tight loop in case of persistent errors
                time.sleep(1)
                
    def _process_message(self, msg) -> None:
        """
        Process a received Kafka message.
        
        Args:
            msg: The Kafka message to process
        """
        try:
            # Decode and parse the event
            event_json = msg.value().decode('utf-8')
            event_dict = json.loads(event_json)
            
            # Extract event type to find handlers
            event_type = event_dict.get('event_type')
            if not event_type:
                logger.warning(f"Received event with no event_type: {event_json[:100]}...")
                return
                
            # Convert string to enum
            try:
                event_type_enum = EventType(event_type)
            except ValueError:
                logger.warning(f"Received event with unknown event_type: {event_type}")
                return
                
            # Deserialize into Event object
            event = Event.parse_raw(event_json)
            
            # Call handlers for this event type
            handlers = self._event_handlers.get(event_type_enum, [])
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}")
                    
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from message: {msg.value()[:100]}...")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            
    def stop_consuming(self) -> None:
        """Stop consuming events."""
        self._running = False
        
        # Close the consumer if it exists
        if self._consumer:
            self._consumer.close()
            self._consumer = None
            
    def close(self) -> None:
        """Close the event bus, releasing all resources."""
        self.stop_consuming()
        if hasattr(self, '_producer'):
            self._producer.flush()
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Helper functions for creating and publishing events
def publish_event(
    event_bus: KafkaEventBus,
    event_type: EventType,
    data: Dict[str, Any],
    correlation_id: Optional[str] = None,
    causation_id: Optional[str] = None,
    target_services: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Event:
    """
    Create and publish an event in one operation.
    
    Args:
        event_bus: The event bus to publish to
        event_type: Type of the event
        data: Event payload data
        correlation_id: Optional ID to correlate related events
        causation_id: Optional ID of the event that caused this event
        target_services: Optional list of specific target services
        metadata: Optional additional metadata
        
    Returns:
        The created event
    """
    event = create_event(
        event_type=event_type,
        source_service=event_bus.service_name,
        data=data,
        correlation_id=correlation_id,
        causation_id=causation_id,
        target_services=target_services,
        metadata=metadata
    )
    
    event_bus.publish(event)
    return event
