"""
Kafka Event Bus Module (V2)

This module provides a Kafka-based implementation of the event bus interface.
"""

import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Union, Awaitable

try:
    from confluent_kafka import Producer, Consumer, KafkaError, KafkaException
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

from .base import IEventBus, Event, EventType, EventHandler, EventFilter


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
        consumer_config: Optional[Dict[str, Any]] = None,
        topic_prefix: str = "forex.",
        num_partitions: int = 3,
        replication_factor: int = 1
    ):
        """
        Initialize Kafka configuration.

        Args:
            bootstrap_servers: Comma-separated list of Kafka bootstrap servers
            service_name: Name of the service using the event bus
            group_id: Consumer group ID (defaults to service_name)
            client_id: Client ID (defaults to service_name)
            auto_create_topics: Whether to automatically create topics
            producer_config: Additional producer configuration
            consumer_config: Additional consumer configuration
            topic_prefix: Prefix for all topics (e.g., "forex.")
            num_partitions: Number of partitions for new topics
            replication_factor: Replication factor for new topics
        """
        self.bootstrap_servers = bootstrap_servers
        self.service_name = service_name
        self.group_id = group_id or service_name
        self.client_id = client_id or service_name
        self.auto_create_topics = auto_create_topics
        self.producer_config = producer_config or {}
        self.consumer_config = consumer_config or {}
        self.topic_prefix = topic_prefix
        self.num_partitions = num_partitions
        self.replication_factor = replication_factor


class KafkaEventBusV2(IEventBus):
    """
    Kafka-based implementation of the event bus interface.

    This class implements the IEventBus interface using Kafka as the underlying message broker.
    It's suitable for production use and provides reliable message delivery and consumption.
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
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize Kafka producer
        producer_config = {
            'bootstrap.servers': config.bootstrap_servers,
            'client.id': f"{config.client_id}-producer",
            'acks': 'all',  # Wait for all replicas to acknowledge
            **config.producer_config
        }
        self._producer = Producer(producer_config)

        # Initialize Kafka admin client
        self._admin_client = AdminClient({'bootstrap.servers': config.bootstrap_servers})

        # Initialize Kafka consumer
        self._consumer = None
        self._consumer_thread = None
        self._running = False
        self._subscribed_topics = set()
        self._event_handlers = {}
        self._handler_unsubscribe_funcs = {}

    async def publish(self, event: Event) -> None:
        """
        Publish an event to the event bus.

        Args:
            event: The event to publish
        """
        if not self._running:
            self._logger.warning("Event bus is not running. Event will not be published.")
            return

        # Convert event to JSON string
        try:
            event_json = event.json()
        except Exception as e:
            self._logger.error(f"Failed to serialize event: {e}")
            return

        # Determine target topic
        topic = self._get_topic_name(event.event_type)

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
        event_types: Union[str, EventType, List[Union[str, EventType]]],
        handler: EventHandler,
        filter_func: Optional[EventFilter] = None
    ) -> Callable[[], None]:
        """
        Subscribe to events of the specified types.

        Args:
            event_types: Type(s) of events to subscribe to
            handler: Function to handle events
            filter_func: Optional function to filter events

        Returns:
            Function to unsubscribe from events
        """
        # Convert single event type to list
        if not isinstance(event_types, list):
            event_types = [event_types]

        # Convert event types to strings
        event_types = [str(et) for et in event_types]

        # Convert event types to topics
        topics = [self._get_topic_name(et) for et in event_types]

        # Register handler for each event type
        for event_type in event_types:
            if event_type not in self._event_handlers:
                self._event_handlers[event_type] = []

            self._event_handlers[event_type].append((handler, filter_func))
            handler_name = getattr(handler, "__name__", str(handler))
            self._logger.debug(f"Subscribed {handler_name} to event type: {event_type}")

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

        # Return unsubscribe function
        def unsubscribe():
            for event_type in event_types:
                self.unsubscribe(event_type, handler)

        return unsubscribe

    def unsubscribe(
        self,
        event_type: Union[str, EventType],
        handler: EventHandler
    ) -> None:
        """
        Unsubscribe from events of a specific type.

        Args:
            event_type: Type of events to unsubscribe from
            handler: Event handler to unsubscribe
        """
        event_type = str(event_type)

        if event_type in self._event_handlers:
            # Find and remove the handler
            to_remove = None
            for h, f in self._event_handlers[event_type]:
                if h == handler:
                    to_remove = (h, f)
                    break

            if to_remove:
                self._event_handlers[event_type].remove(to_remove)
                handler_name = getattr(handler, "__name__", str(handler))
                self._logger.debug(f"Unsubscribed {handler_name} from event type: {event_type}")

                # Remove empty handler lists
                if not self._event_handlers[event_type]:
                    del self._event_handlers[event_type]

                    # Check if we need to unsubscribe from the topic
                    topic = self._get_topic_name(event_type)
                    if topic in self._subscribed_topics:
                        # Check if any other event types use this topic
                        used_by_other_types = False
                        for et in self._event_handlers.keys():
                            if self._get_topic_name(et) == topic:
                                used_by_other_types = True
                                break

                        if not used_by_other_types:
                            self._subscribed_topics.remove(topic)
                            if self._consumer:
                                self._consumer.subscribe(list(self._subscribed_topics))
                                self._logger.debug(f"Unsubscribed from topic: {topic}")

    def subscribe_to_all(
        self,
        handler: EventHandler,
        filter_func: Optional[EventFilter] = None
    ) -> Callable[[], None]:
        """
        Subscribe to all events.

        Args:
            handler: Function to handle events
            filter_func: Optional function to filter events

        Returns:
            Function to unsubscribe from events
        """
        # Not directly supported in Kafka - we need to subscribe to all known event types
        event_types = list(EventType)
        return self.subscribe(event_types, handler, filter_func)

    def unsubscribe_from_all(
        self,
        handler: EventHandler
    ) -> None:
        """
        Unsubscribe from all events.

        Args:
            handler: Event handler to unsubscribe
        """
        # Unsubscribe from all event types
        for event_type in list(self._event_handlers.keys()):
            self.unsubscribe(event_type, handler)

    async def start(self) -> None:
        """
        Start the event bus.

        This method should be called before using the event bus.
        """
        if not self._running:
            self._running = True

            # Initialize consumer if not already done
            if self._consumer is None:
                self._initialize_consumer()

            # Start consumer thread
            self._consumer_thread = threading.Thread(target=self._consume_loop)
            self._consumer_thread.daemon = True
            self._consumer_thread.start()

            self._logger.info("Kafka event bus started")

    async def stop(self) -> None:
        """
        Stop the event bus.

        This method should be called when the event bus is no longer needed.
        """
        if self._running:
            self._running = False

            if self._consumer_thread and self._consumer_thread.is_alive():
                self._consumer_thread.join(timeout=5.0)

            if self._consumer:
                self._consumer.close()
                self._consumer = None

            self._logger.info("Kafka event bus stopped")

    def _initialize_consumer(self) -> None:
        """
        Initialize the Kafka consumer.
        """
        consumer_config = {
            'bootstrap.servers': self.config.bootstrap_servers,
            'group.id': self.config.group_id,
            'client.id': f"{self.config.client_id}-consumer",
            'auto.offset.reset': 'latest',
            'enable.auto.commit': True,
            **self.config.consumer_config
        }

        self._consumer = Consumer(consumer_config)

        if self._subscribed_topics:
            self._consumer.subscribe(list(self._subscribed_topics))
            self._logger.debug(f"Subscribed to topics: {', '.join(self._subscribed_topics)}")

    def _consume_loop(self) -> None:
        """
        Main consumer loop.
        """
        while self._running:
            try:
                # Poll for messages
                msg = self._consumer.poll(1.0)

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # End of partition event
                        self._logger.debug(f"Reached end of partition {msg.partition()}")
                    else:
                        # Error
                        self._logger.error(f"Consumer error: {msg.error()}")
                else:
                    # Process message
                    self._process_message(msg)

            except Exception as e:
                self._logger.error(f"Error in consumer loop: {e}")
                time.sleep(1.0)  # Avoid tight loop in case of persistent errors

    def _process_message(self, msg) -> None:
        """
        Process a Kafka message.

        Args:
            msg: Kafka message
        """
        try:
            # Parse message
            value = msg.value().decode('utf-8')
            event_data = json.loads(value)

            # Get event type from topic
            topic = msg.topic()
            event_type = self._get_event_type_from_topic(topic)

            # Create event object
            # For now, just use the generic Event class
            # In a real implementation, you would have a registry of event classes
            event = Event(
                event_type=event_type,
                payload=event_data.get('payload', {}),
                source_service="unknown"  # This will be overridden by the metadata in the event
            )

            # Call handlers
            handlers = self._event_handlers.get(event_type, [])
            for handler, filter_func in handlers:
                if filter_func is None or filter_func(event):
                    try:
                        # Call handler (note: this is synchronous)
                        # In a real implementation, you might want to use asyncio.create_task
                        # to run handlers asynchronously
                        handler(event)
                    except Exception as e:
                        handler_name = getattr(handler, "__name__", str(handler))
                        self._logger.error(f"Error in event handler {handler_name}: {e}", exc_info=True)

        except Exception as e:
            self._logger.error(f"Error processing message: {e}", exc_info=True)

    def _delivery_callback(self, err, msg) -> None:
        """
        Callback for message delivery reports.

        Args:
            err: Error (if any)
            msg: Message
        """
        if err:
            self._logger.error(f"Message delivery failed: {err}")
        else:
            self._logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")

    def _ensure_topic_exists(self, topic: str) -> None:
        """
        Ensure that a topic exists.

        Args:
            topic: Topic name
        """
        try:
            # Check if topic exists
            metadata = self._admin_client.list_topics(timeout=5)
            if topic in metadata.topics:
                return

            # Create topic
            new_topics = [
                NewTopic(
                    topic,
                    num_partitions=self.config.num_partitions,
                    replication_factor=self.config.replication_factor
                )
            ]

            result = self._admin_client.create_topics(new_topics)

            # Wait for operation to complete
            for topic, future in result.items():
                try:
                    future.result()  # Wait for completion
                    self._logger.info(f"Created topic: {topic}")
                except Exception as e:
                    if "already exists" in str(e):
                        self._logger.debug(f"Topic {topic} already exists")
                    else:
                        self._logger.error(f"Failed to create topic {topic}: {e}")

        except Exception as e:
            self._logger.error(f"Error ensuring topic exists: {e}")

    def _get_topic_name(self, event_type: Union[str, EventType]) -> str:
        """
        Get the topic name for an event type.

        Args:
            event_type: Event type

        Returns:
            Topic name
        """
        event_type_str = str(event_type)
        return f"{self.config.topic_prefix}{event_type_str.replace('.', '_')}"

    def _get_event_type_from_topic(self, topic: str) -> str:
        """
        Get the event type from a topic name.

        Args:
            topic: Topic name

        Returns:
            Event type
        """
        if topic.startswith(self.config.topic_prefix):
            event_type = topic[len(self.config.topic_prefix):]
            return event_type.replace('_', '.')
        return topic
