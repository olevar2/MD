"""
Kafka event bus integration for the feedback system.

This module provides components for publishing and consuming feedback-related
events via Kafka, facilitating asynchronous communication between services.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Callable, Optional

# Kafka client library
try:
    from confluent_kafka import Producer, Consumer, KafkaError, KafkaException
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    # Define a stub class for type hints to work without the dependency
    class Producer:
        pass
    class Consumer:
        pass

logger = logging.getLogger(__name__)


class FeedbackEventProducer:
    """
    Publishes feedback-related events to Kafka topics.
    
    This producer handles serializing event data and reliably publishing
    to configured topics with appropriate metadata and headers.
    """
    
    def __init__(
        self, 
        bootstrap_servers: str,
        client_id: str = None,
        config: Dict[str, Any] = None,
        default_topic: str = "feedback_events"
    ):
        """
        Initialize the Kafka feedback event producer.
        
        Args:
            bootstrap_servers: Comma-separated list of Kafka broker addresses
            client_id: Identifier for this producer client
            config: Additional Kafka producer configuration
            default_topic: Default topic to produce to if none specified
        """
        if not KAFKA_AVAILABLE:
            logger.error("confluent-kafka library not installed. Kafka functionality is disabled.")
            self.producer = None
            return
            
        self.default_topic = default_topic
        self.client_id = client_id or f"feedback-producer-{str(uuid.uuid4())[:8]}"
        
        # Combine configuration
        producer_config = {
            'bootstrap.servers': bootstrap_servers,
            'client.id': self.client_id,
            # Default to appropriate serialization and compression
            'compression.type': 'snappy',
            # Enable idempotent producer for exactly-once semantics
            'enable.idempotence': True,
            # Configure reasonable default acknowledgment strategy
            'acks': 'all',
            # Error handling and retry settings
            'delivery.timeout.ms': 10000,  # 10 seconds
            'request.timeout.ms': 5000,    # 5 seconds
            'retry.backoff.ms': 100        # 100 ms between retries
        }
        
        # Apply custom config if provided
        if config:
            producer_config.update(config)
            
        logger.info("Initializing Kafka producer with bootstrap servers: %s", bootstrap_servers)
        self.producer = Producer(producer_config)
    
    def produce(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        topic: str = None,
        key: str = None,
        partition: int = None,
        headers: Dict[str, str] = None
    ) -> bool:
        """
        Produce a feedback event to Kafka.
        
        Args:
            event_type: The type of event (e.g., 'feedback_created', 'model_retrained')
            event_data: The payload/content of the event
            topic: The Kafka topic to publish to (defaults to self.default_topic)
            key: Optional key for the message (for partitioning)
            partition: Optional explicit partition to publish to
            headers: Optional message headers
            
        Returns:
            bool: True if the message was queued successfully, False otherwise
        """
        if not self.producer:
            logger.error("Kafka producer not available. Event not published: %s", event_type)
            return False
            
        # Use default topic if none provided
        target_topic = topic or self.default_topic
        
        # Set the message key for partitioning if not provided
        # Default to using a UUID representing the event instance
        message_key = key or str(uuid.uuid4())
        
        # Prepare headers if provided
        kafka_headers = []
        if headers:
            kafka_headers = [(k, v.encode('utf-8') if isinstance(v, str) else v) 
                            for k, v in headers.items()]
        
        # Add standard event metadata
        event_envelope = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "producer_id": self.client_id,
            "event_id": str(uuid.uuid4()),
            "data": event_data
        }
        
        # Serialize to JSON
        try:
            message_value = json.dumps(event_envelope).encode('utf-8')
        except (TypeError, ValueError) as e:
            logger.error("Failed to serialize event data: %s", str(e))
            return False
        
        try:
            # Produce the message
            self.producer.produce(
                topic=target_topic,
                value=message_value,
                key=message_key.encode('utf-8') if isinstance(message_key, str) else message_key,
                partition=partition,
                headers=kafka_headers,
                on_delivery=self._delivery_callback
            )
            
            # Poll to trigger delivery reports
            self.producer.poll(0)
            return True
            
        except KafkaException as e:
            logger.error("Failed to produce Kafka message: %s", str(e))
            return False
    
    def flush(self, timeout: float = 10.0) -> int:
        """
        Flush all pending messages and wait for completion.
        
        Args:
            timeout: Maximum time to wait for all messages to be delivered
            
        Returns:
            Number of messages still in queue (0 if all were delivered)
        """
        if not self.producer:
            return 0
        
        return self.producer.flush(timeout)
    
    def _delivery_callback(self, err, msg):
        """Callback executed on message delivery."""
        if err:
            logger.error("Message delivery failed: %s", str(err))
        else:
            logger.debug("Message delivered to %s [%d] at offset %d",
                       msg.topic(), msg.partition(), msg.offset())


class FeedbackEventConsumer:
    """
    Consumes and processes feedback-related events from Kafka topics.
    
    This consumer handles deserializing event data and executing
    registered handlers for specific event types.
    """
    
    def __init__(
        self,
        bootstrap_servers: str,
        group_id: str,
        topics: List[str],
        config: Dict[str, Any] = None,
        auto_commit: bool = True
    ):
        """
        Initialize the Kafka feedback event consumer.
        
        Args:
            bootstrap_servers: Comma-separated list of Kafka broker addresses
            group_id: Consumer group ID for this consumer
            topics: List of topics to subscribe to
            config: Additional Kafka consumer configuration
            auto_commit: Whether to auto-commit offsets
        """
        if not KAFKA_AVAILABLE:
            logger.error("confluent-kafka library not installed. Kafka functionality is disabled.")
            self.consumer = None
            return
            
        self.topics = topics
        self.auto_commit = auto_commit
        self.running = False
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Default consumer configuration
        consumer_config = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': auto_commit,
            # Ensure reasonable commit intervals if auto-commit is enabled
            'auto.commit.interval.ms': 5000,
            # Add client identifier
            'client.id': f"feedback-consumer-{str(uuid.uuid4())[:8]}"
        }
        
        # Apply custom config if provided
        if config:
            consumer_config.update(config)
            
        logger.info("Initializing Kafka consumer with group ID '%s' for topics %s", 
                   group_id, topics)
        self.consumer = Consumer(consumer_config)
    
    def register_handler(self, event_type: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a handler function for a specific event type.
        
        Args:
            event_type: The type of event to handle
            handler: Function that takes the event data as input and processes it
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
            
        self.event_handlers[event_type].append(handler)
        logger.info("Registered handler for event type '%s'", event_type)
    
    def start(self, timeout: float = 1.0) -> None:
        """
        Start consuming messages in a loop.
        
        Args:
            timeout: Poll timeout in seconds
        """
        if not self.consumer:
            logger.error("Kafka consumer not available. Cannot start consumption.")
            return
            
        try:
            # Subscribe to the topics
            self.consumer.subscribe(self.topics)
            self.running = True
            
            logger.info("Starting Kafka consumption loop for topics: %s", self.topics)
            
            while self.running:
                try:
                    # Poll for messages
                    msg = self.consumer.poll(timeout)
                    
                    if msg is None:
                        continue
                        
                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            # End of partition event - not an error
                            logger.debug("Reached end of partition")
                        else:
                            # Actual error
                            logger.error("Error during message consumption: %s", msg.error())
                    else:
                        # Process the message
                        self._process_message(msg)
                        
                        # Commit manually if auto-commit is disabled
                        if not self.auto_commit:
                            self.consumer.commit(msg, asynchronous=False)
                            
                except Exception as e:
                    logger.exception("Error processing Kafka message: %s", str(e))
                    
        except KeyboardInterrupt:
            logger.info("Kafka consumption interrupted")
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop consuming messages and close the consumer."""
        self.running = False
        
        if self.consumer:
            logger.info("Closing Kafka consumer")
            self.consumer.close()
    
    def _process_message(self, msg) -> None:
        """
        Process a received Kafka message.
        
        Args:
            msg: The Kafka message object
        """
        try:
            # Parse the JSON payload
            payload = json.loads(msg.value().decode('utf-8'))
            
            # Extract event type and data
            event_type = payload.get('event_type')
            event_data = payload.get('data', {})
            
            if not event_type:
                logger.warning("Received message with no event type")
                return
                
            # Find and execute registered handlers
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    try:
                        handler(event_data)
                    except Exception as e:
                        logger.exception("Handler for event '%s' failed: %s", event_type, str(e))
            else:
                logger.debug("No handler registered for event type: %s", event_type)
                
        except json.JSONDecodeError as e:
            logger.error("Failed to parse message payload as JSON: %s", str(e))
        except Exception as e:
            logger.exception("Error processing message: %s", str(e))
