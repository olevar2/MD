\
""
Handles publishing events to the message broker (Kafka).
"""

import json
import logging
from kafka import KafkaProducer
from kafka.errors import KafkaError
from analysis_engine.config import settings
from analysis_engine.events.schemas import BaseEvent

logger = logging.getLogger(__name__)

class EventPublisher:
    """Handles publishing events to Kafka."""

    def __init__(self):
        self.producer = None
        self._connect()

    def _connect(self):
        """Establishes connection to the Kafka broker."""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v.dict(), default=str).encode('utf-8'),
                acks='all', # Ensure messages are acknowledged by all in-sync replicas
                retries=3, # Retry sending messages up to 3 times
                linger_ms=10 # Batch messages for 10ms
            )
            logger.info("Successfully connected to Kafka broker for publishing.")
        except KafkaError as e:
            logger.error(f"Failed to connect Kafka producer: {e}", exc_info=True)
            # Implement retry logic or failover if necessary
            self.producer = None

    def publish(self, topic: str, event: BaseEvent):
        """Publishes an event to the specified Kafka topic."""
        if not self.producer:
            logger.error("Kafka producer is not connected. Attempting to reconnect...")
            self._connect()
            if not self.producer:
                logger.error("Failed to publish event: Kafka producer unavailable.")
                # Optionally, queue the event for later or raise an exception
                return

        try:
            future = self.producer.send(topic, value=event, key=str(event.event_id).encode('utf-8'))
            # Block for 'synchronous' sends
            record_metadata = future.get(timeout=10)
            logger.info(
                f"Successfully published event {event.event_id} ({event.event_type}) "
                f"to topic '{record_metadata.topic}' partition {record_metadata.partition} "
                f"at offset {record_metadata.offset}"
            )
        except KafkaError as e:
            logger.error(f"Failed to publish event {event.event_id} to topic {topic}: {e}", exc_info=True)
            # Handle specific Kafka errors (e.g., message too large, broker unavailable)
        except Exception as e:
            logger.error(f"An unexpected error occurred during event publishing: {e}", exc_info=True)

    def close(self):
        """Closes the Kafka producer connection."""
        if self.producer:
            self.producer.flush() # Ensure all buffered messages are sent
            self.producer.close()
            logger.info("Kafka producer connection closed.")

# Global instance (consider dependency injection for better testability)
# event_publisher = EventPublisher()

# Example Usage (within service logic):
# from analysis_engine.events.schemas import AnalysisCompletionEvent, AnalysisCompletionPayload
# from analysis_engine.events.publisher import event_publisher
#
# payload = AnalysisCompletionPayload(
#     analysis_id="123", symbol="EURUSD", timeframe="H1", status="completed", results_summary={...}
# )
# event = AnalysisCompletionEvent(payload=payload)
# event_publisher.publish(topic="analysis_events", event=event)