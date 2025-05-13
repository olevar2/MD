"""
Handles publishing events to the message broker (Kafka).
"""
import json
import logging
from kafka import KafkaProducer
from kafka.errors import KafkaError
from analysis_engine.config import settings
from analysis_engine.events.schemas import BaseEvent
logger = logging.getLogger(__name__)


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class EventPublisher:
    """Handles publishing events to Kafka."""

    def __init__(self):
        self.producer = None
        self._connect()

    @with_exception_handling
    def _connect(self):
        """Establishes connection to the Kafka broker."""
        try:
            self.producer = KafkaProducer(bootstrap_servers=settings.
                KAFKA_BOOTSTRAP_SERVERS, value_serializer=lambda v: json.
                dumps(v.dict(), default=str).encode('utf-8'), acks='all',
                retries=3, linger_ms=10)
            logger.info(
                'Successfully connected to Kafka broker for publishing.')
        except KafkaError as e:
            logger.error(f'Failed to connect Kafka producer: {e}', exc_info
                =True)
            self.producer = None

    @with_exception_handling
    def publish(self, topic: str, event: BaseEvent):
        """Publishes an event to the specified Kafka topic."""
        if not self.producer:
            logger.error(
                'Kafka producer is not connected. Attempting to reconnect...')
            self._connect()
            if not self.producer:
                logger.error(
                    'Failed to publish event: Kafka producer unavailable.')
                return
        try:
            future = self.producer.send(topic, value=event, key=str(event.
                event_id).encode('utf-8'))
            record_metadata = future.get(timeout=10)
            logger.info(
                f"Successfully published event {event.event_id} ({event.event_type}) to topic '{record_metadata.topic}' partition {record_metadata.partition} at offset {record_metadata.offset}"
                )
        except KafkaError as e:
            logger.error(
                f'Failed to publish event {event.event_id} to topic {topic}: {e}'
                , exc_info=True)
        except Exception as e:
            logger.error(
                f'An unexpected error occurred during event publishing: {e}',
                exc_info=True)

    def close(self):
        """Closes the Kafka producer connection."""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info('Kafka producer connection closed.')
