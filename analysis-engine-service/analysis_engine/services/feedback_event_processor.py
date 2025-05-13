"""
Service for managing feedback event processing via Kafka integration.

This service configures and manages the Kafka consumers for feedback events,
coordinating how different parts of the system respond to feedback-related events.
"""
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from core_foundations.events.kafka import FeedbackEventConsumer, FeedbackEventProducer
from analysis_engine.services.model_retraining_service import ModelRetrainingService
from analysis_engine.services.timeframe_feedback_service import TimeframeFeedbackService
logger = logging.getLogger(__name__)


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class FeedbackEventProcessor:
    """
    Service responsible for processing feedback events from Kafka.
    
    This service:
    1. Sets up Kafka consumers for feedback-related topics
    2. Registers handlers for different event types
    3. Coordinates event-driven model retraining and feedback correlation
    """

    def __init__(self, bootstrap_servers: str, consumer_group_id: str,
        model_retraining_service: ModelRetrainingService,
        timeframe_feedback_service: Optional[TimeframeFeedbackService]=None,
        config: Dict[str, Any]=None):
        """
        Initialize the FeedbackEventProcessor.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            consumer_group_id: Consumer group ID for this processor
            model_retraining_service: Service for model retraining
            timeframe_feedback_service: Optional service for handling timeframe feedback
            config: Configuration options
        """
        self.model_retraining_service = model_retraining_service
        self.timeframe_feedback_service = timeframe_feedback_service
        self.config = config or {}
        self.topics = {'feedback': self.config_manager.get('kafka_topics', {}).get(
            'feedback', 'feedback_events'), 'batch': self.config.get(
            'kafka_topics', {}).get('batch', 'feedback_batch_events'),
            'model': self.config_manager.get('kafka_topics', {}).get('model',
            'model_events')}
        self.consumers = {}
        self.consumers['feedback'] = FeedbackEventConsumer(bootstrap_servers
            =bootstrap_servers, group_id=f'{consumer_group_id}-feedback',
            topics=[self.topics['feedback']], config=self.config.get(
            'kafka_consumer_config', {}))
        self.consumers['batch'] = FeedbackEventConsumer(bootstrap_servers=
            bootstrap_servers, group_id=f'{consumer_group_id}-batch',
            topics=[self.topics['batch']], config=self.config.get(
            'kafka_consumer_config', {}))
        self.consumer_threads = {}
        self._register_handlers()
        logger.info(
            'FeedbackEventProcessor initialized with Kafka bootstrap servers: %s'
            , bootstrap_servers)

    def start(self):
        """
        Start consuming and processing feedback events.
        """
        for topic_type, consumer in self.consumers.items():
            self.consumer_threads[topic_type] = threading.Thread(target=
                consumer.start, daemon=True, name=
                f'kafka-consumer-{topic_type}')
            self.consumer_threads[topic_type].start()
            logger.info('Started Kafka consumer thread for %s events',
                topic_type)

    def stop(self):
        """
        Stop all consumers and processing threads.
        """
        for topic_type, consumer in self.consumers.items():
            consumer.stop()
            logger.info('Stopped Kafka consumer for %s events', topic_type)
        for topic_type, thread in self.consumer_threads.items():
            if thread.is_alive():
                thread.join(timeout=5.0)
                logger.info('Consumer thread for %s events has terminated',
                    topic_type)

    def _register_handlers(self):
        """
        Register event handlers for different event types.
        """
        feedback_consumer = self.consumers.get('feedback')
        if feedback_consumer:
            feedback_consumer.register_handler('feedback_created', self.
                _handle_feedback_created)
            if self.timeframe_feedback_service:
                feedback_consumer.register_handler('feedback_created', self
                    ._handle_timeframe_feedback)
        batch_consumer = self.consumers.get('batch')
        if batch_consumer:
            batch_consumer.register_handler('feedback_batch_created', self.
                _handle_batch_created)
            batch_consumer.register_handler('feedback_batch_processed',
                self._handle_batch_processed)

    @with_exception_handling
    def _handle_feedback_created(self, event_data: Dict[str, Any]):
        """
        Handle creation of a new feedback item.
        
        Args:
            event_data: Event data from Kafka message
        """
        logger.debug('Processing feedback_created event for feedback %s',
            event_data.get('feedback_id'))
        if event_data.get('priority') in ('HIGH', 'CRITICAL'):
            model_id = event_data.get('model_id')
            if model_id:
                logger.info(
                    'High priority feedback detected for model %s. Checking if retraining is needed.'
                    , model_id)
                try:
                    if self.config.get('enable_immediate_retraining_check',
                        True):
                        self.model_retraining_service.check_and_trigger_retraining(
                            model_id)
                except Exception as e:
                    logger.error('Error checking retraining for model %s: %s',
                        model_id, str(e))

    @with_exception_handling
    def _handle_timeframe_feedback(self, event_data: Dict[str, Any]):
        """
        Handle timeframe-specific feedback processing.
        
        Args:
            event_data: Event data from Kafka message
        """
        if not self.timeframe_feedback_service:
            return
        if 'timeframe' not in event_data:
            return
        model_id = event_data.get('model_id')
        timeframe = event_data.get('timeframe')
        if model_id and timeframe:
            logger.debug(
                'Processing timeframe feedback for model %s, timeframe %s',
                model_id, timeframe)
            if self.config_manager.get('auto_correlate_timeframes', True):
                try:
                    logger.info(
                        'Would trigger timeframe correlation analysis for model %s, timeframe %s'
                        , model_id, timeframe)
                except Exception as e:
                    logger.error(
                        'Error correlating timeframes for model %s: %s',
                        model_id, str(e))

    def _handle_batch_created(self, event_data: Dict[str, Any]):
        """
        Handle creation of a new feedback batch.
        
        Args:
            event_data: Event data from Kafka message
        """
        batch_id = event_data.get('batch_id')
        model_id = event_data.get('model_id')
        logger.info('Processing batch_created event for batch %s (model: %s)',
            batch_id, model_id)
        if model_id and self.config_manager.get('auto_process_batches', True):
            logger.info(
                'Auto-processing of batch %s is enabled. Would prepare for retraining.'
                , batch_id)

    def _handle_batch_processed(self, event_data: Dict[str, Any]):
        """
        Handle completion of batch processing.
        
        Args:
            event_data: Event data from Kafka message
        """
        batch_id = event_data.get('batch_id')
        status = event_data.get('status')
        model_id = event_data.get('model_id')
        new_version = event_data.get('new_model_version')
        logger.info(
            'Batch %s processed with status: %s. Model %s updated to version %s.'
            , batch_id, status, model_id, new_version or 'N/A')
        if status == 'success' and new_version:
            logger.info(
                'Would trigger deployment workflow for model %s version %s',
                model_id, new_version)
