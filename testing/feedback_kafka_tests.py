"""
Test suite for the feedback event processing system using Kafka.

These tests verify that the Kafka integration correctly handles
publishing and consuming feedback-related events across the system.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import json
from datetime import datetime, timedelta

from core_foundations.events.kafka import FeedbackEventProducer, FeedbackEventConsumer
from core_foundations.models.feedback import (
    ClassifiedFeedback, FeedbackBatch, TimeframeFeedback,
    FeedbackPriority, FeedbackCategory, FeedbackSource, FeedbackStatus
)
from analysis_engine.repositories.feedback_repository import FeedbackRepository
from analysis_engine.services.feedback_event_processor import FeedbackEventProcessor


class TestFeedbackKafkaIntegration(unittest.TestCase):
    """Test cases for the Kafka integration with the feedback system."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock the Kafka Producer/Consumer since we don't want to connect to actual Kafka in tests
        self.mock_kafka_producer = Mock()
        self.mock_kafka_producer.produce.return_value = True
        self.mock_kafka_producer.flush.return_value = 0
        
        # Patch the Kafka client classes
        self.producer_patcher = patch('core_foundations.events.kafka.Producer', return_value=self.mock_kafka_producer)
        self.producer_mock = self.producer_patcher.start()
        
        # Configuration
        self.kafka_config = {
            'bootstrap_servers': 'localhost:9092',
            'client_id': 'test-client',
            'kafka_topics': {
                'feedback': 'test-feedback-topic',
                'batch': 'test-batch-topic',
                'model': 'test-model-topic'
            }
        }
    
    def tearDown(self):
        """Clean up after each test method."""
        self.producer_patcher.stop()
    
    def test_feedback_event_producer_initialization(self):
        """Test that the FeedbackEventProducer initializes correctly."""
        # Create a producer instance
        producer = FeedbackEventProducer(
            bootstrap_servers='localhost:9092',
            client_id='test-client',
            default_topic='test-topic'
        )
        
        # Verify the Kafka Producer was initialized
        self.producer_mock.assert_called_once()
        self.assertIsNotNone(producer.producer)
        self.assertEqual('test-topic', producer.default_topic)
    
    def test_feedback_event_producer_produce(self):
        """Test that events are produced correctly."""
        # Create a producer instance
        producer = FeedbackEventProducer(
            bootstrap_servers='localhost:9092',
            client_id='test-client'
        )
        
        # Test data
        event_type = 'feedback_created'
        event_data = {
            'feedback_id': 'test-feedback-123',
            'model_id': 'test-model-456',
            'priority': 'HIGH'
        }
        
        # Produce an event
        result = producer.produce(
            event_type=event_type,
            event_data=event_data,
            topic='test-topic',
            key='test-key'
        )
        
        # Verify results
        self.assertTrue(result)
        self.mock_kafka_producer.produce.assert_called_once()
        
        # Verify the event envelope structure
        call_args = self.mock_kafka_producer.produce.call_args
        self.assertEqual('test-topic', call_args.kwargs['topic'])
        self.assertEqual(b'test-key', call_args.kwargs['key'])
        
        # Parse the message value to verify the envelope
        message_value = json.loads(call_args.kwargs['value'].decode('utf-8'))
        self.assertEqual(event_type, message_value['event_type'])
        self.assertEqual(event_data, message_value['data'])
        self.assertIn('timestamp', message_value)
        self.assertIn('event_id', message_value)
    
    def test_feedback_repository_emit_events(self):
        """Test that the FeedbackRepository emits Kafka events correctly."""
        # Create a mock db client
        mock_db_client = Mock()
        
        # Create a FeedbackEventProducer with mocked Kafka Producer
        event_producer = FeedbackEventProducer(
            bootstrap_servers='localhost:9092',
            client_id='test-client'
        )
        
        # Create the repository
        repository = FeedbackRepository(
            db_client=mock_db_client,
            event_producer=event_producer,
            config=self.kafka_config
        )
        
        # Create a test feedback item
        feedback = ClassifiedFeedback(
            feedback_id='test-feedback-123',
            model_id='test-model-456',
            category=FeedbackCategory.INCORRECT_PREDICTION,
            priority=FeedbackPriority.HIGH,
            source=FeedbackSource.SYSTEM_AUTO,
            status=FeedbackStatus.NEW,
            content={'prediction_error': 0.25}
        )
        
        # Store the feedback to trigger event emission
        with patch.object(repository, '_db_store_feedback', return_value=feedback.feedback_id):
            repository.store_feedback(feedback)
            
            # Verify an event was produced
            self.mock_kafka_producer.produce.assert_called_once()
            
            # Check the event data
            call_args = self.mock_kafka_producer.produce.call_args
            message_value = json.loads(call_args.kwargs['value'].decode('utf-8'))
            self.assertEqual('feedback_created', message_value['event_type'])
            self.assertEqual(feedback.feedback_id, message_value['data']['feedback_id'])
            self.assertEqual(feedback.model_id, message_value['data']['model_id'])
            self.assertEqual(feedback.priority.value, message_value['data']['priority'])
    
    def test_batch_events(self):
        """Test that batch events are properly emitted."""
        # Create a mock db client
        mock_db_client = Mock()
        
        # Create a FeedbackEventProducer with mocked Kafka Producer
        event_producer = FeedbackEventProducer(
            bootstrap_servers='localhost:9092',
            client_id='test-client'
        )
        
        # Create the repository
        repository = FeedbackRepository(
            db_client=mock_db_client,
            event_producer=event_producer,
            config=self.kafka_config
        )
        
        # Create test feedback items
        feedback_items = [
            ClassifiedFeedback(
                feedback_id=f'test-feedback-{i}',
                model_id='test-model-456',
                category=FeedbackCategory.INCORRECT_PREDICTION,
                priority=FeedbackPriority.MEDIUM,
                status=FeedbackStatus.NEW
            ) for i in range(3)
        ]
        
        # Create a test batch
        batch = FeedbackBatch(
            batch_id='test-batch-789',
            feedback_items=feedback_items,
            batch_priority=FeedbackPriority.MEDIUM,
            created_at=datetime.utcnow(),
            status='NEW',
            metadata={'model_id': 'test-model-456'}
        )
        
        # Emit batch created event
        repository._emit_batch_created_event(batch)
        
        # Verify event was produced with correct data
        self.mock_kafka_producer.produce.assert_called_once()
        
        # Reset mock for next test
        self.mock_kafka_producer.produce.reset_mock()
        
        # Test batch processed event
        processing_results = {
            'status': 'success',
            'model_version': '1.2',
            'metrics': {'accuracy': 0.95}
        }
        
        # Update batch and emit processed event
        batch.status = 'PROCESSED'
        batch.processed_at = datetime.utcnow()
        repository._emit_batch_processed_event(batch, processing_results)
        
        # Verify events were produced (should be two: batch processed and model updated)
        self.assertEqual(2, self.mock_kafka_producer.produce.call_count)
        
        # Verify first call was for batch processed event
        first_call = self.mock_kafka_producer.produce.call_args_list[0]
        batch_message = json.loads(first_call.kwargs['value'].decode('utf-8'))
        self.assertEqual('feedback_batch_processed', batch_message['event_type'])
        
        # Verify second call was for model updated event
        second_call = self.mock_kafka_producer.produce.call_args_list[1]
        model_message = json.loads(second_call.kwargs['value'].decode('utf-8'))
        self.assertEqual('model_updated', model_message['event_type'])
        
    @patch('threading.Thread')
    def test_feedback_event_processor(self, mock_thread_class):
        """Test the FeedbackEventProcessor service."""
        # Create mocks for consumers
        mock_feedback_consumer = Mock()
        mock_batch_consumer = Mock()
        
        # Create mocks for services
        mock_model_retraining_service = Mock()
        mock_timeframe_feedback_service = Mock()
        
        # Create processor with mocked components
        with patch('core_foundations.events.kafka.FeedbackEventConsumer') as mock_consumer_class:
            # Configure mock consumer class to return our mocks
            mock_consumer_class.side_effect = [mock_feedback_consumer, mock_batch_consumer]
            
            # Create the processor
            processor = FeedbackEventProcessor(
                bootstrap_servers='localhost:9092',
                consumer_group_id='test-group',
                model_retraining_service=mock_model_retraining_service,
                timeframe_feedback_service=mock_timeframe_feedback_service,
                config={
                    'kafka_topics': {
                        'feedback': 'test-feedback-topic',
                        'batch': 'test-batch-topic',
                        'model': 'test-model-topic'
                    },
                    'enable_immediate_retraining_check': True
                }
            )
            
            # Verify consumers were created
            self.assertEqual(2, mock_consumer_class.call_count)
            self.assertEqual(
                'test-feedback-topic', 
                mock_consumer_class.call_args_list[0].kwargs['topics'][0]
            )
            self.assertEqual(
                'test-batch-topic', 
                mock_consumer_class.call_args_list[1].kwargs['topics'][0]
            )
            
            # Verify handlers were registered
            self.assertTrue(mock_feedback_consumer.register_handler.called)
            self.assertTrue(mock_batch_consumer.register_handler.called)
            
            # Start the processor
            processor.start()
            
            # Verify threads were started
            self.assertEqual(2, mock_thread_class.call_count)
            
            # Test handling a high-priority feedback event
            high_priority_event = {
                'feedback_id': 'test-feedback-123',
                'model_id': 'test-model-456',
                'priority': 'HIGH',
                'category': 'INCORRECT_PREDICTION'
            }
            
            # Manually call the handler to test
            processor._handle_feedback_created(high_priority_event)
            
            # Verify retraining was checked due to high priority
            mock_model_retraining_service.check_and_trigger_retraining.assert_called_once_with('test-model-456')
            
            # Test handling a timeframe feedback event
            timeframe_event = {
                'feedback_id': 'test-feedback-789',
                'model_id': 'test-model-456',
                'timeframe': '1h',
                'category': 'TIMEFRAME_ADJUSTMENT'
            }
            
            # Reset mock and call handler
            mock_model_retraining_service.check_and_trigger_retraining.reset_mock()
            processor._handle_timeframe_feedback(timeframe_event)
            
            # Verify timeframe handling (our implementation logs but doesn't actually call the service)
            mock_model_retraining_service.check_and_trigger_retraining.assert_not_called()


class TestFeedbackRepositoryMemoryImplementation(unittest.TestCase):
    """Test the file-based implementation of the FeedbackRepository."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        import tempfile
        import shutil
        
        # Create a temporary directory for file storage
        self.test_dir = tempfile.mkdtemp()
        
        # Configuration for file-based storage
        self.config = {
            'use_file_storage': True,
            'storage_dir': self.test_dir
        }
        
        # Create the repository with file storage
        self.repository = FeedbackRepository(
            db_client=None,  # Not needed for file storage
            config=self.config
        )
    
    def tearDown(self):
        """Clean up after each test method."""
        import shutil
        # Remove the temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_file_storage_and_retrieval(self):
        """Test that feedback can be stored and retrieved from files."""
        # Create test feedback items
        feedback_items = []
        for i in range(5):
            feedback = ClassifiedFeedback(
                feedback_id=f'test-feedback-{i}',
                model_id='test-model-456',
                category=FeedbackCategory.INCORRECT_PREDICTION,
                priority=FeedbackPriority.MEDIUM if i < 3 else FeedbackPriority.HIGH,
                source=FeedbackSource.SYSTEM_AUTO,
                status=FeedbackStatus.NEW,
                statistical_significance=0.7 + (i * 0.05),
                timestamp=datetime.utcnow() - timedelta(days=i),
                content={'prediction_error': 0.1 * i}
            )
            feedback_items.append(feedback)
            self.repository.store_feedback(feedback)
        
        # Create TimeframeFeedback item
        tf_feedback = TimeframeFeedback(
            feedback_id='test-tf-feedback',
            timeframe='1h',
            related_timeframes=['4h'],
            model_id='test-model-456',
            category=FeedbackCategory.TIMEFRAME_ADJUSTMENT,
            priority=FeedbackPriority.HIGH,
            source=FeedbackSource.PERFORMANCE_METRICS,
            status=FeedbackStatus.NEW,
            statistical_significance=0.85,
            timestamp=datetime.utcnow(),
            content={'prediction_error': 0.22}
        )
        self.repository.store_feedback(tf_feedback)
        
        # Retrieve high priority feedback
        high_priority_items = self.repository.get_prioritized_feedback_since(
            timestamp=datetime.utcnow() - timedelta(days=10),
            min_priority='HIGH'
        )
        
        # Verify we got the expected items
        self.assertEqual(3, len(high_priority_items))  # 2 high priority + 1 timeframe feedback
        
        # Test batch creation
        feedback_ids = [item.feedback_id for item in feedback_items[:3]]
        batch_id = self.repository.create_feedback_batch(
            feedback_ids=feedback_ids,
            batch_metadata={'model_id': 'test-model-456'}
        )
        
        # Verify batch was created
        self.assertTrue(batch_id)
        
        # Get the batch
        batch = self.repository.get_feedback_batch(batch_id)
        
        # Verify batch contents
        self.assertEqual(batch_id, batch.batch_id)
        self.assertEqual(3, len(batch.feedback_items))
        self.assertEqual('NEW', batch.status)
        self.assertEqual('test-model-456', batch.metadata.get('model_id'))
        
        # Mark the batch as processed
        result = self.repository.mark_batch_processed(
            batch_id=batch_id,
            processing_results={
                'status': 'success',
                'model_version': '1.3',
                'metrics': {'accuracy': 0.96}
            }
        )
        
        # Verify batch was updated
        self.assertTrue(result)
        
        # Get the updated batch
        updated_batch = self.repository.get_feedback_batch(batch_id)
        
        # Verify status was updated
        self.assertEqual('PROCESSED', updated_batch.status)
        self.assertEqual('1.3', updated_batch.metadata.get('model_version'))


if __name__ == '__main__':
    unittest.main()
