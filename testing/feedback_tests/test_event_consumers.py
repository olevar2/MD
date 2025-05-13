# filepath: d:\MD\forex_trading_platform\testing\feedback_tests\test_event_consumers.py
"""
Unit and integration tests for the FeedbackConsumer in event_consumers.py
"""

import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock

# Assuming structure allows importing from analysis_engine and core_foundations
from analysis_engine.adaptive_layer.event_consumers import FeedbackConsumer
from core_foundations.models.feedback_models import TradeFeedback, TradeFeedbackData

# Mock KafkaConsumer from kafka-python
# If using aiokafka, mock AIOKafkaConsumer instead
@pytest.fixture
def mock_kafka_consumer():
    """
    Mock kafka consumer.
    
    """

    with patch('analysis_engine.adaptive_layer.event_consumers.KafkaConsumer', new_callable=MagicMock) as mock_consumer_class:
        mock_consumer_instance = mock_consumer_class.return_value
        mock_consumer_instance.poll = MagicMock()
        mock_consumer_instance.commit = MagicMock()
        mock_consumer_instance.close = MagicMock()
        yield mock_consumer_instance

@pytest.fixture
def feedback_consumer(mock_kafka_consumer):
    """
    Feedback consumer.
    
    Args:
        mock_kafka_consumer: Description of mock_kafka_consumer
    
    """
 # Depends on the mock fixture
    consumer = FeedbackConsumer(
        bootstrap_servers='mock_server:9092',
        group_id='test_group',
        topics=['trade_feedback']
    )
    # Inject the mocked instance
    consumer._consumer = mock_kafka_consumer 
    # Mock the processing methods it calls (replace with actual mocks if needed)
    consumer._process_message = AsyncMock() 
    return consumer

@pytest.mark.asyncio
async def test_consumer_initialization(mock_kafka_consumer):
    """Test that the consumer initializes the underlying KafkaConsumer correctly."""
    consumer = FeedbackConsumer(
        bootstrap_servers='mock_server:9092',
        group_id='test_group',
        topics=['topic1', 'topic2']
    )
    consumer._initialize_consumer() # Call the internal init method
    
    # Assert KafkaConsumer was called with correct args
    from analysis_engine.adaptive_layer.event_consumers import KafkaConsumer # Import locally for assertion
    KafkaConsumer.assert_called_once()
    call_args, call_kwargs = KafkaConsumer.call_args
    assert 'topic1' in call_args
    assert 'topic2' in call_args
    assert call_kwargs.get('bootstrap_servers') == 'mock_server:9092'
    assert call_kwargs.get('group_id') == 'test_group'
    assert callable(call_kwargs.get('value_deserializer'))

@pytest.mark.asyncio
async def test_consumer_run_loop_polls_messages(feedback_consumer, mock_kafka_consumer):
    """Test that the run loop polls for messages and processes them."""
    # Simulate messages returned by poll
    mock_message = MagicMock()
    mock_message.topic = 'trade_feedback'
    mock_message.key = b'key1'
    # Create a valid TradeFeedback payload
    trade_data = TradeFeedbackData(trade_id='t1', strategy_id='s1', symbol='EURUSD', outcome='profit', pnl=10)
    feedback_obj = TradeFeedback(source='test', data=trade_data)
    mock_message.value = json.dumps(feedback_obj.dict(exclude={'feedback_id', 'timestamp'})).encode('utf-8') # Simulate JSON value

    # Configure poll to return messages once, then empty dicts
    mock_kafka_consumer.poll.side_effect = [
        {TopicPartition('trade_feedback', 0): [mock_message]}, # First poll returns message
        {}, # Subsequent polls return nothing
        # Add KeyboardInterrupt or other exception to stop loop if needed for test
    ]

    # Run the consumer for a short time in a separate task
    consumer_task = asyncio.create_task(feedback_consumer.run())
    
    # Allow loop to run a few iterations
    await asyncio.sleep(0.5) 
    
    # Assert poll was called
    mock_kafka_consumer.poll.assert_called()
    
    # Assert _process_message was called (using the already mocked version)
    feedback_consumer._process_message.assert_called()
    # Can add more specific assertions on the arguments passed to _process_message if needed

    # Stop the consumer
    feedback_consumer.stop()
    await consumer_task # Wait for the task to finish
    mock_kafka_consumer.close.assert_called_once()

@pytest.mark.asyncio
async def test_process_message_deserialization_and_routing():
    """Test the internal _process_message logic (requires unmocking it)."""
    consumer = FeedbackConsumer(bootstrap_servers='mock', group_id='test', topics=['trade_feedback'])
    # Mock the downstream processing calls
    # consumer.feedback_loop = AsyncMock() # Example if routing to feedback_loop
    # consumer.trading_feedback_collector = AsyncMock() # Example
    
    # --- Test TradeFeedback --- 
    trade_data = TradeFeedbackData(trade_id='t1', strategy_id='s1', symbol='EURUSD', outcome='profit', pnl=10)
    feedback_obj = TradeFeedback(source='test', data=trade_data)
    # Manually create a dict matching the expected JSON structure (excluding defaults)
    raw_message_value = {
        "source": "test",
        "feedback_type": "trade",
        "data": {
            "trade_id": "t1",
            "strategy_id": "s1",
            "symbol": "EURUSD",
            "outcome": "profit",
            "pnl": 10.0,
            "slippage": None,
            "execution_time_ms": None,
            "market_conditions": None
        }
    }

    # Patch the logger inside the function's scope if needed to check logs
    with patch('analysis_engine.adaptive_layer.event_consumers.logger') as mock_logger:
        await consumer._process_message(topic='trade_feedback', key='k1', value=raw_message_value)
        # Assert that the correct processing logic was called (or logged, based on implementation)
        mock_logger.info.assert_any_call(f"Successfully deserialized feedback event: {feedback_obj.feedback_id} (Type: trade)")
        mock_logger.info.assert_any_call(f"Processing TradeFeedback: {feedback_obj.feedback_id}") # Check placeholder log
        # Add asserts here if routing to specific methods like:
        # consumer.trading_feedback_collector.process_trade_feedback.assert_called_once()

    # --- Test Unknown Topic --- 
    with patch('analysis_engine.adaptive_layer.event_consumers.logger') as mock_logger:
        await consumer._process_message(topic='unknown_topic', key='k2', value={"data": "test"})
        mock_logger.warning.assert_called_with("No deserialization model found for topic 'unknown_topic'. Skipping message.")

    # --- Test Deserialization Error --- 
    with patch('analysis_engine.adaptive_layer.event_consumers.logger') as mock_logger:
        invalid_value = raw_message_value.copy()
        invalid_value['data']['pnl'] = 'not_a_float' # Introduce error
        await consumer._process_message(topic='trade_feedback', key='k3', value=invalid_value)
        mock_logger.error.assert_called_once() # Check that an error was logged

# Add more tests for error handling, different feedback types, commit logic (if manual), etc.

from kafka import TopicPartition # Import needed for test_consumer_run_loop_polls_messages
