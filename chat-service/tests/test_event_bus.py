"""
Tests for event bus implementations (app.events.event_bus)
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.events.event_bus import (
    EventBus,
    KafkaEventBus,
    InMemoryEventBus,
    get_event_bus
)
from app.config.settings import Settings

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_settings_kafka(monkeypatch, test_settings: Settings) -> Settings:
    """Override settings to use Kafka event bus for specific tests."""
    monkeypatch.setattr(test_settings, 'EVENT_BUS_TYPE', 'kafka')
    monkeypatch.setattr(test_settings, 'KAFKA_BOOTSTRAP_SERVERS', 'fake_kafka:9092')
    return test_settings

@pytest.fixture
def mock_settings_in_memory(monkeypatch, test_settings: Settings) -> Settings:
    """Override settings to use InMemory event bus for specific tests."""
    monkeypatch.setattr(test_settings, 'EVENT_BUS_TYPE', 'in-memory')
    return test_settings

# --- InMemoryEventBus Tests --- #

async def test_in_memory_event_bus_publish_subscribe():
    """Test basic publish and subscribe functionality of InMemoryEventBus."""
    bus = InMemoryEventBus()
    test_topic = "test_topic_in_memory"
    received_event = None
    event_processed = asyncio.Event()

    async def handler(event_data):
        nonlocal received_event
        received_event = event_data
        event_processed.set()

    await bus.subscribe(test_topic, handler)
    
    test_event_data = {"message": "hello world"}
    await bus.publish(test_topic, test_event_data)

    await asyncio.wait_for(event_processed.wait(), timeout=1.0)
    
    assert received_event is not None
    assert received_event == test_event_data

async def test_in_memory_event_bus_multiple_handlers():
    """Test InMemoryEventBus with multiple handlers for the same topic."""
    bus = InMemoryEventBus()
    test_topic = "multi_handler_topic"
    
    received_counts = {"handler1": 0, "handler2": 0}
    event_data_store = {"handler1": None, "handler2": None}
    events_processed = {"handler1": asyncio.Event(), "handler2": asyncio.Event()}

    async def handler1(event_data):
        received_counts["handler1"] += 1
        event_data_store["handler1"] = event_data
        events_processed["handler1"].set()

    async def handler2(event_data):
        received_counts["handler2"] += 1
        event_data_store["handler2"] = event_data
        events_processed["handler2"].set()

    await bus.subscribe(test_topic, handler1)
    await bus.subscribe(test_topic, handler2)

    test_event = {"data": "test_payload"}
    await bus.publish(test_topic, test_event)

    await asyncio.wait_for(events_processed["handler1"].wait(), timeout=1.0)
    await asyncio.wait_for(events_processed["handler2"].wait(), timeout=1.0)

    assert received_counts["handler1"] == 1
    assert received_counts["handler2"] == 1
    assert event_data_store["handler1"] == test_event
    assert event_data_store["handler2"] == test_event

async def test_in_memory_event_bus_no_handler():
    """Test InMemoryEventBus publish to a topic with no handlers (should not error)."""
    bus = InMemoryEventBus()
    try:
        await bus.publish("unhandled_topic", {"data": "test"})
    except Exception as e:
        pytest.fail(f"Publishing to unhandled topic raised an exception: {e}")

async def test_in_memory_start_stop_noop():
    """Test InMemoryEventBus start and stop methods (should be no-ops)."""
    bus = InMemoryEventBus()
    await bus.start()
    await bus.stop()
    # No assertions needed, just checking they don't raise errors

# --- KafkaEventBus Tests (Mocked) --- #

@pytest.fixture
def mock_aiokafka_producer(monkeypatch):
    mock_producer_instance = AsyncMock()
    mock_producer_class = MagicMock(return_value=mock_producer_instance)
    monkeypatch.setattr("app.events.event_bus.AIOKafkaProducer", mock_producer_class)
    return mock_producer_instance, mock_producer_class

@pytest.fixture
def mock_aiokafka_consumer(monkeypatch):
    mock_consumer_instance = AsyncMock()
    # Make the consumer instance an async iterable for `async for msg in consumer:`
    async def mock_consumer_iter():
        # Simulate some messages or an empty stream
        # For simplicity, let's make it empty for now, or yield mock messages if needed for specific tests
        if False: # Condition to yield messages, set to True and add messages for specific tests
            yield MagicMock(value={"data": "test_message"})
        return
        yield # Necessary for async generator syntax
    
    mock_consumer_instance.__aiter__.return_value = mock_consumer_iter()
    mock_consumer_instance.subscription.return_value = ["test_topic_kafka"] # Mock subscription
    
    mock_consumer_class = MagicMock(return_value=mock_consumer_instance)
    monkeypatch.setattr("app.events.event_bus.AIOKafkaConsumer", mock_consumer_class)
    return mock_consumer_instance, mock_consumer_class

async def test_kafka_event_bus_publish(mock_settings_kafka, mock_aiokafka_producer):
    """Test KafkaEventBus publish method (mocked)."""
    producer_instance, _ = mock_aiokafka_producer
    bus = KafkaEventBus() # Uses mocked AIOKafkaProducer due to fixture
    await bus.start() # Starts the mocked producer

    test_topic = "test_topic_kafka"
    test_event = {"message": "kafka test"}
    test_key = "test_key"

    await bus.publish(test_topic, test_event, key=test_key)

    producer_instance.send_and_wait.assert_called_once_with(test_topic, test_event, key=test_key)
    await bus.stop()

async def test_kafka_event_bus_subscribe_and_process(mock_settings_kafka, mock_aiokafka_consumer, mock_aiokafka_producer):
    """Test KafkaEventBus subscribe and message processing (mocked)."""
    consumer_instance, consumer_class = mock_aiokafka_consumer
    producer_instance, _ = mock_aiokafka_producer

    bus = KafkaEventBus()
    test_topic = "test_topic_kafka"
    received_event_data = None
    event_processed = asyncio.Event()

    async def handler(event_data):
        nonlocal received_event_data
        received_event_data = event_data
        event_processed.set()

    await bus.subscribe(test_topic, handler)
    assert len(bus.handlers[test_topic]) == 1
    assert len(bus.consumers) == 1
    consumer_class.assert_called_once_with(
        test_topic,
        bootstrap_servers=mock_settings_kafka.KAFKA_BOOTSTRAP_SERVERS,
        group_id=mock_settings_kafka.KAFKA_CONSUMER_GROUP,
        value_deserializer=unittest.mock.ANY # or a more specific check if needed
    )

    # Mock the consumer to yield a message
    mock_message = MagicMock()
    mock_message.value = {"data": "kafka message"}
    async def mock_consumer_iter_with_message():
        yield mock_message
    consumer_instance.__aiter__.return_value = mock_consumer_iter_with_message()
    consumer_instance.subscription.return_value = [test_topic]

    # Start the bus (which starts producer and consumers)
    # The consumer loop runs in bus.start(), so we need to run it in a task
    start_task = asyncio.create_task(bus.start())
    
    # Wait for the handler to process the message
    try:
        await asyncio.wait_for(event_processed.wait(), timeout=2.0)
    except asyncio.TimeoutError:
        pytest.fail("Handler was not called in time")
    finally:
        await bus.stop() # Stop producer and consumers
        start_task.cancel() # Cancel the bus.start() task
        try:
            await start_task
        except asyncio.CancelledError:
            pass # Expected

    assert received_event_data == {"data": "kafka message"}
    producer_instance.start.assert_called_once()
    consumer_instance.start.assert_called_once()
    producer_instance.stop.assert_called_once()
    consumer_instance.stop.assert_called_once()

async def test_kafka_event_bus_publish_error_logging(mock_settings_kafka, mock_aiokafka_producer, caplog):
    """Test KafkaEventBus logs error on publish failure."""
    producer_instance, _ = mock_aiokafka_producer
    producer_instance.send_and_wait.side_effect = Exception("Kafka publish error")
    
    bus = KafkaEventBus()
    await bus.start()
    with pytest.raises(Exception, match="Kafka publish error"):
        await bus.publish("error_topic", {"data": "fail"})
    
    assert "Failed to publish event to topic error_topic" in caplog.text
    await bus.stop()

# --- get_event_bus Tests --- #

def test_get_event_bus_in_memory(mock_settings_in_memory):
    """Test get_event_bus returns InMemoryEventBus when configured."""
    # Need to clear cache if get_settings is cached and settings change
    from app.config.settings import get_settings
    get_settings.cache_clear() 
    # Patch get_settings to return our mock_settings_in_memory for this call context
    with patch('app.events.event_bus.get_settings', return_value=mock_settings_in_memory):
        bus = get_event_bus()
        assert isinstance(bus, InMemoryEventBus)

def test_get_event_bus_kafka(mock_settings_kafka, mock_aiokafka_producer, mock_aiokafka_consumer):
    """Test get_event_bus returns KafkaEventBus when configured (mocked)."""
    from app.config.settings import get_settings
    get_settings.cache_clear()
    with patch('app.events.event_bus.get_settings', return_value=mock_settings_kafka):
        bus = get_event_bus()
        assert isinstance(bus, KafkaEventBus)
        # Check that mocks were used for producer/consumer creation if bus was instantiated
        # This depends on whether get_event_bus() instantiates or just returns a class/factory
        # KafkaEventBus() is instantiated within get_event_bus if type is kafka.
        assert bus.producer == mock_aiokafka_producer[0]
        # Consumers are created on subscribe, so bus.consumers will be empty initially
        assert len(bus.consumers) == 0