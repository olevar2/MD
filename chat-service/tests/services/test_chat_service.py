"""
Tests for chat service layer (app.services.chat_service)
"""
import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime, timezone

from app.services.chat_service import ChatService
from app.events.event_bus import EventBus
from app.exceptions import ValidationError, ResourceNotFoundError, DatabaseError

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio

# --- Test Data --- #

@pytest.fixture
def sample_message_data():
    return {
        "content": "What's the current market trend?",
        "message_type": "text",
        "metadata": {
            "source": "user",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    }

@pytest.fixture
def sample_processed_message(sample_message_data):
    return {
        "message_id": "test-msg-123",
        **sample_message_data,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

# --- Mock Dependencies --- #

@pytest.fixture
def mock_chat_repository():
    return AsyncMock()

@pytest.fixture
def mock_event_bus():
    return AsyncMock(spec=EventBus)

@pytest.fixture
def chat_service(
    mock_chat_repository,
    mock_event_bus
):
    with patch('app.services.chat_service.ChatRepository', return_value=mock_chat_repository):
        service = ChatService(event_bus=mock_event_bus)
        return service

# --- Test Message Processing --- #

async def test_process_message_success(
    chat_service,
    mock_chat_repository,
    mock_event_bus,
    sample_message_data,
    sample_processed_message
):
    """Test successful message processing."""
    mock_chat_repository.create_message.return_value = sample_processed_message
    
    result = await chat_service.process_message(
        user_id="test-user-123",
        message_data=sample_message_data
    )
    
    assert result == sample_processed_message
    mock_chat_repository.create_message.assert_called_once_with(
        user_id="test-user-123",
        message_data=sample_message_data
    )
    mock_event_bus.publish.assert_called_once_with(
        "chat.message.created",
        {
            "user_id": "test-user-123",
            "message": sample_processed_message
        }
    )

async def test_process_message_validation_error(
    chat_service,
    mock_chat_repository,
    mock_event_bus
):
    """Test message processing with invalid data."""
    invalid_message = {
        "content": "",  # Empty content
        "message_type": "invalid_type",
        "metadata": {}
    }
    
    with pytest.raises(ValidationError) as exc_info:
        await chat_service.process_message(
            user_id="test-user-123",
            message_data=invalid_message
        )
    
    assert "Invalid message data" in str(exc_info.value)
    mock_chat_repository.create_message.assert_not_called()
    mock_event_bus.publish.assert_not_called()

async def test_process_message_database_error(
    chat_service,
    mock_chat_repository,
    mock_event_bus,
    sample_message_data
):
    """Test message processing when database operation fails."""
    mock_chat_repository.create_message.side_effect = Exception("Database error")
    
    with pytest.raises(DatabaseError) as exc_info:
        await chat_service.process_message(
            user_id="test-user-123",
            message_data=sample_message_data
        )
    
    assert "Failed to create message" in str(exc_info.value)
    mock_event_bus.publish.assert_not_called()

# --- Test Chat History Retrieval --- #

async def test_get_chat_history_success(
    chat_service,
    mock_chat_repository,
    sample_processed_message
):
    """Test successful chat history retrieval."""
    mock_chat_repository.get_messages.return_value = {
        "messages": [sample_processed_message],
        "total_count": 1,
        "has_more": False
    }
    
    result = await chat_service.get_chat_history(
        user_id="test-user-123",
        limit=10,
        offset=0
    )
    
    assert result["messages"] == [sample_processed_message]
    assert result["total_count"] == 1
    assert result["has_more"] is False
    mock_chat_repository.get_messages.assert_called_once_with(
        user_id="test-user-123",
        limit=10,
        offset=0
    )

async def test_get_chat_history_empty(
    chat_service,
    mock_chat_repository
):
    """Test chat history retrieval when no messages exist."""
    mock_chat_repository.get_messages.return_value = {
        "messages": [],
        "total_count": 0,
        "has_more": False
    }
    
    result = await chat_service.get_chat_history(
        user_id="test-user-123",
        limit=10,
        offset=0
    )
    
    assert result["messages"] == []
    assert result["total_count"] == 0
    assert result["has_more"] is False

async def test_get_chat_history_invalid_params(
    chat_service,
    mock_chat_repository
):
    """Test chat history retrieval with invalid parameters."""
    with pytest.raises(ValidationError) as exc_info:
        await chat_service.get_chat_history(
            user_id="test-user-123",
            limit=-1,  # Invalid limit
            offset=-5  # Invalid offset
        )
    
    assert "Invalid pagination parameters" in str(exc_info.value)
    mock_chat_repository.get_messages.assert_not_called()

async def test_get_chat_history_database_error(
    chat_service,
    mock_chat_repository
):
    """Test chat history retrieval when database operation fails."""
    mock_chat_repository.get_messages.side_effect = Exception("Database error")
    
    with pytest.raises(DatabaseError) as exc_info:
        await chat_service.get_chat_history(
            user_id="test-user-123",
            limit=10,
            offset=0
        )
    
    assert "Failed to retrieve chat history" in str(exc_info.value)

# --- Test Message Validation --- #

async def test_validate_message_success(
    chat_service,
    sample_message_data
):
    """Test successful message validation."""
    try:
        chat_service._validate_message(sample_message_data)
    except ValidationError:
        pytest.fail("Valid message data raised ValidationError")

async def test_validate_message_missing_required_fields(
    chat_service
):
    """Test message validation with missing required fields."""
    invalid_messages = [
        {},  # Empty message
        {"content": "Hello"},  # Missing message_type and metadata
        {"message_type": "text"},  # Missing content and metadata
        {"content": "Hello", "message_type": "text"}  # Missing metadata
    ]
    
    for invalid_msg in invalid_messages:
        with pytest.raises(ValidationError) as exc_info:
            chat_service._validate_message(invalid_msg)
        assert "Missing required fields" in str(exc_info.value)

async def test_validate_message_invalid_message_type(
    chat_service,
    sample_message_data
):
    """Test message validation with invalid message type."""
    invalid_message = sample_message_data.copy()
    invalid_message["message_type"] = "invalid_type"
    
    with pytest.raises(ValidationError) as exc_info:
        chat_service._validate_message(invalid_message)
    assert "Invalid message type" in str(exc_info.value)

async def test_validate_message_invalid_metadata(
    chat_service,
    sample_message_data
):
    """Test message validation with invalid metadata."""
    invalid_messages = [
        {**sample_message_data, "metadata": None},  # None metadata
        {**sample_message_data, "metadata": "not_a_dict"},  # Non-dict metadata
        {**sample_message_data, "metadata": {"source": None}},  # Invalid source
        {**sample_message_data, "metadata": {"source": "invalid_source"}}  # Invalid source value
    ]
    
    for invalid_msg in invalid_messages:
        with pytest.raises(ValidationError) as exc_info:
            chat_service._validate_message(invalid_msg)
        assert "Invalid metadata" in str(exc_info.value)