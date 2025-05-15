"""
Tests for chat API endpoints (app.api.v1.endpoints.chat)
"""
import pytest
from httpx import AsyncClient
from fastapi import status
from unittest.mock import AsyncMock, patch

from app.schemas.chat_schemas import (
    ChatMessage,
    ChatMessageResponse,
    ChatHistoryResponse
)
from app.services.chat_service import ChatService

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio

# --- Test Data --- #

@pytest.fixture
def sample_chat_message():
    return {
        "content": "Hello, how can I help with forex trading?",
        "message_type": "text",
        "metadata": {"source": "user", "timestamp": "2024-01-01T12:00:00Z"}
    }

@pytest.fixture
def sample_chat_history():
    return [
        {
            "message_id": "msg1",
            "content": "What's the current EUR/USD trend?",
            "message_type": "text",
            "metadata": {"source": "user", "timestamp": "2024-01-01T11:55:00Z"},
            "created_at": "2024-01-01T11:55:00Z"
        },
        {
            "message_id": "msg2",
            "content": "The EUR/USD is showing an upward trend...",
            "message_type": "text",
            "metadata": {"source": "assistant", "timestamp": "2024-01-01T11:55:30Z"},
            "created_at": "2024-01-01T11:55:30Z"
        }
    ]

# --- Mock Service Layer --- #

@pytest.fixture
def mock_chat_service():
    with patch('app.api.v1.endpoints.chat.ChatService') as mock_service:
        service_instance = AsyncMock(spec=ChatService)
        mock_service.return_value = service_instance
        yield service_instance

# --- Test Routes --- #

async def test_send_message_success(
    client: AsyncClient,
    mock_chat_service,
    sample_chat_message,
    default_headers
):
    """Test successful message sending."""
    # Mock service response
    mock_response = {
        "message_id": "test-msg-123",
        **sample_chat_message,
        "created_at": "2024-01-01T12:00:00Z"
    }
    mock_chat_service.process_message.return_value = mock_response

    response = await client.post(
        "/api/v1/chat/messages",
        json=sample_chat_message,
        headers=default_headers
    )

    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["message_id"] == "test-msg-123"
    assert response.json()["content"] == sample_chat_message["content"]
    mock_chat_service.process_message.assert_called_once_with(
        user_id=default_headers["X-User-ID"],
        message_data=sample_chat_message
    )

async def test_send_message_validation_error(
    client: AsyncClient,
    mock_chat_service,
    default_headers
):
    """Test message sending with invalid data."""
    invalid_message = {
        "content": "",  # Empty content should fail validation
        "message_type": "invalid_type",
        "metadata": {"source": "user"}
    }

    response = await client.post(
        "/api/v1/chat/messages",
        json=invalid_message,
        headers=default_headers
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "validation error" in response.json()["detail"][0]["msg"].lower()
    mock_chat_service.process_message.assert_not_called()

async def test_send_message_missing_auth(
    client: AsyncClient,
    mock_chat_service,
    sample_chat_message
):
    """Test message sending without authentication headers."""
    response = await client.post(
        "/api/v1/chat/messages",
        json=sample_chat_message
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    mock_chat_service.process_message.assert_not_called()

async def test_get_chat_history_success(
    client: AsyncClient,
    mock_chat_service,
    sample_chat_history,
    default_headers
):
    """Test successful chat history retrieval."""
    mock_chat_service.get_chat_history.return_value = {
        "messages": sample_chat_history,
        "total_count": len(sample_chat_history),
        "has_more": False
    }

    response = await client.get(
        "/api/v1/chat/history",
        headers=default_headers,
        params={"limit": 10, "offset": 0}
    )

    assert response.status_code == status.HTTP_200_OK
    assert len(response.json()["messages"]) == 2
    assert response.json()["total_count"] == 2
    assert response.json()["has_more"] is False
    mock_chat_service.get_chat_history.assert_called_once_with(
        user_id=default_headers["X-User-ID"],
        limit=10,
        offset=0
    )

async def test_get_chat_history_invalid_params(
    client: AsyncClient,
    mock_chat_service,
    default_headers
):
    """Test chat history retrieval with invalid parameters."""
    response = await client.get(
        "/api/v1/chat/history",
        headers=default_headers,
        params={"limit": -1, "offset": -5}  # Invalid values
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "validation error" in response.json()["detail"][0]["msg"].lower()
    mock_chat_service.get_chat_history.assert_not_called()

async def test_get_chat_history_missing_auth(
    client: AsyncClient,
    mock_chat_service
):
    """Test chat history retrieval without authentication headers."""
    response = await client.get(
        "/api/v1/chat/history",
        params={"limit": 10, "offset": 0}
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    mock_chat_service.get_chat_history.assert_not_called()

async def test_get_chat_history_service_error(
    client: AsyncClient,
    mock_chat_service,
    default_headers
):
    """Test chat history retrieval when service raises an error."""
    mock_chat_service.get_chat_history.side_effect = Exception("Database error")

    response = await client.get(
        "/api/v1/chat/history",
        headers=default_headers,
        params={"limit": 10, "offset": 0}
    )

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "error" in response.json()