"""
Tests for dependency injection functions (app.dependencies)
"""
import pytest
from fastapi import Request, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import (
    get_correlation_id,
    get_user_id,
    get_chat_repository,
    get_event_bus_instance,
    validate_api_key,
    get_settings_instance
)
from app.repositories.chat_repository import ChatRepository
from app.events.event_bus import EventBus, InMemoryEventBus
from app.config.settings import Settings

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio

# --- Mock Request Object --- #

@pytest.fixture
def mock_request_factory():
    """Factory to create mock FastAPI Request objects."""
    class MockState:
        def __init__(self):
            self.correlation_id = None

    class MockRequest:
        def __init__(self, headers: dict = None, state: MockState = None):
            self.headers = headers if headers is not None else {}
            self.state = state if state is not None else MockState()

    return MockRequest

# --- Tests for get_correlation_id --- #

def test_get_correlation_id_present(mock_request_factory):
    """Test get_correlation_id when correlation_id is in request.state."""
    mock_state = mock_request_factory().state
    mock_state.correlation_id = "test-corr-id"
    request = mock_request_factory(state=mock_state)
    assert get_correlation_id(request) == "test-corr-id"

def test_get_correlation_id_not_present(mock_request_factory):
    """Test get_correlation_id when correlation_id is not in request.state."""
    request = mock_request_factory()
    assert get_correlation_id(request) is None

# --- Tests for get_user_id --- #

def test_get_user_id_present(mock_request_factory):
    """Test get_user_id when X-User-ID header is present."""
    request = mock_request_factory(headers={"X-User-ID": "test-user"})
    assert get_user_id(request) == "test-user"

def test_get_user_id_missing(mock_request_factory):
    """Test get_user_id when X-User-ID header is missing."""
    request = mock_request_factory(headers={})
    with pytest.raises(HTTPException) as exc_info:
        get_user_id(request)
    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert "User ID header is missing" in exc_info.value.detail

# --- Tests for get_chat_repository --- #

async def test_get_chat_repository(db_session: AsyncSession):
    """Test get_chat_repository provides a ChatRepository instance."""
    repo_generator = get_chat_repository(db_session)
    try:
        repo = await anext(repo_generator)
        assert isinstance(repo, ChatRepository)
        assert repo.db == db_session
    finally:
        # Ensure generator is properly closed if anext raises StopAsyncIteration or other error
        try:
            await anext(repo_generator) # Should raise StopAsyncIteration if properly exhausted
        except StopAsyncIteration:
            pass
        # If the generator has a .aclose() method, call it. For simple yields, this might not be needed.
        if hasattr(repo_generator, 'aclose'):
            await repo_generator.aclose()

# --- Tests for get_event_bus_instance --- #

async def test_get_event_bus_instance(mock_event_bus: InMemoryEventBus, test_settings: Settings):
    """Test get_event_bus_instance provides an EventBus instance (InMemoryEventBus for tests)."""
    # Ensure settings reflect in-memory bus for this test context
    original_event_bus_type = test_settings.EVENT_BUS_TYPE
    test_settings.EVENT_BUS_TYPE = "in-memory" 

    bus_generator = get_event_bus_instance()
    try:
        bus = await anext(bus_generator)
        assert isinstance(bus, EventBus)
        assert isinstance(bus, InMemoryEventBus) # Based on conftest.py override
        # Check if start was called (mock_event_bus in conftest already calls start)
    finally:
        try:
            await anext(bus_generator)
        except StopAsyncIteration:
            pass
        if hasattr(bus_generator, 'aclose'):
            await bus_generator.aclose()
        test_settings.EVENT_BUS_TYPE = original_event_bus_type # Restore setting

# --- Tests for validate_api_key --- #

def test_validate_api_key_valid(mock_request_factory, test_settings: Settings):
    """Test validate_api_key with a valid API key."""
    headers = {test_settings.API_KEY_NAME: test_settings.SECRET_KEY}
    request = mock_request_factory(headers=headers)
    try:
        validate_api_key(request) # Should not raise
    except HTTPException:
        pytest.fail("validate_api_key raised HTTPException unexpectedly for a valid key")

def test_validate_api_key_missing(mock_request_factory, test_settings: Settings):
    """Test validate_api_key when API key is missing."""
    request = mock_request_factory(headers={})
    with pytest.raises(HTTPException) as exc_info:
        validate_api_key(request)
    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert "API key is missing" in exc_info.value.detail

def test_validate_api_key_invalid(mock_request_factory, test_settings: Settings):
    """Test validate_api_key with an invalid API key."""
    headers = {test_settings.API_KEY_NAME: "invalid-key"}
    request = mock_request_factory(headers=headers)
    with pytest.raises(HTTPException) as exc_info:
        validate_api_key(request)
    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Invalid API key" in exc_info.value.detail

# --- Tests for get_settings_instance --- #

def test_get_settings_instance(test_settings: Settings):
    """Test get_settings_instance returns the correct Settings object (test_settings in this case)."""
    # Note: get_settings() in app.dependencies uses lru_cache.
    # The conftest.py overrides get_settings for the app, but here we test the direct call.
    # For a fully isolated test of get_settings_instance, you might need to clear lru_cache
    # or ensure the global get_settings() is patched if it's different from test_settings.
    # However, in the test environment, get_settings() should ideally return test_settings.
    
    # To ensure we are testing the behavior within our test context where get_settings() is overridden by conftest
    # we rely on the fact that get_settings_instance() calls the (potentially overridden) get_settings().
    settings_instance = get_settings_instance()
    assert isinstance(settings_instance, Settings)
    assert settings_instance.DATABASE_URL == test_settings.DATABASE_URL # Check a test-specific value
    assert settings_instance.SECRET_KEY == test_settings.SECRET_KEY