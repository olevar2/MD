"""
Pytest configuration and fixtures for chat-service tests
"""
import pytest
import asyncio
from typing import AsyncGenerator, Generator
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.main import app  # Import your FastAPI app
from app.database import Base, get_db
from app.config.settings import Settings, get_settings
from app.dependencies import get_event_bus_instance # To mock event bus
from app.events.event_bus import EventBus, InMemoryEventBus # For mock event bus

# --- Settings for Testing --- #

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for each test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Override settings for testing (e.g., use a test database)."""
    return Settings(
        DATABASE_URL="sqlite+aiosqlite:///./test_chat_service.db", # In-memory SQLite for tests
        API_DEBUG=True,
        EVENT_BUS_TYPE="in-memory", # Use in-memory event bus for tests
        SECRET_KEY="testsecretkey"
    )

# --- Database Fixtures --- #

@pytest.fixture(scope="session")
async def test_engine(test_settings: Settings):
    """Create an async engine for the test database."""
    engine = create_async_engine(test_settings.DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all) # Create tables
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all) # Drop tables after tests
    await engine.dispose()

@pytest.fixture(scope="function")
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Provide a database session for each test function, with rollback."""
    AsyncTestingSessionLocal = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with AsyncTestingSessionLocal() as session:
        await session.begin_nested() # Use savepoints for rollback
        yield session
        await session.rollback() # Rollback changes after each test

# --- Application Fixtures (FastAPI Test Client) --- #

@pytest.fixture(scope="function")
async def mock_event_bus() -> InMemoryEventBus:
    """Provides a mock in-memory event bus for testing."""
    return InMemoryEventBus()

@pytest.fixture(scope="function")
async def client(
    db_session: AsyncSession, 
    test_settings: Settings, 
    mock_event_bus: InMemoryEventBus
) -> AsyncGenerator[AsyncClient, None]:
    """Provide an HTTPX AsyncClient for making requests to the FastAPI app."""
    
    # Override dependencies for testing
    def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    def override_get_settings() -> Settings:
        return test_settings
    
    async def override_get_event_bus() -> EventBus:
        await mock_event_bus.start() # Ensure it's started
        return mock_event_bus

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_settings] = override_get_settings
    app.dependency_overrides[get_event_bus_instance] = override_get_event_bus
    
    async with AsyncClient(app=app, base_url="http://testserver") as ac:
        yield ac
    
    # Clear overrides after tests
    app.dependency_overrides.clear()

# --- Helper Fixtures --- #

@pytest.fixture
def default_user_id() -> str:
    return "test-user-123"

@pytest.fixture
def default_api_key(test_settings: Settings) -> str:
    return test_settings.SECRET_KEY # Use the test secret key as a valid API key

@pytest.fixture
def default_headers(default_user_id: str, default_api_key: str) -> dict:
    return {
        "X-User-ID": default_user_id,
        "X-API-Key": default_api_key,
        "X-Correlation-ID": "test-correlation-id"
    }