"""
Test Configuration

This module provides common test fixtures and configuration for the Analysis Engine Service tests.
"""
import os
import sys
import pytest
import asyncio
from typing import AsyncGenerator, Generator
from fastapi import FastAPI
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
    'mocks')))
from mocks import common_lib
sys.modules['common_lib'] = common_lib
sys.modules['common_lib.config'] = common_lib.config
from analysis_engine.config.settings import AnalysisEngineSettings as Settings, get_settings
from analysis_engine.core.service_container import ServiceContainer
from analysis_engine.main import create_app
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock
from analysis_engine.core.memory_monitor import MemoryMonitor
from analysis_engine.core.database import DatabaseManager
from analysis_engine.core.cache import CacheManager


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@pytest.fixture(scope='function')
def event_loop() ->Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for each test function.

    This matches the asyncio_default_fixture_loop_scope=function setting in pytest.ini.
    """
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope='session')
@with_exception_handling
def test_settings() ->Settings:
    """Load test settings from .env.test file."""
    try:
        settings = Settings()
        if not settings.JWT_SECRET or len(settings.JWT_SECRET) < 16:
            raise ValueError(
                'JWT_SECRET not loaded correctly or too short from .env.test')
        return settings
    except Exception as e:
        print(f'Error loading test settings: {e}')
        raise RuntimeError('Failed to load test settings from .env.test'
            ) from e


@pytest.fixture(scope='session')
def app(test_settings: Settings) ->FastAPI:
    """Create a test FastAPI application."""
    app = create_app()
    return app


@pytest.fixture(scope='session')
async def service_container() ->AsyncGenerator[ServiceContainer, None]:
    """Create a test service container."""
    container = ServiceContainer()
    yield container
    await container.cleanup()


@pytest.fixture(scope='session')
def client(app: FastAPI) ->TestClient:
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture(scope='function')
def mock_db_pool() ->AsyncMock:
    """Create a mock database connection pool."""
    pool = AsyncMock()
    pool.acquire = AsyncMock()
    pool.release = AsyncMock()
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=[])
    mock_conn.execute = AsyncMock()
    mock_conn.commit = AsyncMock()
    mock_conn.rollback = AsyncMock()
    pool.acquire.return_value = mock_conn
    return pool


@pytest.fixture(scope='function')
def mock_redis() ->AsyncMock:
    """Create a mock Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock()
    redis.delete = AsyncMock()
    return redis


@pytest.fixture(scope='function')
def mock_db_manager(mock_db_pool: AsyncMock) ->DatabaseManager:
    """Create a mock DatabaseManager instance."""
    return DatabaseManager(mock_db_pool)


@pytest.fixture(scope='function')
def mock_cache_manager(mock_redis: AsyncMock) ->CacheManager:
    """Create a mock CacheManager instance."""
    return CacheManager(mock_redis)


@pytest.fixture(scope='function')
def mock_analysis_service() ->MagicMock:
    """Create a mock analysis service."""
    service = MagicMock()
    return service


@pytest.fixture(scope='function')
async def clean_service_container(service_container: ServiceContainer
    ) ->AsyncGenerator[ServiceContainer, None]:
    """Create a clean service container for each test."""
    yield service_container
    await service_container.cleanup()


@pytest.fixture(scope='session')
async def memory_monitor():
    """Create a memory monitor instance for testing."""
    monitor = MemoryMonitor()
    yield monitor
    await monitor.stop_monitoring()


@pytest.fixture(autouse=True)
async def setup_teardown():
    """Setup and teardown for each test."""
    yield
