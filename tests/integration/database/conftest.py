"""
Fixtures for database integration tests.
"""
import os
import pytest
import pytest_asyncio
import asyncio
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager
import unittest.mock as mock
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from common_lib.database import (
    get_connection_pool,
    get_sync_db_session,
    get_async_db_session,
    get_asyncpg_connection,
)

# Import and set USE_MOCKS flag
from common_lib.database.config import USE_MOCKS
import common_lib.database.config

# Set USE_MOCKS to True
common_lib.database.config.USE_MOCKS = True

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test database configuration - using mock values
TEST_DB_HOST = os.environ.get("TEST_DB_HOST", "localhost")
TEST_DB_PORT = int(os.environ.get("TEST_DB_PORT", "5432"))
TEST_DB_USER = os.environ.get("TEST_DB_USER", "postgres")
TEST_DB_PASSWORD = os.environ.get("TEST_DB_PASSWORD", "postgres")
TEST_DB_NAME = os.environ.get("TEST_DB_NAME", "test_forex_platform")
TEST_DB_URL = f"postgresql+asyncpg://{TEST_DB_USER}:{TEST_DB_PASSWORD}@{TEST_DB_HOST}:{TEST_DB_PORT}/{TEST_DB_NAME}"

# Test Redis configuration - using mock values
TEST_REDIS_HOST = os.environ.get("TEST_REDIS_HOST", "localhost")
TEST_REDIS_PORT = int(os.environ.get("TEST_REDIS_PORT", "6379"))
TEST_REDIS_URL = f"redis://{TEST_REDIS_HOST}:{TEST_REDIS_PORT}/0"


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def test_db_pool():
    """Create a test database connection pool."""
    # Use our improved mock implementation
    from common_lib.database.testing import get_mock_connection_pool
    
    # Create a mock pool for testing
    pool = get_mock_connection_pool("test_service")
    
    # Initialize the pool
    await pool.initialize_async()
    await pool.initialize_asyncpg()
    
    yield pool
    
    # Close the pool
    await pool.close_async()


@pytest_asyncio.fixture(scope="session")
async def test_redis_pool():
    """Create a test Redis connection pool."""
    # Create a mock Redis client
    redis = mock.MagicMock()
    
    # Mock Redis methods
    redis.get = mock.AsyncMock(return_value="test_value")
    redis.set = mock.AsyncMock(return_value=True)
    redis.delete = mock.AsyncMock(return_value=1)
    redis.exists = mock.AsyncMock(return_value=1)
    redis.expire = mock.AsyncMock(return_value=True)
    redis.close = mock.AsyncMock()
    
    yield redis


@pytest_asyncio.fixture
async def test_db_session(test_db_pool):
    """Create a test database session."""
    # Use our improved mock implementation
    from common_lib.database.testing import get_mock_async_db_session
    
    async with get_mock_async_db_session("test_service") as session:
        yield session


@pytest.fixture
def service_names():
    """Return a list of service names for testing."""
    return [
        "analysis_engine_service",
        "data_pipeline_service",
        "feature_store_service",
        "market_analysis_service",
    ]


@pytest_asyncio.fixture
async def connection_pools(service_names):
    """Create connection pools for each service."""
    # Use our improved mock implementation
    from common_lib.database.testing import get_mock_connection_pool
    
    pools = {}
    
    # Create a connection pool for each service
    for service_name in service_names:
        pool = get_mock_connection_pool(service_name)
        await pool.initialize_async()
        await pool.initialize_asyncpg()
        pools[service_name] = pool
    
    yield pools
    
    # Close the pools
    for pool in pools.values():
        await pool.close_async()


@pytest_asyncio.fixture
async def clean_test_tables(test_db_pool):
    """Clean test tables before and after each test."""
    # No need to do anything with mocks
    yield


@pytest.fixture
def test_data():
    """Generate test data for integration tests."""
    # Generate test data for regular table
    regular_data = [
        {"name": f"test_{i}", "value": i * 10}
        for i in range(1, 101)
    ]
    
    # Generate test data for time series table
    from datetime import datetime, timedelta
    
    start_time = datetime.utcnow() - timedelta(days=30)
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    
    time_series_data = []
    for symbol in symbols:
        for i in range(1000):
            time = start_time + timedelta(hours=i)
            value = 1.0 + (i % 100) / 1000.0
            time_series_data.append({
                "time": time,
                "symbol": symbol,
                "value": value,
            })
    
    return {
        "regular": regular_data,
        "time_series": time_series_data,
    }