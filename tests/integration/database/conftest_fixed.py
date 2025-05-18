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
    USE_MOCKS,
)

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
    if USE_MOCKS:
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
    else:
        # Create a real database connection pool
        engine = create_async_engine(
            TEST_DB_URL,
            echo=False,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
        )
        
        async_session_factory = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        
        # Create a simple pool wrapper
        class TestPool:
            def __init__(self, engine, session_factory):
                self.engine = engine
                self.session_factory = session_factory
            
            @asynccontextmanager
            async def get_async_session(self):
                async with self.session_factory() as session:
                    yield session
        
        pool = TestPool(engine, async_session_factory)
        
        yield pool
        
        # Close the engine
        await engine.dispose()


@pytest_asyncio.fixture(scope="session")
async def test_redis_pool():
    """Create a test Redis connection pool."""
    if USE_MOCKS:
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
    else:
        # Create a real Redis connection
        import aioredis
        
        redis = await aioredis.from_url(
            TEST_REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
        
        yield redis
        
        # Close the Redis connection
        await redis.close()


@pytest_asyncio.fixture
async def test_db_session(test_db_pool):
    """Create a test database session."""
    if USE_MOCKS:
        # Use our improved mock implementation
        from common_lib.database.testing import get_mock_async_db_session
        
        async with get_mock_async_db_session("test_service") as session:
            yield session
    else:
        # Use the real pool to get a session
        async with test_db_pool.get_async_session() as session:
            # Start a transaction that will be rolled back
            await session.begin()
            yield session
            # Roll back the transaction
            await session.rollback()


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
    if USE_MOCKS:
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
    else:
        # Create real connection pools
        from common_lib.database.connection_pool import DatabaseConnectionPool
        
        pools = {}
        
        # Create a connection pool for each service
        for service_name in service_names:
            config = {
                "database_url": TEST_DB_URL,
                "async_database_url": TEST_DB_URL,
                "pool_size": 5,
                "max_overflow": 10,
                "pool_timeout": 30,
                "pool_recycle": 1800,
                "echo": False,
                "prepared_statement_cache_size": 100,
            }
            
            pool = DatabaseConnectionPool(service_name, **config)
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
    if USE_MOCKS:
        # No need to do anything with mocks
        yield
    else:
        # Clean tables before the test
        async with test_db_pool.get_async_session() as session:
            # Drop and recreate test schema
            await session.execute("DROP SCHEMA IF EXISTS test_schema CASCADE")
            await session.execute("CREATE SCHEMA test_schema")
            
            # Create test tables
            await session.execute("""
                CREATE TABLE test_schema.test_table (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    value INTEGER NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            
            await session.execute("""
                CREATE TABLE test_schema.time_series_table (
                    time TIMESTAMP WITH TIME ZONE NOT NULL,
                    symbol TEXT NOT NULL,
                    value DOUBLE PRECISION NOT NULL,
                    PRIMARY KEY (time, symbol)
                )
            """)
            
            await session.commit()
        
        yield
        
        # Clean tables after the test
        async with test_db_pool.get_async_session() as session:
            await session.execute("DROP SCHEMA IF EXISTS test_schema CASCADE")
            await session.commit()


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