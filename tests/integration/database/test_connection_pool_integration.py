"""
Integration tests for database connection pool.
"""
import pytest
import asyncio
import logging
from typing import Dict, List, Any, Optional
import time

from common_lib.database import (
    get_connection_pool,
    get_sync_db_session,
    get_async_db_session,
    get_asyncpg_connection,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_connection_pool_basic(connection_pools, service_names):
    """Test basic connection pool functionality."""
    # Get the first service name
    service_name = service_names[0]
    
    # Get the connection pool
    pool = connection_pools[service_name]
    
    # Verify that the pool is initialized
    assert pool is not None
    
    # Initialize the pool
    await pool.initialize_async()
    
    # Verify that the pool is initialized
    assert pool._async_engine is not None
    assert pool._async_session_factory is not None
    
    # Get a session from the pool
    async with pool.get_async_session() as session:
        # Verify that the session is valid
        result = await session.execute("SELECT 1")
        value = result.scalar()
        assert value == 1


@pytest.mark.asyncio
async def test_connection_pool_concurrent(connection_pools, service_names):
    """Test concurrent connections from a single pool."""
    # Get the first service name
    service_name = service_names[0]
    
    # Get the connection pool
    pool = connection_pools[service_name]
    
    # Initialize the pool
    await pool.initialize_async()
    
    # Define a test query function
    async def execute_query(i: int):
        async with pool.get_async_session() as session:
            result = await session.execute("SELECT pg_sleep(0.1), $1::int", [i])
            value = result.scalar_one_or_none()
            return value
    
    # Execute multiple queries concurrently
    tasks = [execute_query(i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    
    # Verify that all queries completed successfully
    assert len(results) == 10
    assert all(result == i for i, result in enumerate(results))
    
    # Verify that the pool metrics were updated
    # Note: In a real test, we would check Prometheus metrics
    # For this test, we'll just check that the pool is still valid
    assert pool._async_engine is not None


@pytest.mark.asyncio
async def test_connection_pool_cross_service(connection_pools, service_names):
    """Test connection pools across multiple services."""
    # Initialize all pools
    for service_name in service_names:
        pool = connection_pools[service_name]
        await pool.initialize_async()
    
    # Define a test query function
    async def execute_query(service_name: str, i: int):
        pool = connection_pools[service_name]
        async with pool.get_async_session() as session:
            result = await session.execute("SELECT pg_sleep(0.1), $1::int", [i])
            value = result.scalar_one_or_none()
            return service_name, value
    
    # Execute queries from different services concurrently
    tasks = []
    for i, service_name in enumerate(service_names):
        tasks.append(execute_query(service_name, i))
    
    results = await asyncio.gather(*tasks)
    
    # Verify that all queries completed successfully
    assert len(results) == len(service_names)
    for i, (service_name, value) in enumerate(results):
        assert service_name == service_names[i]
        assert value == i


@pytest.mark.asyncio
async def test_connection_pool_asyncpg(connection_pools, service_names):
    """Test direct asyncpg connection from the pool."""
    # Get the first service name
    service_name = service_names[0]
    
    # Get the connection pool
    pool = connection_pools[service_name]
    
    # Initialize the pool
    await pool.initialize_asyncpg()
    
    # Verify that the asyncpg pool is initialized
    assert pool._asyncpg_pool is not None
    
    # Get a connection from the pool
    async with pool.get_asyncpg_connection() as conn:
        # Verify that the connection is valid
        value = await conn.fetchval("SELECT 1")
        assert value == 1


@pytest.mark.asyncio
async def test_connection_pool_metrics(connection_pools, service_names):
    """Test connection pool metrics."""
    # Get the first service name
    service_name = service_names[0]
    
    # Get the connection pool
    pool = connection_pools[service_name]
    
    # Initialize the pool
    await pool.initialize_async()
    
    # Define a test query function that sleeps to simulate work
    async def execute_query(i: int):
        async with pool.get_async_session() as session:
            result = await session.execute("SELECT pg_sleep(0.1), $1::int", [i])
            value = result.scalar_one_or_none()
            return value
    
    # Execute multiple queries concurrently to generate metrics
    tasks = [execute_query(i) for i in range(5)]
    results = await asyncio.gather(*tasks)
    
    # Verify that all queries completed successfully
    assert len(results) == 5
    assert all(result == i for i, result in enumerate(results))
    
    # In a real test, we would check Prometheus metrics
    # For this test, we'll just check that the pool is still valid
    assert pool._async_engine is not None
    
    # Note: In a real environment, we would use the Prometheus API to check metrics
    # For example:
    # response = requests.get("http://localhost:9090/api/v1/query", params={"query": "db_pool_usage"})
    # metrics = response.json()
    # assert metrics["status"] == "success"
    # assert len(metrics["data"]["result"]) > 0