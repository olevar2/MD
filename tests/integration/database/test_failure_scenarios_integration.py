"""
Integration tests for database failure scenarios.
"""
import pytest
import asyncio
import logging
from typing import Dict, List, Any, Optional
import time
from unittest.mock import patch, MagicMock, AsyncMock
from sqlalchemy.exc import OperationalError, TimeoutError
import asyncpg

from common_lib.database import (
    get_connection_pool,
    get_sync_db_session,
    get_async_db_session,
    get_asyncpg_connection,
    with_prepared_statement,
    async_with_prepared_statement,
    execute_prepared_statement,
    execute_prepared_statement_async,
    bulk_insert_async,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_connection_failure(connection_pools, service_names):
    """Test handling of database connection failures."""
    # Get the first service name
    service_name = service_names[0]
    
    # Get the connection pool
    pool = connection_pools[service_name]
    
    # Initialize the pool
    await pool.initialize_async()
    
    # Patch the get_async_session method to simulate a connection failure
    original_get_async_session = pool.get_async_session
    
    failure_count = 0
    max_failures = 3
    
    class SessionContextManager:
        async def __aenter__(self):
            nonlocal failure_count
            if failure_count < max_failures:
                failure_count += 1
                raise OperationalError("Connection failed", None, None)
            return await original_get_async_session().__aenter__()
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if failure_count >= max_failures:
                return await original_get_async_session().__aexit__(exc_type, exc_val, exc_tb)
    
    # Replace the get_async_session method
    pool.get_async_session = lambda: SessionContextManager()
    
    try:
        # Try to execute a query
        async with pool.get_async_session() as session:
            result = await session.execute("SELECT 1")
            value = result.scalar_one()
            assert value == 1
        
        # Verify that the connection was retried
        assert failure_count == max_failures
    finally:
        # Restore the original method
        pool.get_async_session = original_get_async_session


@pytest.mark.asyncio
async def test_transaction_rollback(connection_pools, service_names, clean_test_tables):
    """Test transaction rollback on failure."""
    # Get the first service name
    service_name = service_names[0]
    
    # Get the connection pool
    pool = connection_pools[service_name]
    
    # Initialize the pool
    await pool.initialize_async()
    
    # Insert some test data
    async with pool.get_async_session() as session:
        await session.execute(
            "INSERT INTO test_schema.test_table (name, value) VALUES (:name, :value)",
            {"name": "test_rollback", "value": 100},
        )
        await session.commit()
    
    # Try to execute a transaction that will fail
    try:
        async with pool.get_async_session() as session:
            # First statement succeeds
            await session.execute(
                "UPDATE test_schema.test_table SET value = :value WHERE name = :name",
                {"name": "test_rollback", "value": 200},
            )
            
            # Second statement fails (invalid SQL)
            await session.execute("THIS IS NOT VALID SQL")
            
            await session.commit()
    except Exception:
        # Exception is expected
        pass
    
    # Verify that the transaction was rolled back
    async with pool.get_async_session() as session:
        result = await session.execute(
            "SELECT value FROM test_schema.test_table WHERE name = :name",
            {"name": "test_rollback"},
        )
        value = result.scalar_one()
        assert value == 100  # Original value, not 200


@pytest.mark.asyncio
async def test_prepared_statement_failure(connection_pools, service_names):
    """Test handling of prepared statement failures."""
    # Get the first service name
    service_name = service_names[0]
    
    # Get the connection pool
    pool = connection_pools[service_name]
    
    # Initialize the pool
    await pool.initialize_async()
    
    # Define a function that uses prepared statements with an invalid query
    @async_with_prepared_statement(service_name, "invalid_query")
    async def execute_invalid_query(session):
        result = await execute_prepared_statement_async(
            session,
            "THIS IS NOT VALID SQL",
            {},
            service_name,
            "invalid_query",
        )
        return result
    
    # Try to execute the invalid query
    with pytest.raises(Exception):
        async with pool.get_async_session() as session:
            await execute_invalid_query(session)
    
    # Verify that the session is still usable
    async with pool.get_async_session() as session:
        result = await session.execute("SELECT 1")
        value = result.scalar_one()
        assert value == 1


@pytest.mark.asyncio
async def test_bulk_operation_failure(connection_pools, service_names, clean_test_tables):
    """Test handling of bulk operation failures."""
    # Get the first service name
    service_name = service_names[0]
    
    # Get the connection pool
    pool = connection_pools[service_name]
    
    # Initialize the pool
    await pool.initialize_async()
    
    # Prepare test data with invalid values
    values = [
        {"name": f"test_{i}", "value": "not_an_integer" if i == 5 else i}  # Invalid value at index 5
        for i in range(10)
    ]
    
    # Try to perform bulk insert with invalid data
    with pytest.raises(Exception):
        async with pool.get_async_session() as session:
            await bulk_insert_async(
                session,
                "test_schema.test_table",
                values,
                service_name,
            )
            await session.commit()
    
    # Verify that no data was inserted (transaction was rolled back)
    async with pool.get_async_session() as session:
        result = await session.execute("SELECT COUNT(*) FROM test_schema.test_table")
        count = result.scalar_one()
        assert count == 0


@pytest.mark.asyncio
async def test_asyncpg_connection_failure(connection_pools, service_names):
    """Test handling of asyncpg connection failures."""
    # Get the first service name
    service_name = service_names[0]
    
    # Get the connection pool
    pool = connection_pools[service_name]
    
    # Initialize the pool
    await pool.initialize_asyncpg()
    
    # Patch the get_asyncpg_connection method to simulate a connection failure
    original_get_asyncpg_connection = pool.get_asyncpg_connection
    
    failure_count = 0
    max_failures = 3
    
    class ConnectionContextManager:
        async def __aenter__(self):
            nonlocal failure_count
            if failure_count < max_failures:
                failure_count += 1
                raise asyncpg.exceptions.ConnectionFailureError("Connection failed")
            return await original_get_asyncpg_connection().__aenter__()
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if failure_count >= max_failures:
                return await original_get_asyncpg_connection().__aexit__(exc_type, exc_val, exc_tb)
    
    # Replace the get_asyncpg_connection method
    pool.get_asyncpg_connection = lambda: ConnectionContextManager()
    
    try:
        # Try to execute a query
        async with pool.get_asyncpg_connection() as conn:
            value = await conn.fetchval("SELECT 1")
            assert value == 1
        
        # Verify that the connection was retried
        assert failure_count == max_failures
    finally:
        # Restore the original method
        pool.get_asyncpg_connection = original_get_asyncpg_connection