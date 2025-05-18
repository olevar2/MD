"""
Integration tests for prepared statements.
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
    with_prepared_statement,
    async_with_prepared_statement,
    execute_prepared_statement,
    execute_prepared_statement_async,
    execute_prepared_statement_asyncpg,
    fetch_prepared_statement_asyncpg,
    get_prepared_statement_cache,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_prepared_statement_basic(connection_pools, service_names, clean_test_tables, test_data):
    """Test basic prepared statement functionality."""
    # Get the first service name
    service_name = service_names[0]
    
    # Get the connection pool
    pool = connection_pools[service_name]
    
    # Initialize the pool
    await pool.initialize_async()
    
    # Define a function that uses prepared statements
    @async_with_prepared_statement(service_name, "insert_test_data")
    async def insert_test_data(session, name, value):
        result = await execute_prepared_statement_async(
            session,
            "INSERT INTO test_schema.test_table (name, value) VALUES (:name, :value) RETURNING id",
            {"name": name, "value": value},
            service_name,
            "insert_test_data",
        )
        return result.scalar_one()
    
    # Insert test data using prepared statements
    async with pool.get_async_session() as session:
        for item in test_data["regular"][:10]:  # Insert first 10 items
            id = await insert_test_data(session, item["name"], item["value"])
            assert id is not None
    
    # Verify that the data was inserted
    async with pool.get_async_session() as session:
        result = await session.execute("SELECT COUNT(*) FROM test_schema.test_table")
        count = result.scalar_one()
        assert count == 10


@pytest.mark.asyncio
async def test_prepared_statement_cache(connection_pools, service_names):
    """Test prepared statement caching."""
    # Get the first service name
    service_name = service_names[0]
    
    # Get the connection pool
    pool = connection_pools[service_name]
    
    # Initialize the pool
    await pool.initialize_async()
    
    # Get the prepared statement cache
    cache = get_prepared_statement_cache(service_name)
    
    # Define a function that uses prepared statements
    @async_with_prepared_statement(service_name, "select_test_data")
    async def select_test_data(session, value):
        result = await execute_prepared_statement_async(
            session,
            "SELECT * FROM test_schema.test_table WHERE value > :value",
            {"value": value},
            service_name,
            "select_test_data",
        )
        return result.fetchall()
    
    # Execute the function multiple times
    async with pool.get_async_session() as session:
        for i in range(5):
            rows = await select_test_data(session, i * 10)
    
    # Verify that the statement was cached
    cache_key = f"sqlalchemy:SELECT * FROM test_schema.test_table WHERE value > :value"
    assert cache_key in cache._cache


@pytest.mark.asyncio
async def test_prepared_statement_cross_service(connection_pools, service_names, clean_test_tables, test_data):
    """Test prepared statements across multiple services."""
    # Initialize all pools
    for service_name in service_names:
        pool = connection_pools[service_name]
        await pool.initialize_async()
    
    # Define a function that uses prepared statements for each service
    async def execute_prepared_query(service_name, value):
        pool = connection_pools[service_name]
        
        @async_with_prepared_statement(service_name, "select_test_data")
        async def select_test_data(session, value):
            result = await execute_prepared_statement_async(
                session,
                "SELECT * FROM test_schema.test_table WHERE value > :value",
                {"value": value},
                service_name,
                "select_test_data",
            )
            return result.fetchall()
        
        async with pool.get_async_session() as session:
            # First, insert some test data
            for i, item in enumerate(test_data["regular"][:5]):
                await session.execute(
                    "INSERT INTO test_schema.test_table (name, value) VALUES (:name, :value)",
                    {"name": f"{service_name}_{i}", "value": i * 10},
                )
            await session.commit()
            
            # Then, select using prepared statement
            rows = await select_test_data(session, value)
            return service_name, len(rows)
    
    # Execute prepared statements from different services concurrently
    tasks = []
    for i, service_name in enumerate(service_names):
        tasks.append(execute_prepared_query(service_name, i * 5))
    
    results = await asyncio.gather(*tasks)
    
    # Verify that all queries completed successfully
    assert len(results) == len(service_names)
    
    # Verify that each service has its own prepared statement cache
    for service_name in service_names:
        cache = get_prepared_statement_cache(service_name)
        cache_key = f"sqlalchemy:SELECT * FROM test_schema.test_table WHERE value > :value"
        assert cache_key in cache._cache


@pytest.mark.asyncio
async def test_prepared_statement_asyncpg(connection_pools, service_names, clean_test_tables, test_data):
    """Test prepared statements with asyncpg."""
    # Get the first service name
    service_name = service_names[0]
    
    # Get the connection pool
    pool = connection_pools[service_name]
    
    # Initialize the pool
    await pool.initialize_asyncpg()
    
    # Insert test data using regular asyncpg
    async with pool.get_asyncpg_connection() as conn:
        for item in test_data["regular"][:10]:  # Insert first 10 items
            await conn.execute(
                "INSERT INTO test_schema.test_table (name, value) VALUES ($1, $2)",
                item["name"],
                item["value"],
            )
    
    # Define a function that uses prepared statements with asyncpg
    async def select_with_prepared_statement(value):
        async with pool.get_asyncpg_connection() as conn:
            # Prepare the statement
            stmt_name = await prepare_asyncpg_statement(
                conn,
                "SELECT * FROM test_schema.test_table WHERE value > $1",
                service_name,
                "select_test_data_asyncpg",
            )
            
            # Execute the prepared statement
            rows = await fetch_prepared_statement_asyncpg(
                conn,
                stmt_name,
                [value],
                service_name,
                "select_test_data_asyncpg",
            )
            
            return rows
    
    # Execute the function multiple times
    results = []
    for i in range(5):
        rows = await select_with_prepared_statement(i * 10)
        results.append(len(rows))
    
    # Verify that the results are as expected
    assert results[0] >= results[1] >= results[2] >= results[3] >= results[4]