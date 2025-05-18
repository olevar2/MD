"""
Integration tests for bulk operations.
"""
import pytest
import asyncio
import logging
from typing import Dict, List, Any, Optional
import time
from sqlalchemy import Table, Column, Integer, String, MetaData, TIMESTAMP, text, Float

from common_lib.database import (
    get_connection_pool,
    get_sync_db_session,
    get_async_db_session,
    get_asyncpg_connection,
    bulk_insert,
    bulk_insert_async,
    bulk_update,
    bulk_update_async,
    bulk_delete,
    bulk_delete_async,
    bulk_insert_asyncpg,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define SQLAlchemy tables for testing
metadata = MetaData()

test_table = Table(
    "test_table",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String),
    Column("value", Integer),
    Column("created_at", TIMESTAMP),
    schema="test_schema",
)

test_time_series = Table(
    "test_time_series",
    metadata,
    Column("time", TIMESTAMP, primary_key=True),
    Column("symbol", String, primary_key=True),
    Column("value", Float),
    schema="test_schema",
)


@pytest.mark.asyncio
async def test_bulk_insert_basic(connection_pools, service_names, clean_test_tables, test_data):
    """Test basic bulk insert functionality."""
    # Get the first service name
    service_name = service_names[0]
    
    # Get the connection pool
    pool = connection_pools[service_name]
    
    # Initialize the pool
    await pool.initialize_async()
    
    # Prepare test data for bulk insert
    values = [
        {"name": item["name"], "value": item["value"]}
        for item in test_data["regular"][:50]  # Insert first 50 items
    ]
    
    # Perform bulk insert
    async with pool.get_async_session() as session:
        result = await bulk_insert_async(
            session,
            test_table,
            values,
            service_name,
            return_defaults=True,
        )
        await session.commit()
    
    # Verify that the data was inserted
    async with pool.get_async_session() as session:
        result = await session.execute("SELECT COUNT(*) FROM test_schema.test_table")
        count = result.scalar_one()
        assert count == 50


@pytest.mark.asyncio
async def test_bulk_update(connection_pools, service_names, clean_test_tables, test_data):
    """Test bulk update functionality."""
    # Get the first service name
    service_name = service_names[0]
    
    # Get the connection pool
    pool = connection_pools[service_name]
    
    # Initialize the pool
    await pool.initialize_async()
    
    # First, insert test data
    values = [
        {"name": item["name"], "value": item["value"]}
        for item in test_data["regular"][:50]  # Insert first 50 items
    ]
    
    async with pool.get_async_session() as session:
        result = await bulk_insert_async(
            session,
            test_table,
            values,
            service_name,
            return_defaults=True,
        )
        await session.commit()
    
    # Prepare test data for bulk update
    updated_values = []
    for i, item in enumerate(result):
        updated_values.append({
            "id": item["id"],
            "name": f"updated_{item['name']}",
            "value": item["value"] * 2,
        })
    
    # Perform bulk update
    async with pool.get_async_session() as session:
        count = await bulk_update_async(
            session,
            test_table,
            updated_values,
            "id",
            service_name,
        )
        await session.commit()
    
    # Verify that the data was updated
    async with pool.get_async_session() as session:
        result = await session.execute("SELECT COUNT(*) FROM test_schema.test_table WHERE name LIKE 'updated_%'")
        count = result.scalar_one()
        assert count == 50


@pytest.mark.asyncio
async def test_bulk_delete(connection_pools, service_names, clean_test_tables, test_data):
    """Test bulk delete functionality."""
    # Get the first service name
    service_name = service_names[0]
    
    # Get the connection pool
    pool = connection_pools[service_name]
    
    # Initialize the pool
    await pool.initialize_async()
    
    # First, insert test data
    values = [
        {"name": item["name"], "value": item["value"]}
        for item in test_data["regular"][:50]  # Insert first 50 items
    ]
    
    async with pool.get_async_session() as session:
        result = await bulk_insert_async(
            session,
            test_table,
            values,
            service_name,
            return_defaults=True,
        )
        await session.commit()
    
    # Get IDs to delete
    ids_to_delete = [item["id"] for item in result if item["value"] % 2 == 0]
    
    # Perform bulk delete
    async with pool.get_async_session() as session:
        count = await bulk_delete_async(
            session,
            test_table,
            ids_to_delete,
            "id",
            service_name,
        )
        await session.commit()
    
    # Verify that the data was deleted
    async with pool.get_async_session() as session:
        result = await session.execute("SELECT COUNT(*) FROM test_schema.test_table")
        count = result.scalar_one()
        assert count == 50 - len(ids_to_delete)


@pytest.mark.asyncio
async def test_bulk_insert_asyncpg(connection_pools, service_names, clean_test_tables, test_data):
    """Test bulk insert with asyncpg."""
    # Get the first service name
    service_name = service_names[0]
    
    # Get the connection pool
    pool = connection_pools[service_name]
    
    # Initialize the pool
    await pool.initialize_asyncpg()
    
    # Prepare test data for bulk insert
    columns = ["name", "value"]
    values = [
        [item["name"], item["value"]]
        for item in test_data["regular"][:100]  # Insert first 100 items
    ]
    
    # Perform bulk insert with asyncpg
    async with pool.get_asyncpg_connection() as conn:
        result = await bulk_insert_asyncpg(
            conn,
            "test_schema.test_table",
            columns,
            values,
            service_name,
            return_ids=True,
            id_column="id",
        )
    
    # Verify that the data was inserted
    async with pool.get_asyncpg_connection() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM test_schema.test_table")
        assert count == 100
    
    # Verify that IDs were returned
    assert len(result) == 100
    assert all(isinstance(id, int) for id in result)


@pytest.mark.asyncio
async def test_bulk_operations_cross_service(connection_pools, service_names, clean_test_tables, test_data):
    """Test bulk operations across multiple services."""
    # Initialize all pools
    for service_name in service_names:
        pool = connection_pools[service_name]
        await pool.initialize_async()
    
    # Define a function that performs bulk operations for each service
    async def perform_bulk_operations(service_name, start_idx, count):
        pool = connection_pools[service_name]
        
        # Prepare test data for bulk insert
        values = [
            {"name": f"{service_name}_{i}", "value": i * 10}
            for i in range(start_idx, start_idx + count)
        ]
        
        # Perform bulk insert
        async with pool.get_async_session() as session:
            result = await bulk_insert_async(
                session,
                test_table,
                values,
                service_name,
                return_defaults=True,
            )
            await session.commit()
        
        # Prepare test data for bulk update
        updated_values = []
        for item in result:
            updated_values.append({
                "id": item["id"],
                "name": f"updated_{item['name']}",
                "value": item["value"] * 2,
            })
        
        # Perform bulk update
        async with pool.get_async_session() as session:
            count = await bulk_update_async(
                session,
                test_table,
                updated_values,
                "id",
                service_name,
            )
            await session.commit()
        
        return service_name, len(result)
    
    # Execute bulk operations from different services concurrently
    tasks = []
    for i, service_name in enumerate(service_names):
        tasks.append(perform_bulk_operations(service_name, i * 25, 25))
    
    results = await asyncio.gather(*tasks)
    
    # Verify that all operations completed successfully
    assert len(results) == len(service_names)
    for service_name, count in results:
        assert count == 25
    
    # Verify that all data was inserted and updated
    async with connection_pools[service_names[0]].get_async_session() as session:
        result = await session.execute("SELECT COUNT(*) FROM test_schema.test_table")
        count = result.scalar_one()
        assert count == 25 * len(service_names)
        
        result = await session.execute("SELECT COUNT(*) FROM test_schema.test_table WHERE name LIKE 'updated_%'")
        count = result.scalar_one()
        assert count == 25 * len(service_names)


@pytest.mark.asyncio
async def test_bulk_insert_time_series(connection_pools, service_names, clean_test_tables, test_data):
    """Test bulk insert for time series data."""
    # Get the first service name
    service_name = service_names[0]
    
    # Get the connection pool
    pool = connection_pools[service_name]
    
    # Initialize the pool
    await pool.initialize_async()
    
    # Prepare test data for bulk insert
    values = [
        {"time": item["time"], "symbol": item["symbol"], "value": item["value"]}
        for item in test_data["time_series"][:1000]  # Insert first 1000 items
    ]
    
    # Perform bulk insert
    async with pool.get_async_session() as session:
        result = await bulk_insert_async(
            session,
            test_time_series,
            values,
            service_name,
            return_defaults=False,
            chunk_size=100,  # Use smaller chunks for time series data
        )
        await session.commit()
    
    # Verify that the data was inserted
    async with pool.get_async_session() as session:
        result = await session.execute("SELECT COUNT(*) FROM test_schema.test_time_series")
        count = result.scalar_one()
        assert count == 1000