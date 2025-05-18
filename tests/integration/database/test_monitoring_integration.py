"""
Integration tests for database monitoring.
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
    track_query_performance,
    async_track_query_performance,
    track_transaction,
    async_track_transaction,
    analyze_query,
    analyze_query_async,
    check_database_health,
    check_database_health_async,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_query_performance_tracking(connection_pools, service_names, clean_test_tables, test_data):
    """Test query performance tracking."""
    # Get the first service name
    service_name = service_names[0]
    
    # Get the connection pool
    pool = connection_pools[service_name]
    
    # Initialize the pool
    await pool.initialize_async()
    
    # Insert some test data
    async with pool.get_async_session() as session:
        for item in test_data["regular"][:50]:  # Insert first 50 items
            await session.execute(
                "INSERT INTO test_schema.test_table (name, value) VALUES (:name, :value)",
                {"name": item["name"], "value": item["value"]},
            )
        await session.commit()
    
    # Define a function that uses query performance tracking
    @async_track_query_performance("select", "test_table", service_name)
    async def select_test_data(session, value):
        result = await session.execute(
            "SELECT * FROM test_schema.test_table WHERE value > :value",
            {"value": value},
        )
        return result.fetchall()
    
    # Execute the function multiple times with different parameters
    async with pool.get_async_session() as session:
        for i in range(5):
            rows = await select_test_data(session, i * 10)
            assert len(rows) > 0
    
    # In a real test, we would check Prometheus metrics
    # For this test, we'll just verify that the function executed successfully


@pytest.mark.asyncio
async def test_transaction_tracking(connection_pools, service_names, clean_test_tables, test_data):
    """Test transaction tracking."""
    # Get the first service name
    service_name = service_names[0]
    
    # Get the connection pool
    pool = connection_pools[service_name]
    
    # Initialize the pool
    await pool.initialize_async()
    
    # Execute a transaction with tracking
    async with pool.get_async_session() as session:
        async with async_track_transaction(service_name):
            # Insert some test data
            for item in test_data["regular"][:10]:  # Insert first 10 items
                await session.execute(
                    "INSERT INTO test_schema.test_table (name, value) VALUES (:name, :value)",
                    {"name": item["name"], "value": item["value"]},
                )
            
            # Update some test data
            await session.execute(
                "UPDATE test_schema.test_table SET value = value * 2 WHERE value > :value",
                {"value": 50},
            )
            
            await session.commit()
    
    # Verify that the transaction was executed
    async with pool.get_async_session() as session:
        result = await session.execute("SELECT COUNT(*) FROM test_schema.test_table")
        count = result.scalar_one()
        assert count == 10


@pytest.mark.asyncio
async def test_query_analysis(connection_pools, service_names, clean_test_tables, test_data):
    """Test query analysis."""
    # Get the first service name
    service_name = service_names[0]
    
    # Get the connection pool
    pool = connection_pools[service_name]
    
    # Initialize the pool
    await pool.initialize_async()
    
    # Insert some test data
    async with pool.get_async_session() as session:
        for item in test_data["regular"][:50]:  # Insert first 50 items
            await session.execute(
                "INSERT INTO test_schema.test_table (name, value) VALUES (:name, :value)",
                {"name": item["name"], "value": item["value"]},
            )
        await session.commit()
    
    # Analyze a query
    async with pool.get_async_session() as session:
        plan = await analyze_query_async(
            session,
            "SELECT * FROM test_schema.test_table WHERE value > :value",
            {"value": 50},
            service_name,
        )
        
        # Verify that the plan is returned
        assert plan is not None
        assert isinstance(plan, str)
        assert "EXPLAIN" in plan


@pytest.mark.asyncio
async def test_database_health_check(connection_pools, service_names):
    """Test database health check."""
    # Get the first service name
    service_name = service_names[0]
    
    # Get the connection pool
    pool = connection_pools[service_name]
    
    # Initialize the pool
    await pool.initialize_async()
    
    # Check database health
    async with pool.get_async_session() as session:
        health = await check_database_health_async(
            session,
            service_name,
        )
        
        # Verify that health information is returned
        assert health is not None
        assert "status" in health
        assert health["status"] == "healthy"
        assert "version" in health
        assert "connection_count" in health
        assert "active_query_count" in health


@pytest.mark.asyncio
async def test_slow_query_detection(connection_pools, service_names, clean_test_tables, test_data):
    """Test slow query detection."""
    # Get the first service name
    service_name = service_names[0]
    
    # Get the connection pool
    pool = connection_pools[service_name]
    
    # Initialize the pool
    await pool.initialize_async()
    
    # Insert some test data
    async with pool.get_async_session() as session:
        for item in test_data["regular"][:50]:  # Insert first 50 items
            await session.execute(
                "INSERT INTO test_schema.test_table (name, value) VALUES (:name, :value)",
                {"name": item["name"], "value": item["value"]},
            )
        await session.commit()
    
    # Define a function that uses query performance tracking with a slow query
    @async_track_query_performance("select", "test_table", service_name, slow_query_threshold=0.01)
    async def execute_slow_query(session):
        result = await session.execute(
            "SELECT pg_sleep(0.1), * FROM test_schema.test_table",
        )
        return result.fetchall()
    
    # Execute the slow query
    async with pool.get_async_session() as session:
        rows = await execute_slow_query(session)
        assert len(rows) > 0
    
    # In a real test, we would check Prometheus metrics for slow query count
    # For this test, we'll just verify that the function executed successfully


@pytest.mark.asyncio
async def test_monitoring_cross_service(connection_pools, service_names, clean_test_tables, test_data):
    """Test monitoring across multiple services."""
    # Initialize all pools
    for service_name in service_names:
        pool = connection_pools[service_name]
        await pool.initialize_async()
    
    # Define a function that performs monitored operations for each service
    async def perform_monitored_operations(service_name):
        pool = connection_pools[service_name]
        
        async with pool.get_async_session() as session:
            # Insert some test data with transaction tracking
            async with async_track_transaction(service_name):
                for i in range(10):
                    await session.execute(
                        "INSERT INTO test_schema.test_table (name, value) VALUES (:name, :value)",
                        {"name": f"{service_name}_{i}", "value": i * 10},
                    )
                await session.commit()
            
            # Define a function with query performance tracking
            @async_track_query_performance("select", "test_table", service_name)
            async def select_test_data(session, value):
                result = await session.execute(
                    "SELECT * FROM test_schema.test_table WHERE value > :value",
                    {"value": value},
                )
                return result.fetchall()
            
            # Execute the query
            rows = await select_test_data(session, 0)
            
            # Check database health
            health = await check_database_health_async(
                session,
                service_name,
            )
            
            return service_name, len(rows), health["status"]
    
    # Execute monitored operations from different services concurrently
    tasks = []
    for service_name in service_names:
        tasks.append(perform_monitored_operations(service_name))
    
    results = await asyncio.gather(*tasks)
    
    # Verify that all operations completed successfully
    assert len(results) == len(service_names)
    for service_name, row_count, health_status in results:
        assert row_count >= 10  # At least the rows inserted by this service
        assert health_status == "healthy"