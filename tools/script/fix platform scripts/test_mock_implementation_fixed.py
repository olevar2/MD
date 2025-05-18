"""
Test script to verify that the mock implementation works correctly.

This script tests the mock implementation of the database utilities.
"""
import os
import sys
import asyncio
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add common-lib to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'common-lib'))

# Enable mocks
import common_lib.database
common_lib.database.USE_MOCKS = True

# Import the mock utilities directly
from common_lib.database.testing import (
    get_mock_connection_pool,
    get_mock_async_db_session,
    get_mock_sync_db_session,
    mock_execute_prepared_statement,
    mock_bulk_insert,
    mock_bulk_update,
    mock_bulk_delete,
    mock_track_query_performance,
    mock_track_transaction,
    mock_analyze_query,
    mock_check_database_health,
    mock_prepare_asyncpg_statement,
    mock_fetch_prepared_statement_asyncpg,
)

from common_lib.database import (
    with_prepared_statement,
    execute_prepared_statement,
    bulk_insert,
    track_query_performance,
    track_transaction,
)


@with_prepared_statement("test_service", "test_operation")
async def test_prepared_statement(session, param1, param2):
    """Test prepared statement."""
    query = "SELECT * FROM test_table WHERE id = :id AND name = :name"
    params = {"id": param1, "name": param2}
    
    result = await execute_prepared_statement(
        session, query, params, "test_service", "test_operation"
    )
    
    return result


async def test_prepared_statements():
    """Test prepared statements."""
    logger.info("Testing prepared statements...")
    
    # Get a mock connection pool
    pool = get_mock_connection_pool("test_service")
    
    # Initialize the pool
    await pool.initialize_async()
    
    # Get a session
    async with get_mock_async_db_session("test_service") as session:
        # Execute a prepared statement
        result = await test_prepared_statement(session, 1, "test")
        
        # Get the results
        rows = result.fetchall()
        logger.info(f"Query result: {rows}")
        
        # Commit the session
        await session.commit()
    
    # Close the pool
    await pool.close_async()
    
    logger.info("Prepared statements test completed successfully!")


@track_query_performance("test_operation", "test_table", "test_service")
async def test_tracked_query(session):
    """Test tracked query."""
    query = "SELECT * FROM test_table WHERE id = :id"
    params = {"id": 1}
    
    result = await session.execute(query, params)
    
    return result


@track_transaction("test_operation", "test_service")
async def test_tracked_transaction(session):
    """Test tracked transaction."""
    query = "INSERT INTO test_table (name, value) VALUES (:name, :value)"
    params = {"name": "test", "value": 100}
    
    await session.execute(query, params)
    await session.commit()


async def test_monitoring():
    """Test monitoring utilities."""
    logger.info("Testing monitoring utilities...")
    
    # Get a mock connection pool
    pool = get_mock_connection_pool("test_service")
    
    # Initialize the pool
    await pool.initialize_async()
    
    # Get a session
    async with get_mock_async_db_session("test_service") as session:
        # Execute a tracked query
        result = await test_tracked_query(session)
        rows = result.fetchall()
        logger.info(f"Tracked query result: {rows}")
        
        # Execute a tracked transaction
        await test_tracked_transaction(session)
        
        # Analyze a query
        query = "SELECT * FROM test_table WHERE id = :id"
        params = {"id": 1}
        
        # Use the mock implementation directly
        analysis = await mock_analyze_query(session, query, params, "test_service")
        logger.info(f"Query analysis: {analysis}")
        
        # Check database health
        health = await mock_check_database_health("test_service")
        logger.info(f"Database health: {health}")
    
    # Close the pool
    await pool.close_async()
    
    logger.info("Monitoring test completed successfully!")


async def test_bulk_operations():
    """Test bulk operations."""
    logger.info("Testing bulk operations...")
    
    # Get a mock connection pool
    pool = get_mock_connection_pool("test_service")
    
    # Initialize the pool
    await pool.initialize_async()
    
    # Get a session
    async with get_mock_async_db_session("test_service") as session:
        # Create test data
        data = [
            {"id": i, "name": f"test_{i}", "value": i * 10}
            for i in range(1, 101)
        ]
        
        # Bulk insert
        result = await mock_bulk_insert(session, "test_table", data, "test_service")
        logger.info(f"Bulk insert result: {result}")
        
        # Bulk update
        result = await mock_bulk_update(session, "test_table", data, "id", "test_service")
        logger.info(f"Bulk update result: {result}")
        
        # Bulk delete
        primary_keys = [i for i in range(1, 101)]
        result = await mock_bulk_delete(session, "test_table", primary_keys, "id", "test_service")
        logger.info(f"Bulk delete result: {result}")
        
        # Commit the session
        await session.commit()
    
    # Close the pool
    await pool.close_async()
    
    logger.info("Bulk operations test completed successfully!")


async def test_connection_pool():
    """Test the connection pool."""
    logger.info("Testing connection pool...")
    
    # Get a mock connection pool
    pool = get_mock_connection_pool("test_service")
    
    # Initialize the pool
    await pool.initialize_async()
    
    # Get a session
    async with get_mock_async_db_session("test_service") as session:
        # Execute a query
        result = await session.execute("SELECT 1")
        value = result.scalar_one()
        logger.info(f"Query result: {value}")
        
        # Commit the session
        await session.commit()
    
    # Close the pool
    await pool.close_async()
    
    logger.info("Connection pool test completed successfully!")


async def test_asyncpg():
    """Test asyncpg utilities."""
    logger.info("Testing asyncpg utilities...")
    
    # Get a mock connection pool
    pool = get_mock_connection_pool("test_service")
    
    # Initialize the pool
    await pool.initialize_asyncpg()
    
    # Get a connection
    async with pool.get_asyncpg_connection() as conn:
        # Prepare a statement
        query = "SELECT * FROM test_table WHERE id = $1"
        stmt_name = await mock_prepare_asyncpg_statement(conn, query, "test_service")
        logger.info(f"Prepared statement: {stmt_name}")
        
        # Fetch from the prepared statement
        params = [1]
        rows = await mock_fetch_prepared_statement_asyncpg(conn, stmt_name, params, "test_service")
        logger.info(f"Fetch result: {rows}")
    
    # Close the pool
    await pool.close_async()
    
    logger.info("Asyncpg test completed successfully!")


async def main():
    """Main function."""
    logger.info("Starting database utilities tests with mocks...")
    
    # Test connection pool
    await test_connection_pool()
    
    # Test prepared statements
    await test_prepared_statements()
    
    # Test monitoring
    await test_monitoring()
    
    # Test bulk operations
    await test_bulk_operations()
    
    # Test asyncpg
    await test_asyncpg()
    
    logger.info("All tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())