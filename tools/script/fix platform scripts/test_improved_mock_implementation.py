"""
Test script to verify that the improved mock implementation works correctly.

This script tests the improved mock implementation of the database utilities.
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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'common-lib'))

# Enable mocks
import common_lib.database
common_lib.database.USE_MOCKS = True

# Import the improved mock utilities
from common_lib.database.improved_testing import (
    ImprovedMockSession,
    get_improved_mock_connection_pool,
    get_improved_mock_async_db_session,
    get_improved_mock_sync_db_session,
)


async def test_improved_mock_session():
    """Test the improved mock session."""
    logger.info("Testing improved mock session...")
    
    # Create a mock session
    session = ImprovedMockSession()
    
    # Test SELECT query
    result = await session.execute("SELECT * FROM users WHERE id = :id", {"id": 1})
    rows = result.fetchall()
    logger.info(f"SELECT result: {rows}")
    
    # Test INSERT query
    result = await session.execute(
        "INSERT INTO users (username, email) VALUES (:username, :email)",
        {"username": "new_user", "email": "new_user@example.com"},
    )
    user_id = result.scalar_one()
    logger.info(f"INSERT result: {user_id}")
    
    # Test UPDATE query
    result = await session.execute(
        "UPDATE users SET email = :email WHERE id = :id",
        {"id": 1, "email": "updated_email@example.com"},
    )
    logger.info(f"UPDATE result: {result.rowcount} rows updated")
    
    # Test DELETE query
    result = await session.execute(
        "DELETE FROM users WHERE id = :id",
        {"id": 2},
    )
    logger.info(f"DELETE result: {result.rowcount} rows deleted")
    
    # Test COUNT query
    result = await session.execute("SELECT COUNT(*) FROM users")
    count = result.scalar_one()
    logger.info(f"COUNT result: {count}")
    
    # Commit the session
    await session.commit()
    logger.info(f"Session committed: {session.committed}")
    
    # Roll back the session
    await session.rollback()
    logger.info(f"Session rolled back: {session.rolled_back}")
    
    # Close the session
    await session.close()
    logger.info(f"Session closed: {session.closed}")
    
    logger.info("Improved mock session test completed successfully!")


async def test_improved_mock_connection_pool():
    """Test the improved mock connection pool."""
    logger.info("Testing improved mock connection pool...")
    
    # Get a mock connection pool
    pool = get_improved_mock_connection_pool("test_service")
    
    # Initialize the pool
    await pool.initialize_async()
    
    # Get a session
    async with get_improved_mock_async_db_session("test_service") as session:
        # Test SELECT query
        result = await session.execute("SELECT * FROM users WHERE id = :id", {"id": 1})
        rows = result.fetchall()
        logger.info(f"SELECT result: {rows}")
        
        # Test INSERT query
        result = await session.execute(
            "INSERT INTO users (username, email) VALUES (:username, :email)",
            {"username": "new_user", "email": "new_user@example.com"},
        )
        user_id = result.scalar_one()
        logger.info(f"INSERT result: {user_id}")
        
        # Test UPDATE query
        result = await session.execute(
            "UPDATE users SET email = :email WHERE id = :id",
            {"id": 1, "email": "updated_email@example.com"},
        )
        logger.info(f"UPDATE result: {result.rowcount} rows updated")
        
        # Test DELETE query
        result = await session.execute(
            "DELETE FROM users WHERE id = :id",
            {"id": 2},
        )
        logger.info(f"DELETE result: {result.rowcount} rows deleted")
        
        # Test COUNT query
        result = await session.execute("SELECT COUNT(*) FROM users")
        count = result.scalar_one()
        logger.info(f"COUNT result: {count}")
        
        # Commit the session
        await session.commit()
    
    # Close the pool
    await pool.close_async()
    
    logger.info("Improved mock connection pool test completed successfully!")


async def test_improved_mock_sync_db_session():
    """Test the improved mock sync database session."""
    logger.info("Testing improved mock sync database session...")
    
    # Get a mock connection pool
    pool = get_improved_mock_connection_pool("test_service")
    
    # Initialize the pool
    await pool.initialize_sync()
    
    # Get a session
    session = get_improved_mock_sync_db_session("test_service")
    
    # Test SELECT query
    result = await session.execute("SELECT * FROM users WHERE id = :id", {"id": 1})
    rows = result.fetchall()
    logger.info(f"SELECT result: {rows}")
    
    # Test INSERT query
    result = await session.execute(
        "INSERT INTO users (username, email) VALUES (:username, :email)",
        {"username": "new_user", "email": "new_user@example.com"},
    )
    user_id = result.scalar_one()
    logger.info(f"INSERT result: {user_id}")
    
    # Test UPDATE query
    result = await session.execute(
        "UPDATE users SET email = :email WHERE id = :id",
        {"id": 1, "email": "updated_email@example.com"},
    )
    logger.info(f"UPDATE result: {result.rowcount} rows updated")
    
    # Test DELETE query
    result = await session.execute(
        "DELETE FROM users WHERE id = :id",
        {"id": 2},
    )
    logger.info(f"DELETE result: {result.rowcount} rows deleted")
    
    # Test COUNT query
    result = await session.execute("SELECT COUNT(*) FROM users")
    count = result.scalar_one()
    logger.info(f"COUNT result: {count}")
    
    # Commit the session
    await session.commit()
    
    # Close the pool
    pool.close_sync()
    
    logger.info("Improved mock sync database session test completed successfully!")


async def main():
    """Main function."""
    logger.info("Starting improved mock implementation tests...")
    
    # Test improved mock session
    await test_improved_mock_session()
    
    # Test improved mock connection pool
    await test_improved_mock_connection_pool()
    
    # Test improved mock sync database session
    await test_improved_mock_sync_db_session()
    
    logger.info("All improved mock implementation tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())