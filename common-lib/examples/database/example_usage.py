"""
Example usage of the database utilities.

This script demonstrates how to use the database utilities in common_lib.database.
"""
import asyncio
import logging
from typing import List, Dict, Any

from sqlalchemy import Table, Column, Integer, String, MetaData, create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import Session, sessionmaker

from common_lib.database import (
    # Connection pool
    get_connection_pool,
    get_sync_db_session,
    get_async_db_session,
    get_asyncpg_connection,
    
    # Prepared statements
    with_prepared_statement,
    async_with_prepared_statement,
    execute_prepared_statement,
    execute_prepared_statement_async,
    execute_prepared_statement_asyncpg,
    fetch_prepared_statement_asyncpg,
    
    # Bulk operations
    bulk_insert,
    bulk_insert_async,
    bulk_update,
    bulk_update_async,
    bulk_delete,
    bulk_delete_async,
    bulk_insert_asyncpg,
    
    # Monitoring
    track_query_performance,
    async_track_query_performance,
    track_transaction,
    async_track_transaction,
    analyze_query,
    analyze_query_async,
    check_database_health,
    check_database_health_async,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a test table
metadata = MetaData()
users_table = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String),
    Column("email", String),
)


# Example 1: Using the connection pool
def example_connection_pool():
    """Example of using the connection pool."""
    logger.info("Example 1: Using the connection pool")
    
    # Get a connection pool for a service
    pool = get_connection_pool("example_service")
    
    # Use a synchronous session
    with pool.get_sync_session() as session:
        # Execute a query
        result = session.execute("SELECT 1")
        logger.info(f"Query result: {result.scalar()}")
    
    logger.info("Connection pool example completed")


# Example 2: Using prepared statements
@with_prepared_statement("example_service", "get_user")
def get_user(session: Session, user_id: int) -> Dict[str, Any]:
    """
    Get a user by ID using a prepared statement.
    
    Args:
        session: Database session
        user_id: User ID
        
    Returns:
        User data
    """
    # Execute a prepared statement
    result = execute_prepared_statement(
        session,
        "SELECT id, name, email FROM users WHERE id = :user_id",
        {"user_id": user_id},
        "example_service",
        "get_user",
    )
    
    # Get the first row
    row = result.fetchone()
    
    # Return user data
    if row:
        return {
            "id": row[0],
            "name": row[1],
            "email": row[2],
        }
    
    return None


@async_with_prepared_statement("example_service", "get_user_async")
async def get_user_async(session: AsyncSession, user_id: int) -> Dict[str, Any]:
    """
    Get a user by ID using a prepared statement asynchronously.
    
    Args:
        session: Database session
        user_id: User ID
        
    Returns:
        User data
    """
    # Execute a prepared statement
    result = await execute_prepared_statement_async(
        session,
        "SELECT id, name, email FROM users WHERE id = :user_id",
        {"user_id": user_id},
        "example_service",
        "get_user_async",
    )
    
    # Get the first row
    row = result.fetchone()
    
    # Return user data
    if row:
        return {
            "id": row[0],
            "name": row[1],
            "email": row[2],
        }
    
    return None


def example_prepared_statements():
    """Example of using prepared statements."""
    logger.info("Example 2: Using prepared statements")
    
    # Use a synchronous session
    with get_sync_db_session("example_service") as session:
        # Get a user
        user = get_user(session, 1)
        logger.info(f"User: {user}")
    
    logger.info("Prepared statements example completed")


# Example 3: Using bulk operations
def example_bulk_operations():
    """Example of using bulk operations."""
    logger.info("Example 3: Using bulk operations")
    
    # Create test data
    users = [
        {"id": 1, "name": "User 1", "email": "user1@example.com"},
        {"id": 2, "name": "User 2", "email": "user2@example.com"},
        {"id": 3, "name": "User 3", "email": "user3@example.com"},
    ]
    
    # Use a synchronous session
    with get_sync_db_session("example_service") as session:
        # Bulk insert
        bulk_insert(
            session,
            users_table,
            users,
            "example_service",
        )
        
        # Bulk update
        updated_users = [
            {"id": 1, "name": "Updated User 1", "email": "updated1@example.com"},
            {"id": 2, "name": "Updated User 2", "email": "updated2@example.com"},
        ]
        
        bulk_update(
            session,
            users_table,
            updated_users,
            "id",
            "example_service",
        )
        
        # Bulk delete
        bulk_delete(
            session,
            users_table,
            [3],
            "id",
            "example_service",
        )
    
    logger.info("Bulk operations example completed")


# Example 4: Using monitoring
@track_query_performance("select", "users", "example_service")
def get_all_users(session: Session) -> List[Dict[str, Any]]:
    """
    Get all users with performance tracking.
    
    Args:
        session: Database session
        
    Returns:
        List of users
    """
    # Execute a query
    result = session.execute("SELECT id, name, email FROM users")
    
    # Return users
    return [
        {
            "id": row[0],
            "name": row[1],
            "email": row[2],
        }
        for row in result
    ]


def example_monitoring():
    """Example of using monitoring."""
    logger.info("Example 4: Using monitoring")
    
    # Use a synchronous session
    with get_sync_db_session("example_service") as session:
        # Use transaction tracking
        with track_transaction("example_service"):
            # Get all users
            users = get_all_users(session)
            logger.info(f"Users: {users}")
            
            # Analyze a query
            plan = analyze_query(
                session,
                "SELECT * FROM users WHERE id = :user_id",
                {"user_id": 1},
                "example_service",
            )
            logger.info(f"Query plan: {plan}")
            
            # Check database health
            health = check_database_health(
                session,
                "example_service",
            )
            logger.info(f"Database health: {health}")
    
    logger.info("Monitoring example completed")


# Main function
async def main():
    """Main function."""
    # Example 1: Using the connection pool
    example_connection_pool()
    
    # Example 2: Using prepared statements
    example_prepared_statements()
    
    # Example 3: Using bulk operations
    example_bulk_operations()
    
    # Example 4: Using monitoring
    example_monitoring()


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())