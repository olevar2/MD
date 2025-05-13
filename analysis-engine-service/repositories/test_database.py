"""
Tests for database and cache functionality.

This module contains tests for database connections, queries,
and cache management.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from analysis_engine.core.database import DatabaseManager # Keep this if type hints are used
from analysis_engine.core.cache import CacheManager # Keep this if type hints are used
from analysis_engine.core.errors import DatabaseError, CacheError
# Fixtures like 'mock_db_pool', 'mock_redis', 'db_manager', 'cache_manager' 
# are now imported from conftest.py (using the names 'mock_db_pool', 'mock_redis', 
# 'mock_db_manager', 'mock_cache_manager' respectively)

@pytest.mark.asyncio
async def test_database_connection(mock_db_manager: DatabaseManager, mock_db_pool: AsyncMock): # Use shared fixtures
    """Test database connection management."""
    async with db_manager.get_connection() as conn:
        assert conn is not None
        mock_db_pool.acquire.assert_called_once()
    
    mock_db_pool.release.assert_called_once()

@pytest.mark.asyncio
async def test_database_query(mock_db_manager: DatabaseManager, mock_db_pool: AsyncMock): # Use shared fixtures
    """Test database query execution."""
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=[{"id": 1, "name": "test"}])
    mock_db_pool.acquire.return_value = mock_conn
    
    result = await db_manager.execute_query("SELECT * FROM test")
    assert len(result) == 1
    assert result[0]["id"] == 1
    assert result[0]["name"] == "test"

@pytest.mark.asyncio
async def test_database_error_handling(mock_db_manager: DatabaseManager, mock_db_pool: AsyncMock): # Use shared fixtures
    """Test database error handling."""
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(side_effect=Exception("Database error"))
    mock_db_pool.acquire.return_value = mock_conn
    
    with pytest.raises(DatabaseError):
        await db_manager.execute_query("SELECT * FROM test")

@pytest.mark.asyncio
async def test_cache_set_get(mock_cache_manager: CacheManager, mock_redis: AsyncMock): # Use shared fixtures
    """Test cache set and get operations."""
    test_key = "test_key"
    test_value = "test_value"
    
    await cache_manager.set(test_key, test_value)
    mock_redis.set.assert_called_once_with(test_key, test_value)
    
    mock_redis.get.return_value = test_value
    value = await cache_manager.get(test_key)
    assert value == test_value

@pytest.mark.asyncio
async def test_cache_delete(mock_cache_manager: CacheManager, mock_redis: AsyncMock): # Use shared fixtures
    """Test cache delete operation."""
    test_key = "test_key"
    
    await cache_manager.delete(test_key)
    mock_redis.delete.assert_called_once_with(test_key)

@pytest.mark.asyncio
async def test_cache_error_handling(mock_cache_manager: CacheManager, mock_redis: AsyncMock): # Use shared fixtures
    """Test cache error handling."""
    mock_redis.get.side_effect = Exception("Cache error")
    
    with pytest.raises(CacheError):
        await cache_manager.get("test_key")

@pytest.mark.asyncio
async def test_database_transaction(mock_db_manager: DatabaseManager, mock_db_pool: AsyncMock): # Use shared fixtures
    """Test database transaction management."""
    mock_conn = AsyncMock()
    mock_db_pool.acquire.return_value = mock_conn
    
    async with db_manager.transaction() as transaction:
        await transaction.execute("INSERT INTO test (name) VALUES ('test')")
        await transaction.execute("UPDATE test SET name = 'updated' WHERE name = 'test'")
    
    mock_conn.commit.assert_called_once()

@pytest.mark.asyncio
async def test_database_transaction_rollback(mock_db_manager: DatabaseManager, mock_db_pool: AsyncMock): # Use shared fixtures
    """Test database transaction rollback."""
    mock_conn = AsyncMock()
    mock_db_pool.acquire.return_value = mock_conn
    
    with pytest.raises(DatabaseError):
        async with db_manager.transaction() as transaction:
            await transaction.execute("INSERT INTO test (name) VALUES ('test')")
            raise Exception("Test error")
    
    mock_conn.rollback.assert_called_once()

@pytest.mark.asyncio
async def test_cache_expiration(mock_cache_manager: CacheManager, mock_redis: AsyncMock): # Use shared fixtures
    """Test cache expiration."""
    test_key = "test_key"
    test_value = "test_value"
    expiration = 3600
    
    await cache_manager.set(test_key, test_value, expiration)
    mock_redis.set.assert_called_once_with(test_key, test_value, ex=expiration)

@pytest.mark.asyncio
async def test_database_connection_pool(mock_db_manager: DatabaseManager, mock_db_pool: AsyncMock): # Use shared fixtures
    """Test database connection pool management."""
    # Test multiple connections
    async with db_manager.get_connection() as conn1:
        async with db_manager.get_connection() as conn2:
            assert conn1 is not conn2
    
    assert mock_db_pool.acquire.call_count == 2
    assert mock_db_pool.release.call_count == 2

@pytest.mark.asyncio
async def test_cache_batch_operations(mock_cache_manager: CacheManager, mock_redis: AsyncMock): # Use shared fixtures
    """Test cache batch operations."""
    test_data = {
        "key1": "value1",
        "key2": "value2",
        "key3": "value3"
    }
    
    # Test batch set
    await cache_manager.batch_set(test_data)
    assert mock_redis.set.call_count == 3
    
    # Test batch get
    mock_redis.get.side_effect = ["value1", "value2", "value3"]
    values = await cache_manager.batch_get(list(test_data.keys()))
    assert len(values) == 3
    assert values == ["value1", "value2", "value3"]

@pytest.mark.asyncio
async def test_database_query_timeout(mock_db_manager: DatabaseManager, mock_db_pool: AsyncMock): # Use shared fixtures
    """Test database query timeout."""
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(side_effect=asyncio.TimeoutError())
    mock_db_pool.acquire.return_value = mock_conn
    
    with pytest.raises(DatabaseError) as exc_info:
        await db_manager.execute_query("SELECT * FROM test", timeout=1)
    
    assert "timeout" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_cache_connection_error(mock_cache_manager: CacheManager, mock_redis: AsyncMock): # Use shared fixtures
    """Test cache connection error handling."""
    mock_redis.get.side_effect = ConnectionError("Redis connection error")
    
    with pytest.raises(CacheError) as exc_info:
        await cache_manager.get("test_key")
    
    assert "connection" in str(exc_info.value).lower()