"""
Tests for the prepared statements module.
"""
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
import asyncpg

from common_lib.database.prepared_statements import (
    PreparedStatementCache,
    get_prepared_statement_cache,
    prepare_statement,
    prepare_asyncpg_statement,
    with_prepared_statement,
    async_with_prepared_statement,
    execute_prepared_statement,
    execute_prepared_statement_async,
    execute_prepared_statement_asyncpg,
    fetch_prepared_statement_asyncpg,
)


class TestPreparedStatementCache(unittest.TestCase):
    """Tests for the PreparedStatementCache class."""
    
    def setUp(self):
        """Set up test environment."""
        # Clear singleton instances
        PreparedStatementCache._instances = {}
    
    def test_get_instance(self):
        """Test get_instance method."""
        # Get an instance
        cache1 = PreparedStatementCache.get_instance("test_service")
        
        # Get the same instance again
        cache2 = PreparedStatementCache.get_instance("test_service")
        
        # Check that they are the same instance
        self.assertIs(cache1, cache2)
        
        # Get a different instance
        cache3 = PreparedStatementCache.get_instance("other_service")
        
        # Check that it's a different instance
        self.assertIsNot(cache1, cache3)
    
    def test_get_set(self):
        """Test get and set methods."""
        # Create a cache
        cache = PreparedStatementCache("test_service")
        
        # Set a value
        cache.set("test_key", "test_value")
        
        # Get the value
        value = cache.get("test_key")
        
        # Check that the value is returned
        self.assertEqual(value, "test_value")
        
        # Get a non-existent value
        value = cache.get("non_existent_key")
        
        # Check that None is returned
        self.assertIsNone(value)
    
    def test_max_size(self):
        """Test max_size limit."""
        # Create a cache with max_size=2
        cache = PreparedStatementCache("test_service", max_size=2)
        
        # Set two values
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Check that both values are in the cache
        self.assertEqual(cache.get("key1"), "value1")
        self.assertEqual(cache.get("key2"), "value2")
        
        # Set a third value
        cache.set("key3", "value3")
        
        # Check that one of the original values is no longer in the cache
        self.assertTrue(cache.get("key1") is None or cache.get("key2") is None)
        
        # Check that the new value is in the cache
        self.assertEqual(cache.get("key3"), "value3")
    
    @pytest.mark.asyncio
    async def test_get_or_set(self):
        """Test get_or_set method."""
        # Create a cache
        cache = PreparedStatementCache("test_service")
        
        # Create a mock create_func
        create_func = MagicMock(return_value="test_value")
        
        # Get or set a value
        value = await cache.get_or_set("test_key", create_func)
        
        # Check that the value is returned
        self.assertEqual(value, "test_value")
        
        # Check that create_func was called
        create_func.assert_called_once()
        
        # Reset the mock
        create_func.reset_mock()
        
        # Get or set the same value again
        value = await cache.get_or_set("test_key", create_func)
        
        # Check that the value is returned
        self.assertEqual(value, "test_value")
        
        # Check that create_func was not called
        create_func.assert_not_called()
        
        return None
    
    def test_clear(self):
        """Test clear method."""
        # Create a cache
        cache = PreparedStatementCache("test_service")
        
        # Set some values
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Check that the values are in the cache
        self.assertEqual(cache.get("key1"), "value1")
        self.assertEqual(cache.get("key2"), "value2")
        
        # Clear the cache
        cache.clear()
        
        # Check that the values are no longer in the cache
        self.assertIsNone(cache.get("key1"))
        self.assertIsNone(cache.get("key2"))


class TestPreparedStatements(unittest.TestCase):
    """Tests for the prepared statements functions."""
    
    def setUp(self):
        """Set up test environment."""
        # Clear singleton instances
        PreparedStatementCache._instances = {}
    
    @patch("common_lib.database.prepared_statements.text")
    def test_prepare_statement(self, mock_text):
        """Test prepare_statement function."""
        # Create mock session and statement
        mock_session = MagicMock()
        mock_stmt = MagicMock()
        mock_text.return_value = mock_stmt
        
        # Prepare a statement
        stmt = prepare_statement(mock_session, "SELECT * FROM test")
        
        # Check that text was called
        mock_text.assert_called_once_with("SELECT * FROM test")
        
        # Check that the statement is returned
        self.assertEqual(stmt, mock_stmt)
        
        # Prepare the same statement again
        stmt = prepare_statement(mock_session, "SELECT * FROM test")
        
        # Check that text was called only once (cached)
        mock_text.assert_called_once()
    
    @pytest.mark.asyncio
    @patch("common_lib.database.prepared_statements.PreparedStatementCache.get")
    @patch("common_lib.database.prepared_statements.PreparedStatementCache.set")
    async def test_prepare_asyncpg_statement(self, mock_set, mock_get):
        """Test prepare_asyncpg_statement function."""
        # Create mock connection
        mock_conn = AsyncMock()
        
        # Mock cache get to return None (not in cache)
        mock_get.return_value = None
        
        # Prepare a statement
        stmt_name = await prepare_asyncpg_statement(mock_conn, "SELECT * FROM test")
        
        # Check that connection.prepare was called
        mock_conn.prepare.assert_called_once()
        
        # Check that cache.set was called
        mock_set.assert_called_once()
        
        # Reset mocks
        mock_conn.reset_mock()
        mock_set.reset_mock()
        
        # Mock cache get to return a statement name (in cache)
        mock_get.return_value = "stmt_name"
        
        # Prepare the same statement again
        stmt_name = await prepare_asyncpg_statement(mock_conn, "SELECT * FROM test")
        
        # Check that connection.prepare was not called
        mock_conn.prepare.assert_not_called()
        
        # Check that cache.set was not called
        mock_set.assert_not_called()
        
        return None
    
    def test_with_prepared_statement(self):
        """Test with_prepared_statement decorator."""
        # Create a decorated function
        @with_prepared_statement("test_service", "test_operation")
        def test_func():
            return "test_result"
        
        # Call the function
        result = test_func()
        
        # Check that the result is returned
        self.assertEqual(result, "test_result")
    
    @pytest.mark.asyncio
    async def test_async_with_prepared_statement(self):
        """Test async_with_prepared_statement decorator."""
        # Create a decorated function
        @async_with_prepared_statement("test_service", "test_operation")
        async def test_func():
            return "test_result"
        
        # Call the function
        result = await test_func()
        
        # Check that the result is returned
        self.assertEqual(result, "test_result")
        
        return None
    
    @patch("common_lib.database.prepared_statements.prepare_statement")
    def test_execute_prepared_statement(self, mock_prepare_statement):
        """Test execute_prepared_statement function."""
        # Create mock session and statement
        mock_session = MagicMock()
        mock_stmt = MagicMock()
        mock_prepare_statement.return_value = mock_stmt
        
        # Execute a statement
        result = execute_prepared_statement(
            mock_session,
            "SELECT * FROM test",
            {"param": "value"},
            "test_service",
            "test_operation"
        )
        
        # Check that prepare_statement was called
        mock_prepare_statement.assert_called_once()
        
        # Check that session.execute was called
        mock_session.execute.assert_called_once_with(mock_stmt, {"param": "value"})
        
        # Check that the result is returned
        self.assertEqual(result, mock_session.execute.return_value)
    
    @pytest.mark.asyncio
    @patch("common_lib.database.prepared_statements.prepare_statement")
    async def test_execute_prepared_statement_async(self, mock_prepare_statement):
        """Test execute_prepared_statement_async function."""
        # Create mock session and statement
        mock_session = AsyncMock()
        mock_stmt = MagicMock()
        mock_prepare_statement.return_value = mock_stmt
        
        # Execute a statement
        result = await execute_prepared_statement_async(
            mock_session,
            "SELECT * FROM test",
            {"param": "value"},
            "test_service",
            "test_operation"
        )
        
        # Check that prepare_statement was called
        mock_prepare_statement.assert_called_once()
        
        # Check that session.execute was called
        mock_session.execute.assert_called_once_with(mock_stmt, {"param": "value"})
        
        # Check that the result is returned
        self.assertEqual(result, mock_session.execute.return_value)
        
        return None
    
    @pytest.mark.asyncio
    @patch("common_lib.database.prepared_statements.prepare_asyncpg_statement")
    async def test_execute_prepared_statement_asyncpg(self, mock_prepare_asyncpg_statement):
        """Test execute_prepared_statement_asyncpg function."""
        # Create mock connection
        mock_conn = AsyncMock()
        
        # Mock prepare_asyncpg_statement to return a statement name
        mock_prepare_asyncpg_statement.return_value = "stmt_name"
        
        # Execute a statement
        result = await execute_prepared_statement_asyncpg(
            mock_conn,
            "SELECT * FROM test",
            ["param_value"],
            "test_service",
            "test_operation"
        )
        
        # Check that prepare_asyncpg_statement was called
        mock_prepare_asyncpg_statement.assert_called_once()
        
        # Check that connection.execute was called
        mock_conn.execute.assert_called_once_with("stmt_name", "param_value")
        
        # Check that the result is returned
        self.assertEqual(result, mock_conn.execute.return_value)
        
        return None
    
    @pytest.mark.asyncio
    @patch("common_lib.database.prepared_statements.prepare_asyncpg_statement")
    async def test_fetch_prepared_statement_asyncpg(self, mock_prepare_asyncpg_statement):
        """Test fetch_prepared_statement_asyncpg function."""
        # Create mock connection
        mock_conn = AsyncMock()
        
        # Mock prepare_asyncpg_statement to return a statement name
        mock_prepare_asyncpg_statement.return_value = "stmt_name"
        
        # Fetch results from a statement
        result = await fetch_prepared_statement_asyncpg(
            mock_conn,
            "SELECT * FROM test",
            ["param_value"],
            "test_service",
            "test_operation"
        )
        
        # Check that prepare_asyncpg_statement was called
        mock_prepare_asyncpg_statement.assert_called_once()
        
        # Check that connection.fetch was called
        mock_conn.fetch.assert_called_once_with("stmt_name", "param_value")
        
        # Check that the result is returned
        self.assertEqual(result, mock_conn.fetch.return_value)
        
        return None


if __name__ == "__main__":
    unittest.main()