"""
Tests for the database connection pool.
"""
import os
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from common_lib.database.connection_pool import (
    DatabaseConnectionPool,
    get_connection_pool,
    get_sync_db_session,
    get_async_db_session,
    get_asyncpg_connection,
)


class TestDatabaseConnectionPool(unittest.TestCase):
    """Tests for the DatabaseConnectionPool class."""
    
    def setUp(self):
        """Set up test environment."""
        # Clear singleton instances
        DatabaseConnectionPool._instances = {}
    
    def test_get_instance(self):
        """Test get_instance method."""
        # Get an instance
        pool1 = DatabaseConnectionPool.get_instance("test_service")
        
        # Get the same instance again
        pool2 = DatabaseConnectionPool.get_instance("test_service")
        
        # Check that they are the same instance
        self.assertIs(pool1, pool2)
        
        # Get a different instance
        pool3 = DatabaseConnectionPool.get_instance("other_service")
        
        # Check that it's a different instance
        self.assertIsNot(pool1, pool3)
    
    def test_get_config(self):
        """Test _get_config method."""
        # Create a pool with default config
        pool = DatabaseConnectionPool("test_service")
        
        # Check that default config values are set
        self.assertIn("database_url", pool.config)
        self.assertIn("pool_size", pool.config)
        self.assertIn("max_overflow", pool.config)
        self.assertIn("pool_timeout", pool.config)
        self.assertIn("pool_recycle", pool.config)
        self.assertIn("echo", pool.config)
        self.assertIn("prepared_statement_cache_size", pool.config)
        
        # Create a pool with custom config
        pool = DatabaseConnectionPool("test_service", database_url="postgresql://test:test@localhost:5432/test")
        
        # Check that custom config values are set
        self.assertEqual(pool.config["database_url"], "postgresql://test:test@localhost:5432/test")
        
        # Check that async_database_url is set correctly
        self.assertEqual(pool.config["async_database_url"], "postgresql+asyncpg://test:test@localhost:5432/test")
    
    @patch("common_lib.database.connection_pool.create_engine")
    @patch("common_lib.database.connection_pool.sessionmaker")
    def test_initialize_sync(self, mock_sessionmaker, mock_create_engine):
        """Test initialize_sync method."""
        # Create mock engine and session
        mock_engine = MagicMock()
        mock_session = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_sessionmaker.return_value = mock_session
        
        # Create a pool
        pool = DatabaseConnectionPool("test_service")
        
        # Initialize sync engine
        pool.initialize_sync()
        
        # Check that engine and session factory are created
        mock_create_engine.assert_called_once()
        mock_sessionmaker.assert_called_once()
        
        # Check that engine and session factory are set
        self.assertEqual(pool._sync_engine, mock_engine)
        self.assertEqual(pool._sync_session_factory, mock_session)
    
    @pytest.mark.asyncio
    @patch("common_lib.database.connection_pool.create_async_engine")
    @patch("common_lib.database.connection_pool.sessionmaker")
    async def test_initialize_async(self, mock_sessionmaker, mock_create_async_engine):
        """Test initialize_async method."""
        # Create mock engine and session
        mock_engine = AsyncMock()
        mock_session = MagicMock()
        mock_create_async_engine.return_value = mock_engine
        mock_sessionmaker.return_value = mock_session
        
        # Create a pool
        pool = DatabaseConnectionPool("test_service")
        
        # Initialize async engine
        await pool.initialize_async()
        
        # Check that engine and session factory are created
        mock_create_async_engine.assert_called_once()
        mock_sessionmaker.assert_called_once()
        
        # Check that engine and session factory are set
        self.assertEqual(pool._async_engine, mock_engine)
        self.assertEqual(pool._async_session_factory, mock_session)
        return None
    
    @pytest.mark.asyncio
    @patch("common_lib.database.connection_pool.asyncpg.create_pool")
    async def test_initialize_asyncpg(self, mock_create_pool):
        """Test initialize_asyncpg method."""
        # Create mock pool
        mock_pool = AsyncMock()
        mock_create_pool.return_value = mock_pool
        
        # Create a pool
        pool = DatabaseConnectionPool("test_service")
        
        # Initialize asyncpg pool
        await pool.initialize_asyncpg()
        
        # Check that pool is created
        mock_create_pool.assert_called_once()
        
        # Check that pool is set
        self.assertEqual(pool._asyncpg_pool, mock_pool)
        return None
    
    @patch("common_lib.database.connection_pool.DatabaseConnectionPool.initialize_sync")
    @patch("common_lib.database.connection_pool.DatabaseConnectionPool.get_sync_session")
    def test_get_sync_db_session(self, mock_get_sync_session, mock_initialize_sync):
        """Test get_sync_db_session function."""
        # Create mock session
        mock_session = MagicMock()
        mock_get_sync_session.return_value.__enter__.return_value = mock_session
        
        # Use get_sync_db_session
        with get_sync_db_session("test_service") as session:
            # Check that session is returned
            self.assertEqual(session, mock_session)
    
    @pytest.mark.asyncio
    @patch("common_lib.database.connection_pool.DatabaseConnectionPool.initialize_async")
    @patch("common_lib.database.connection_pool.DatabaseConnectionPool.get_async_session")
    async def test_get_async_db_session(self, mock_get_async_session, mock_initialize_async):
        """Test get_async_db_session function."""
        # Create mock session
        mock_session = MagicMock()
        mock_get_async_session.return_value.__aenter__.return_value = mock_session
        
        # Use get_async_db_session
        async with get_async_db_session("test_service") as session:
            # Check that session is returned
            self.assertEqual(session, mock_session)
        return None
    
    @pytest.mark.asyncio
    @patch("common_lib.database.connection_pool.DatabaseConnectionPool.initialize_asyncpg")
    @patch("common_lib.database.connection_pool.DatabaseConnectionPool.get_asyncpg_connection")
    async def test_get_asyncpg_connection(self, mock_get_asyncpg_connection, mock_initialize_asyncpg):
        """Test get_asyncpg_connection function."""
        # Create mock connection
        mock_connection = MagicMock()
        mock_get_asyncpg_connection.return_value.__aenter__.return_value = mock_connection
        
        # Use get_asyncpg_connection
        async with get_asyncpg_connection("test_service") as conn:
            # Check that connection is returned
            self.assertEqual(conn, mock_connection)
        return None


if __name__ == "__main__":
    unittest.main()