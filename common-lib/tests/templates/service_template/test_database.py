"""
Unit tests for the service template database module.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from common_lib.templates.service_template.database import Database
from common_lib.config import DatabaseConfig


class TestDatabase:
    """Tests for the Database class."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()

    @pytest.fixture
    def mock_config(self):
        """Create a mock database config."""
        return DatabaseConfig(
            host="localhost",
            port=5432,
            username="postgres",
            password="postgres",
            database="test",
            min_connections=1,
            max_connections=10
        )

    @pytest.fixture
    def database(self, mock_config, mock_logger):
        """Create a database instance."""
        return Database(config=mock_config, logger=mock_logger)

    def test_init(self, database, mock_config, mock_logger):
        """Test initialization."""
        assert database.config == mock_config
        assert database.logger == mock_logger
        assert database.pool is None

    @patch("common_lib.templates.service_template.database.get_database_config")
    def test_init_with_default_config(self, mock_get_database_config, mock_logger):
        """Test initialization with default config."""
        # Mock the get_database_config function
        mock_config = MagicMock()
        mock_get_database_config.return_value = mock_config

        # Create a database instance
        database = Database(logger=mock_logger)

        # Verify the result
        assert database.config == mock_config
        assert database.logger == mock_logger
        assert database.pool is None

        # Verify the mock was called
        mock_get_database_config.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect(self, database):
        """Test connect method."""
        # Set the pool
        database.pool = MagicMock()

        # Call the method
        await database.connect()

        # Verify the pool was not changed
        assert database.pool is not None

    @pytest.mark.asyncio
    @patch("common_lib.templates.service_template.database.asyncpg.create_pool")
    async def test_connect_error(self, mock_create_pool, database):
        """Test connect method with error."""
        # Mock the create_pool function to raise an exception
        mock_create_pool.side_effect = Exception("Connection error")

        # Call the method
        with pytest.raises(Exception) as excinfo:
            await database.connect()

        # Verify the exception
        assert str(excinfo.value) == "Connection error"

        # Verify the logger was called
        database.logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_already_connected(self, database):
        """Test connect method when already connected."""
        # Set the pool
        database.pool = MagicMock()

        # Call the method
        await database.connect()

        # Verify the pool was not changed
        assert database.pool is not None

    @pytest.mark.asyncio
    async def test_close(self, database):
        """Test close method."""
        # Set the pool
        mock_pool = AsyncMock()
        database.pool = mock_pool

        # Call the method
        await database.close()

        # Verify the result
        assert database.pool is None

        # Verify the mock was called
        mock_pool.close.assert_called_once()

        # Verify the logger was called
        database.logger.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_not_connected(self, database):
        """Test close method when not connected."""
        # Call the method
        await database.close()

        # Verify the pool is still None
        assert database.pool is None

    @pytest.mark.asyncio
    async def test_execute(self, database):
        """Test execute method."""
        # Set the pool
        mock_pool = AsyncMock()
        mock_pool.execute.return_value = "result"
        database.pool = mock_pool

        # Call the method
        result = await database.execute("SELECT 1", 1, 2, timeout=10)

        # Verify the result
        assert result == "result"

        # Verify the mock was called
        mock_pool.execute.assert_called_once_with("SELECT 1", 1, 2, timeout=10)

    @pytest.mark.asyncio
    async def test_execute_not_connected(self, database):
        """Test execute method when not connected."""
        # Set the pool to None
        database.pool = None

        # Create a mock pool
        mock_pool = AsyncMock()
        mock_pool.execute.return_value = "result"

        # Mock the connect method
        database.connect = AsyncMock(side_effect=lambda: setattr(database, "pool", mock_pool))

        # Call the method
        result = await database.execute("SELECT 1", 1, 2, timeout=10)

        # Verify the result
        assert result == "result"

        # Verify the mocks were called
        database.connect.assert_called_once()
        mock_pool.execute.assert_called_once_with("SELECT 1", 1, 2, timeout=10)

    @pytest.mark.asyncio
    async def test_execute_error(self, database):
        """Test execute method with error."""
        # Set the pool
        mock_pool = AsyncMock()
        mock_pool.execute.side_effect = Exception("Query error")
        database.pool = mock_pool

        # Call the method
        with pytest.raises(Exception) as excinfo:
            await database.execute("SELECT 1")

        # Verify the exception
        assert str(excinfo.value) == "Query error"

        # Verify the logger was called
        database.logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch(self, database):
        """Test fetch method."""
        # Set the pool
        mock_pool = AsyncMock()
        mock_pool.fetch.return_value = [
            {"id": 1, "name": "test1"},
            {"id": 2, "name": "test2"}
        ]
        database.pool = mock_pool

        # Call the method
        result = await database.fetch("SELECT * FROM test", 1, 2, timeout=10)

        # Verify the result
        assert len(result) == 2

        # Verify the mock was called
        mock_pool.fetch.assert_called_once_with("SELECT * FROM test", 1, 2, timeout=10)

    @pytest.mark.asyncio
    async def test_fetchrow(self, database):
        """Test fetchrow method."""
        # Set the pool
        mock_pool = AsyncMock()
        mock_pool.fetchrow.return_value = {"id": 1, "name": "test"}
        database.pool = mock_pool

        # Call the method
        result = await database.fetchrow("SELECT * FROM test WHERE id = $1", 1, timeout=10)

        # Verify the result
        assert result is not None

        # Verify the mock was called
        mock_pool.fetchrow.assert_called_once_with("SELECT * FROM test WHERE id = $1", 1, timeout=10)

    @pytest.mark.asyncio
    async def test_fetchrow_none(self, database):
        """Test fetchrow method with None result."""
        # Set the pool
        mock_pool = AsyncMock()
        mock_pool.fetchrow.return_value = None
        database.pool = mock_pool

        # Call the method
        result = await database.fetchrow("SELECT * FROM test WHERE id = $1", 1)

        # Verify the result
        assert result is None

        # Verify the mock was called
        mock_pool.fetchrow.assert_called_once_with("SELECT * FROM test WHERE id = $1", 1, timeout=None)

    @pytest.mark.asyncio
    async def test_fetchval(self, database):
        """Test fetchval method."""
        # Set the pool
        mock_pool = AsyncMock()
        mock_pool.fetchval.return_value = 1
        database.pool = mock_pool

        # Call the method
        result = await database.fetchval("SELECT count(*) FROM test", timeout=10)

        # Verify the result
        assert result == 1

        # Verify the mock was called
        mock_pool.fetchval.assert_called_once_with("SELECT count(*) FROM test", timeout=10)

    @pytest.mark.asyncio
    async def test_transaction(self, database):
        """Test transaction method."""
        # Set the pool
        mock_pool = AsyncMock()
        mock_transaction = AsyncMock()
        mock_pool.transaction = AsyncMock(return_value=mock_transaction)
        database.pool = mock_pool

        # Call the method
        result = await database.transaction()

        # Verify the result
        assert result is not None

        # Verify the mock was called
        mock_pool.transaction.assert_called_once()
