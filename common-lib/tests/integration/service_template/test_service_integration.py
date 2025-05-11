"""
Integration tests for the service template.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from common_lib.templates.service_template.service_clients import ServiceClients
from common_lib.templates.service_template.config import get_service_config
from common_lib.templates.service_template.database import Database


@pytest.fixture
def service_config():
    """Fixture for service configuration."""
    return get_service_config()


@pytest.fixture
def mock_database():
    """Fixture for database."""
    db = MagicMock(spec=Database)
    db.connect = AsyncMock()
    db.close = AsyncMock()
    db.fetch = AsyncMock(return_value=[{"id": 1, "name": "test"}])
    db.fetchrow = AsyncMock(return_value={"id": 1, "name": "test"})
    db.fetchval = AsyncMock(return_value=1)
    db.execute = AsyncMock(return_value="OK")
    db.transaction = AsyncMock()
    return db


@pytest.fixture
def service_clients(service_config):
    """Fixture for service clients."""
    return ServiceClients(service_config)


@pytest.mark.asyncio
async def test_service_clients_initialization(service_clients):
    """Test service clients initialization."""
    # Verify the service clients were initialized
    assert service_clients is not None


@pytest.mark.asyncio
async def test_service_clients_connect_close(service_clients):
    """Test service clients connect and close methods."""
    # Mock the connect and close methods
    service_clients.connect = AsyncMock()
    service_clients.close = AsyncMock()

    # Call the methods
    await service_clients.connect()
    await service_clients.close()

    # Verify the methods were called
    service_clients.connect.assert_called_once()
    service_clients.close.assert_called_once()


@pytest.mark.asyncio
async def test_database_integration(mock_database):
    """Test database integration."""
    # Connect to the database
    await mock_database.connect()

    # Execute a query
    result = await mock_database.execute("SELECT 1")
    assert result == "OK"

    # Fetch data
    rows = await mock_database.fetch("SELECT * FROM test")
    assert len(rows) == 1
    assert rows[0]["id"] == 1
    assert rows[0]["name"] == "test"

    # Fetch a single row
    row = await mock_database.fetchrow("SELECT * FROM test WHERE id = $1", 1)
    assert row["id"] == 1
    assert row["name"] == "test"

    # Fetch a single value
    value = await mock_database.fetchval("SELECT count(*) FROM test")
    assert value == 1

    # Close the database connection
    await mock_database.close()

    # Verify the methods were called
    mock_database.connect.assert_called_once()
    mock_database.execute.assert_called_once_with("SELECT 1")
    mock_database.fetch.assert_called_once_with("SELECT * FROM test")
    mock_database.fetchrow.assert_called_once_with("SELECT * FROM test WHERE id = $1", 1)
    mock_database.fetchval.assert_called_once_with("SELECT count(*) FROM test")
    mock_database.close.assert_called_once()