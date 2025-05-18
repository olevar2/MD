"""
Tests for the bulk operations module.
"""
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio

import pytest
from sqlalchemy import Table, Column, Integer, String, MetaData
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
import asyncpg

from common_lib.database.bulk_operations import (
    bulk_insert,
    bulk_insert_async,
    bulk_update,
    bulk_update_async,
    bulk_delete,
    bulk_delete_async,
    bulk_insert_asyncpg,
)


class TestBulkOperations(unittest.TestCase):
    """Tests for the bulk operations functions."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a test table
        self.metadata = MetaData()
        self.test_table = Table(
            "test_table",
            self.metadata,
            Column("id", Integer, primary_key=True),
            Column("name", String),
            Column("value", Integer),
        )
    
    @patch("common_lib.database.bulk_operations.insert")
    def test_bulk_insert(self, mock_insert):
        """Test bulk_insert function."""
        # Create mock session and result
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_session.execute.return_value = mock_result
        
        # Create test data
        test_data = [
            {"id": 1, "name": "test1", "value": 100},
            {"id": 2, "name": "test2", "value": 200},
        ]
        
        # Perform bulk insert
        result = bulk_insert(
            mock_session,
            self.test_table,
            test_data,
            "test_service",
            return_defaults=False,
        )
        
        # Check that insert was called
        mock_insert.assert_called_once()
        
        # Check that session.execute was called
        mock_session.execute.assert_called_once()
        
        # Check that session.commit was called
        mock_session.commit.assert_called_once()
        
        # Check that an empty list is returned (return_defaults=False)
        self.assertEqual(result, [])
    
    @patch("common_lib.database.bulk_operations.insert")
    def test_bulk_insert_with_return_defaults(self, mock_insert):
        """Test bulk_insert function with return_defaults=True."""
        # Create mock session and result
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_session.execute.return_value = mock_result
        
        # Mock result to return rows
        mock_row1 = MagicMock()
        mock_row2 = MagicMock()
        mock_result.__iter__.return_value = [mock_row1, mock_row2]
        
        # Mock dict conversion
        mock_row1.__dict__ = {"id": 1, "name": "test1", "value": 100}
        mock_row2.__dict__ = {"id": 2, "name": "test2", "value": 200}
        
        # Create test data
        test_data = [
            {"id": 1, "name": "test1", "value": 100},
            {"id": 2, "name": "test2", "value": 200},
        ]
        
        # Perform bulk insert
        result = bulk_insert(
            mock_session,
            self.test_table,
            test_data,
            "test_service",
            return_defaults=True,
        )
        
        # Check that insert was called
        mock_insert.assert_called_once()
        
        # Check that session.execute was called
        mock_session.execute.assert_called_once()
        
        # Check that session.commit was called
        mock_session.commit.assert_called_once()
        
        # Check that the result contains the rows
        self.assertEqual(len(result), 2)
    
    @pytest.mark.asyncio
    @patch("common_lib.database.bulk_operations.insert")
    async def test_bulk_insert_async(self, mock_insert):
        """Test bulk_insert_async function."""
        # Create mock session and result
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_session.execute.return_value = mock_result
        
        # Create test data
        test_data = [
            {"id": 1, "name": "test1", "value": 100},
            {"id": 2, "name": "test2", "value": 200},
        ]
        
        # Perform bulk insert
        result = await bulk_insert_async(
            mock_session,
            self.test_table,
            test_data,
            "test_service",
            return_defaults=False,
        )
        
        # Check that insert was called
        mock_insert.assert_called_once()
        
        # Check that session.execute was called
        mock_session.execute.assert_called_once()
        
        # Check that session.commit was called
        mock_session.commit.assert_called_once()
        
        # Check that an empty list is returned (return_defaults=False)
        self.assertEqual(result, [])
    
    @patch("common_lib.database.bulk_operations.update")
    @patch("common_lib.database.bulk_operations.text")
    def test_bulk_update(self, mock_text, mock_update):
        """Test bulk_update function."""
        # Create mock session and result
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_session.execute.return_value = mock_result
        mock_result.rowcount = 2
        
        # Create test data
        test_data = [
            {"id": 1, "name": "test1", "value": 100},
            {"id": 2, "name": "test2", "value": 200},
        ]
        
        # Perform bulk update
        result = bulk_update(
            mock_session,
            self.test_table,
            test_data,
            "id",
            "test_service",
        )
        
        # Check that update was called
        mock_update.assert_called_once()
        
        # Check that session.execute was called
        mock_session.execute.assert_called_once()
        
        # Check that session.commit was called
        mock_session.commit.assert_called_once()
        
        # Check that the number of updated rows is returned
        self.assertEqual(result, 2)
    
    @pytest.mark.asyncio
    @patch("common_lib.database.bulk_operations.update")
    @patch("common_lib.database.bulk_operations.text")
    async def test_bulk_update_async(self, mock_text, mock_update):
        """Test bulk_update_async function."""
        # Create mock session and result
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_session.execute.return_value = mock_result
        mock_result.rowcount = 2
        
        # Create test data
        test_data = [
            {"id": 1, "name": "test1", "value": 100},
            {"id": 2, "name": "test2", "value": 200},
        ]
        
        # Perform bulk update
        result = await bulk_update_async(
            mock_session,
            self.test_table,
            test_data,
            "id",
            "test_service",
        )
        
        # Check that update was called
        mock_update.assert_called_once()
        
        # Check that session.execute was called
        mock_session.execute.assert_called_once()
        
        # Check that session.commit was called
        mock_session.commit.assert_called_once()
        
        # Check that the number of updated rows is returned
        self.assertEqual(result, 2)
    
    @patch("common_lib.database.bulk_operations.delete")
    def test_bulk_delete(self, mock_delete):
        """Test bulk_delete function."""
        # Create mock session and result
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_session.execute.return_value = mock_result
        mock_result.rowcount = 2
        
        # Create test data
        test_ids = [1, 2]
        
        # Perform bulk delete
        result = bulk_delete(
            mock_session,
            self.test_table,
            test_ids,
            "id",
            "test_service",
        )
        
        # Check that delete was called
        mock_delete.assert_called_once()
        
        # Check that session.execute was called
        mock_session.execute.assert_called_once()
        
        # Check that session.commit was called
        mock_session.commit.assert_called_once()
        
        # Check that the number of deleted rows is returned
        self.assertEqual(result, 2)
    
    @pytest.mark.asyncio
    @patch("common_lib.database.bulk_operations.delete")
    async def test_bulk_delete_async(self, mock_delete):
        """Test bulk_delete_async function."""
        # Create mock session and result
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_session.execute.return_value = mock_result
        mock_result.rowcount = 2
        
        # Create test data
        test_ids = [1, 2]
        
        # Perform bulk delete
        result = await bulk_delete_async(
            mock_session,
            self.test_table,
            test_ids,
            "id",
            "test_service",
        )
        
        # Check that delete was called
        mock_delete.assert_called_once()
        
        # Check that session.execute was called
        mock_session.execute.assert_called_once()
        
        # Check that session.commit was called
        mock_session.commit.assert_called_once()
        
        # Check that the number of deleted rows is returned
        self.assertEqual(result, 2)
    
    @pytest.mark.asyncio
    async def test_bulk_insert_asyncpg(self):
        """Test bulk_insert_asyncpg function."""
        # Create mock connection
        mock_conn = AsyncMock()
        
        # Create test data
        test_columns = ["id", "name", "value"]
        test_values = [
            [1, "test1", 100],
            [2, "test2", 200],
        ]
        
        # Perform bulk insert
        result = await bulk_insert_asyncpg(
            mock_conn,
            "test_table",
            test_columns,
            test_values,
            "test_service",
            return_ids=False,
        )
        
        # Check that connection.executemany was called
        mock_conn.executemany.assert_called_once()
        
        # Check that an empty list is returned (return_ids=False)
        self.assertEqual(result, [])
    
    @pytest.mark.asyncio
    async def test_bulk_insert_asyncpg_with_return_ids(self):
        """Test bulk_insert_asyncpg function with return_ids=True."""
        # Create mock connection
        mock_conn = AsyncMock()
        
        # Mock fetchval to return IDs
        mock_conn.fetchval.side_effect = [1, 2]
        
        # Create test data
        test_columns = ["id", "name", "value"]
        test_values = [
            [1, "test1", 100],
            [2, "test2", 200],
        ]
        
        # Perform bulk insert
        result = await bulk_insert_asyncpg(
            mock_conn,
            "test_table",
            test_columns,
            test_values,
            "test_service",
            return_ids=True,
        )
        
        # Check that connection.fetchval was called twice
        self.assertEqual(mock_conn.fetchval.call_count, 2)
        
        # Check that the result contains the IDs
        self.assertEqual(result, [1, 2])
    
    @pytest.mark.asyncio
    async def test_bulk_insert_asyncpg_with_copy(self):
        """Test bulk_insert_asyncpg function with copy_records_to_table."""
        # Create mock connection
        mock_conn = AsyncMock()
        
        # Create test data with more than 100 rows to trigger copy_records_to_table
        test_columns = ["id", "name", "value"]
        test_values = [[i, f"test{i}", i * 100] for i in range(101)]
        
        # Perform bulk insert
        result = await bulk_insert_asyncpg(
            mock_conn,
            "test_table",
            test_columns,
            test_values,
            "test_service",
            return_ids=False,
        )
        
        # Check that connection.copy_records_to_table was called
        mock_conn.copy_records_to_table.assert_called_once()
        
        # Check that an empty list is returned (return_ids=False)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()