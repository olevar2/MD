"""
Tests for the database monitoring module.
"""
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from common_lib.database.monitoring import (
    track_query_performance,
    async_track_query_performance,
    track_transaction,
    async_track_transaction,
    analyze_query,
    analyze_query_async,
    check_database_health,
    check_database_health_async,
)


class TestDatabaseMonitoring(unittest.TestCase):
    """Tests for the database monitoring functions."""
    
    def test_track_query_performance(self):
        """Test track_query_performance decorator."""
        # Create a decorated function
        @track_query_performance("select", "test_table", "test_service")
        def test_func():
            return "test_result"
        
        # Call the function
        result = test_func()
        
        # Check that the result is returned
        self.assertEqual(result, "test_result")
    
    def test_track_query_performance_with_error(self):
        """Test track_query_performance decorator with error."""
        # Create a decorated function that raises an exception
        @track_query_performance("select", "test_table", "test_service")
        def test_func():
            raise ValueError("test_error")
        
        # Call the function and check that the exception is raised
        with self.assertRaises(ValueError):
            test_func()
    
    @pytest.mark.asyncio
    async def test_async_track_query_performance(self):
        """Test async_track_query_performance decorator."""
        # Create a decorated function
        @async_track_query_performance("select", "test_table", "test_service")
        async def test_func():
            return "test_result"
        
        # Call the function
        result = await test_func()
        
        # Check that the result is returned
        self.assertEqual(result, "test_result")
    
    @pytest.mark.asyncio
    async def test_async_track_query_performance_with_error(self):
        """Test async_track_query_performance decorator with error."""
        # Create a decorated function that raises an exception
        @async_track_query_performance("select", "test_table", "test_service")
        async def test_func():
            raise ValueError("test_error")
        
        # Call the function and check that the exception is raised
        with self.assertRaises(ValueError):
            await test_func()
    
    def test_track_transaction(self):
        """Test track_transaction context manager."""
        # Use the context manager
        with track_transaction("test_service"):
            # Do something
            pass
    
    def test_track_transaction_with_error(self):
        """Test track_transaction context manager with error."""
        # Use the context manager with an error
        with self.assertRaises(ValueError):
            with track_transaction("test_service"):
                raise ValueError("test_error")
    
    @pytest.mark.asyncio
    async def test_async_track_transaction(self):
        """Test async_track_transaction context manager."""
        # Use the context manager
        async with async_track_transaction("test_service"):
            # Do something
            pass
    
    @pytest.mark.asyncio
    async def test_async_track_transaction_with_error(self):
        """Test async_track_transaction context manager with error."""
        # Use the context manager with an error
        with self.assertRaises(ValueError):
            async with async_track_transaction("test_service"):
                raise ValueError("test_error")
    
    @patch("common_lib.database.monitoring.text")
    def test_analyze_query(self, mock_text):
        """Test analyze_query function."""
        # Create mock session and result
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_session.execute.return_value = mock_result
        
        # Mock result to return rows
        mock_row1 = MagicMock()
        mock_row2 = MagicMock()
        mock_result.__iter__.return_value = [mock_row1, mock_row2]
        
        # Mock row values
        mock_row1.__getitem__.return_value = "plan line 1"
        mock_row2.__getitem__.return_value = "plan line 2"
        
        # Analyze a query
        result = analyze_query(
            mock_session,
            "SELECT * FROM test",
            {"param": "value"},
            "test_service",
        )
        
        # Check that text was called with EXPLAIN ANALYZE
        mock_text.assert_called_once_with("EXPLAIN ANALYZE SELECT * FROM test")
        
        # Check that session.execute was called
        mock_session.execute.assert_called_once()
        
        # Check that the result contains the plan
        self.assertEqual(result["service"], "test_service")
        self.assertEqual(result["query"], "EXPLAIN ANALYZE SELECT * FROM test")
        self.assertEqual(result["plan"], ["plan line 1", "plan line 2"])
    
    @pytest.mark.asyncio
    @patch("common_lib.database.monitoring.text")
    async def test_analyze_query_async(self, mock_text):
        """Test analyze_query_async function."""
        # Create mock session and result
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_session.execute.return_value = mock_result
        
        # Mock result to return rows
        mock_row1 = MagicMock()
        mock_row2 = MagicMock()
        mock_result.__iter__.return_value = [mock_row1, mock_row2]
        
        # Mock row values
        mock_row1.__getitem__.return_value = "plan line 1"
        mock_row2.__getitem__.return_value = "plan line 2"
        
        # Analyze a query
        result = await analyze_query_async(
            mock_session,
            "SELECT * FROM test",
            {"param": "value"},
            "test_service",
        )
        
        # Check that text was called with EXPLAIN ANALYZE
        mock_text.assert_called_once_with("EXPLAIN ANALYZE SELECT * FROM test")
        
        # Check that session.execute was called
        mock_session.execute.assert_called_once()
        
        # Check that the result contains the plan
        self.assertEqual(result["service"], "test_service")
        self.assertEqual(result["query"], "EXPLAIN ANALYZE SELECT * FROM test")
        self.assertEqual(result["plan"], ["plan line 1", "plan line 2"])
    
    @patch("common_lib.database.monitoring.text")
    def test_check_database_health(self, mock_text):
        """Test check_database_health function."""
        # Create mock session and results
        mock_session = MagicMock()
        mock_session.execute.side_effect = [
            MagicMock(scalar=lambda: 1),  # SELECT 1
            MagicMock(scalar=lambda: "PostgreSQL 13.3"),  # version()
            MagicMock(scalar=lambda: 10),  # connection_count
            MagicMock(scalar=lambda: 5),  # active_query_count
            MagicMock(scalar=lambda: 2),  # long_running_query_count
            MagicMock(scalar=lambda: "100 MB"),  # database_size
            MagicMock(scalar=lambda: 20),  # table_count
        ]
        
        # Check database health
        result = check_database_health(
            mock_session,
            "test_service",
        )
        
        # Check that session.execute was called multiple times
        self.assertEqual(mock_session.execute.call_count, 7)
        
        # Check that the result contains health information
        self.assertEqual(result["service"], "test_service")
        self.assertEqual(result["responsive"], True)
        self.assertEqual(result["version"], "PostgreSQL 13.3")
        self.assertEqual(result["connection_count"], 10)
        self.assertEqual(result["active_query_count"], 5)
        self.assertEqual(result["long_running_query_count"], 2)
        self.assertEqual(result["database_size"], "100 MB")
        self.assertEqual(result["table_count"], 20)
    
    @patch("common_lib.database.monitoring.text")
    def test_check_database_health_with_error(self, mock_text):
        """Test check_database_health function with error."""
        # Create mock session that raises an exception
        mock_session = MagicMock()
        mock_session.execute.side_effect = ValueError("test_error")
        
        # Check database health
        result = check_database_health(
            mock_session,
            "test_service",
        )
        
        # Check that the result indicates an error
        self.assertEqual(result["service"], "test_service")
        self.assertEqual(result["responsive"], False)
        self.assertEqual(result["error"], "test_error")
    
    @pytest.mark.asyncio
    @patch("common_lib.database.monitoring.text")
    async def test_check_database_health_async(self, mock_text):
        """Test check_database_health_async function."""
        # Create mock session and results
        mock_session = AsyncMock()
        mock_session.execute.side_effect = [
            MagicMock(scalar=lambda: 1),  # SELECT 1
            MagicMock(scalar=lambda: "PostgreSQL 13.3"),  # version()
            MagicMock(scalar=lambda: 10),  # connection_count
            MagicMock(scalar=lambda: 5),  # active_query_count
            MagicMock(scalar=lambda: 2),  # long_running_query_count
            MagicMock(scalar=lambda: "100 MB"),  # database_size
            MagicMock(scalar=lambda: 20),  # table_count
        ]
        
        # Check database health
        result = await check_database_health_async(
            mock_session,
            "test_service",
        )
        
        # Check that session.execute was called multiple times
        self.assertEqual(mock_session.execute.call_count, 7)
        
        # Check that the result contains health information
        self.assertEqual(result["service"], "test_service")
        self.assertEqual(result["responsive"], True)
        self.assertEqual(result["version"], "PostgreSQL 13.3")
        self.assertEqual(result["connection_count"], 10)
        self.assertEqual(result["active_query_count"], 5)
        self.assertEqual(result["long_running_query_count"], 2)
        self.assertEqual(result["database_size"], "100 MB")
        self.assertEqual(result["table_count"], 20)
    
    @pytest.mark.asyncio
    @patch("common_lib.database.monitoring.text")
    async def test_check_database_health_async_with_error(self, mock_text):
        """Test check_database_health_async function with error."""
        # Create mock session that raises an exception
        mock_session = AsyncMock()
        mock_session.execute.side_effect = ValueError("test_error")
        
        # Check database health
        result = await check_database_health_async(
            mock_session,
            "test_service",
        )
        
        # Check that the result indicates an error
        self.assertEqual(result["service"], "test_service")
        self.assertEqual(result["responsive"], False)
        self.assertEqual(result["error"], "test_error")


if __name__ == "__main__":
    unittest.main()