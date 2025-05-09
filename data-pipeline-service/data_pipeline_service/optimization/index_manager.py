"""
Database Index Manager Module.

This module provides utilities for managing database indexes,
particularly for time series data in TimescaleDB.
"""

import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text

logger = logging.getLogger(__name__)


class IndexManager:
    """
    Manages database indexes for optimized queries.
    
    This class provides methods to create, validate, and manage
    database indexes for optimized query performance.
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize the index manager.
        
        Args:
            session: SQLAlchemy async session
        """
        self.session = session
        
        # Define required indexes for time series tables
        self.required_indexes = {
            "ohlcv": [
                {"name": "time_idx", "columns": ["timestamp"]},
                {"name": "symbol_idx", "columns": ["symbol"]},
                {"name": "symbol_time_idx", "columns": ["symbol", "timestamp"]}
            ],
            "ohlcv_1m": [
                {"name": "time_idx", "columns": ["timestamp"]},
                {"name": "symbol_idx", "columns": ["symbol"]},
                {"name": "symbol_time_idx", "columns": ["symbol", "timestamp"]}
            ],
            "ohlcv_5m": [
                {"name": "time_idx", "columns": ["timestamp"]},
                {"name": "symbol_idx", "columns": ["symbol"]},
                {"name": "symbol_time_idx", "columns": ["symbol", "timestamp"]}
            ],
            "ohlcv_15m": [
                {"name": "time_idx", "columns": ["timestamp"]},
                {"name": "symbol_idx", "columns": ["symbol"]},
                {"name": "symbol_time_idx", "columns": ["symbol", "timestamp"]}
            ],
            "ohlcv_30m": [
                {"name": "time_idx", "columns": ["timestamp"]},
                {"name": "symbol_idx", "columns": ["symbol"]},
                {"name": "symbol_time_idx", "columns": ["symbol", "timestamp"]}
            ],
            "ohlcv_1h": [
                {"name": "time_idx", "columns": ["timestamp"]},
                {"name": "symbol_idx", "columns": ["symbol"]},
                {"name": "symbol_time_idx", "columns": ["symbol", "timestamp"]}
            ],
            "ohlcv_4h": [
                {"name": "time_idx", "columns": ["timestamp"]},
                {"name": "symbol_idx", "columns": ["symbol"]},
                {"name": "symbol_time_idx", "columns": ["symbol", "timestamp"]}
            ],
            "ohlcv_1d": [
                {"name": "time_idx", "columns": ["timestamp"]},
                {"name": "symbol_idx", "columns": ["symbol"]},
                {"name": "symbol_time_idx", "columns": ["symbol", "timestamp"]}
            ],
            "ohlcv_1w": [
                {"name": "time_idx", "columns": ["timestamp"]},
                {"name": "symbol_idx", "columns": ["symbol"]},
                {"name": "symbol_time_idx", "columns": ["symbol", "timestamp"]}
            ],
            "tick_data": [
                {"name": "time_idx", "columns": ["timestamp"]},
                {"name": "symbol_idx", "columns": ["symbol"]},
                {"name": "symbol_time_idx", "columns": ["symbol", "timestamp"]}
            ]
        }
        
        # Track existing indexes
        self.existing_indexes: Dict[str, Set[str]] = {}
    
    async def ensure_indexes(self, table_name: str) -> None:
        """
        Ensure that all required indexes exist for a table.
        
        Args:
            table_name: Name of the table to check
        """
        if table_name not in self.required_indexes:
            logger.debug(f"No index requirements defined for table {table_name}")
            return
            
        # Get existing indexes if we haven't already
        if table_name not in self.existing_indexes:
            await self._get_existing_indexes(table_name)
            
        # Check and create missing indexes
        for index_def in self.required_indexes[table_name]:
            index_name = index_def["name"]
            
            if index_name not in self.existing_indexes.get(table_name, set()):
                await self._create_index(table_name, index_def)
    
    async def _get_existing_indexes(self, table_name: str) -> None:
        """
        Get existing indexes for a table.
        
        Args:
            table_name: Name of the table to check
        """
        try:
            # Query to get existing indexes
            query = """
            SELECT
                i.relname as index_name
            FROM
                pg_class t,
                pg_class i,
                pg_index ix,
                pg_attribute a
            WHERE
                t.relname = :table_name AND
                t.oid = ix.indrelid AND
                i.oid = ix.indexrelid AND
                a.attrelid = t.oid AND
                a.attnum = ANY(ix.indkey)
            GROUP BY
                i.relname
            """
            
            result = await self.session.execute(text(query), {"table_name": table_name})
            rows = result.fetchall()
            
            # Store existing indexes
            self.existing_indexes[table_name] = {row[0] for row in rows}
            
            logger.debug(f"Found existing indexes for {table_name}: {self.existing_indexes[table_name]}")
            
        except Exception as e:
            logger.error(f"Error getting existing indexes for {table_name}: {str(e)}")
            # Initialize with empty set if error
            self.existing_indexes[table_name] = set()
    
    async def _create_index(self, table_name: str, index_def: Dict[str, Any]) -> None:
        """
        Create an index on a table.
        
        Args:
            table_name: Name of the table
            index_def: Index definition
        """
        index_name = index_def["name"]
        columns = index_def["columns"]
        
        try:
            # Build CREATE INDEX statement
            column_list = ", ".join(columns)
            
            # Check if this is a time series table (use TimescaleDB optimized index)
            if "timestamp" in columns:
                # Use TimescaleDB's CREATE INDEX WITH optimization for time columns
                query = f"""
                CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({column_list})
                WITH (timescaledb.transaction_per_chunk = true)
                """
            else:
                # Regular index
                query = f"""
                CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({column_list})
                """
            
            # Execute the query
            await self.session.execute(text(query))
            await self.session.commit()
            
            # Update our tracking
            if table_name in self.existing_indexes:
                self.existing_indexes[table_name].add(index_name)
            else:
                self.existing_indexes[table_name] = {index_name}
                
            logger.info(f"Created index {index_name} on {table_name}({column_list})")
            
        except Exception as e:
            logger.error(f"Error creating index {index_name} on {table_name}: {str(e)}")
            await self.session.rollback()
    
    async def analyze_table(self, table_name: str) -> None:
        """
        Run ANALYZE on a table to update statistics.
        
        Args:
            table_name: Name of the table to analyze
        """
        try:
            query = f"ANALYZE {table_name}"
            await self.session.execute(text(query))
            logger.info(f"Analyzed table {table_name}")
        except Exception as e:
            logger.error(f"Error analyzing table {table_name}: {str(e)}")


# Factory function to create index manager
async def get_index_manager(session: AsyncSession) -> IndexManager:
    """
    Get an index manager instance.
    
    Args:
        session: SQLAlchemy async session
        
    Returns:
        IndexManager instance
    """
    return IndexManager(session)
