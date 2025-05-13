"""
Query Optimizer Module.

This module provides utilities for optimizing database queries,
particularly for time series data in TimescaleDB.
"""

import logging
import re
from typing import Dict, Any, Tuple, List, Optional, Union
from datetime import datetime, timedelta

import sqlparse
from sqlparse.sql import Token, TokenList, Identifier, IdentifierList, Where, Comparison

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """
    Optimizes SQL queries for TimescaleDB.
    
    This class provides methods to analyze and optimize SQL queries,
    particularly for time series data stored in TimescaleDB.
    """
    
    def __init__(self, enable_chunk_optimization: bool = True, enable_index_hints: bool = True):
        """
        Initialize the query optimizer.
        
        Args:
            enable_chunk_optimization: Whether to enable chunk exclusion optimization
            enable_index_hints: Whether to enable index hints
        """
        self.enable_chunk_optimization = enable_chunk_optimization
        self.enable_index_hints = enable_index_hints
        
        # Known time series tables and their time columns
        self.time_series_tables = {
            "ohlcv": "timestamp",
            "ohlcv_1m": "timestamp",
            "ohlcv_5m": "timestamp",
            "ohlcv_15m": "timestamp",
            "ohlcv_30m": "timestamp",
            "ohlcv_1h": "timestamp",
            "ohlcv_4h": "timestamp",
            "ohlcv_1d": "timestamp",
            "ohlcv_1w": "timestamp",
            "tick_data": "timestamp",
            "price_data": "timestamp",
            "features.indicators": "timestamp"
        }
        
        # Known indexes for tables
        self.table_indexes = {
            "ohlcv": ["symbol_idx", "time_idx"],
            "ohlcv_1m": ["symbol_idx", "time_idx"],
            "ohlcv_5m": ["symbol_idx", "time_idx"],
            "ohlcv_15m": ["symbol_idx", "time_idx"],
            "ohlcv_30m": ["symbol_idx", "time_idx"],
            "ohlcv_1h": ["symbol_idx", "time_idx"],
            "ohlcv_4h": ["symbol_idx", "time_idx"],
            "ohlcv_1d": ["symbol_idx", "time_idx"],
            "ohlcv_1w": ["symbol_idx", "time_idx"],
            "tick_data": ["symbol_idx", "time_idx"],
            "price_data": ["symbol_idx", "time_idx"],
            "features.indicators": ["indicator_idx", "symbol_idx", "time_idx"]
        }
    
    def optimize_query(
        self, 
        query: str, 
        params: Dict[str, Any] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Optimize a SQL query.
        
        Args:
            query: SQL query to optimize
            params: Query parameters
            
        Returns:
            Tuple of (optimized query, updated parameters)
        """
        params = params or {}
        
        # Parse the query
        parsed = sqlparse.parse(query)[0]
        
        # Extract table name
        table_name = self._extract_table_name(parsed)
        
        # Check if this is a time series table
        if table_name in self.time_series_tables:
            # Apply TimescaleDB-specific optimizations
            query = self._add_timescaledb_hints(query, table_name)
            
            # Add index hints if enabled
            if self.enable_index_hints:
                query = self._add_index_hints(query, table_name, params)
        
        return query, params
    
    def _extract_table_name(self, parsed_query: TokenList) -> Optional[str]:
        """
        Extract the table name from a parsed SQL query.
        
        Args:
            parsed_query: Parsed SQL query
            
        Returns:
            Table name or None if not found
        """
        from_seen = False
        
        for token in parsed_query.tokens:
            if token.is_keyword and token.normalized == 'FROM':
                from_seen = True
                continue
                
            if from_seen:
                if isinstance(token, Identifier):
                    return token.get_real_name()
                elif isinstance(token, IdentifierList):
                    # Handle multiple tables (take the first one)
                    for identifier in token.get_identifiers():
                        return identifier.get_real_name()
                    
        return None
    
    def _add_timescaledb_hints(self, query: str, table_name: str) -> str:
        """
        Add TimescaleDB-specific hints to a query.
        
        Args:
            query: SQL query
            table_name: Table name
            
        Returns:
            Query with TimescaleDB hints
        """
        if not self.enable_chunk_optimization:
            return query
            
        # Add chunk exclusion hint
        if "/*+" not in query:
            # Add hint after SELECT
            query = re.sub(
                r'SELECT\b',
                'SELECT /*+ TimescaleDB.enable_chunk_exclusion=true */',
                query,
                count=1,
                flags=re.IGNORECASE
            )
        
        return query
    
    def _add_index_hints(
        self, 
        query: str, 
        table_name: str, 
        params: Dict[str, Any]
    ) -> str:
        """
        Add index hints to a query.
        
        Args:
            query: SQL query
            table_name: Table name
            params: Query parameters
            
        Returns:
            Query with index hints
        """
        if not self.enable_index_hints or table_name not in self.table_indexes:
            return query
            
        # Check if we have time range parameters
        has_time_range = False
        time_column = self.time_series_tables.get(table_name)
        
        if time_column:
            # Check if query has time range conditions
            has_time_range = (
                f"{time_column} >=" in query or 
                f"{time_column} >" in query or 
                f"{time_column} <=" in query or 
                f"{time_column} <" in query
            )
        
        # Check if we have symbol parameter
        has_symbol = "symbol =" in query
        
        # Add appropriate index hints
        if has_time_range and has_symbol:
            # Use composite index if available, otherwise use time index
            if "symbol_time_idx" in self.table_indexes.get(table_name, []):
                query = self._replace_table_with_index(query, table_name, "symbol_time_idx")
            elif "time_idx" in self.table_indexes.get(table_name, []):
                query = self._replace_table_with_index(query, table_name, "time_idx")
        elif has_time_range:
            # Use time index
            if "time_idx" in self.table_indexes.get(table_name, []):
                query = self._replace_table_with_index(query, table_name, "time_idx")
        elif has_symbol:
            # Use symbol index
            if "symbol_idx" in self.table_indexes.get(table_name, []):
                query = self._replace_table_with_index(query, table_name, "symbol_idx")
        
        return query
    
    def _replace_table_with_index(self, query: str, table_name: str, index_name: str) -> str:
        """
        Replace table reference with index hint.
        
        Args:
            query: SQL query
            table_name: Table name
            index_name: Index name
            
        Returns:
            Query with index hint
        """
        # Replace FROM table_name with FROM table_name USING INDEX index_name
        return re.sub(
            f"FROM\\s+{table_name}\\b",
            f"FROM {table_name} USING INDEX {index_name}",
            query,
            count=1,
            flags=re.IGNORECASE
        )


# Singleton instance
query_optimizer = QueryOptimizer()


def optimize_query(
    query: str, 
    params: Dict[str, Any] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Optimize a SQL query.
    
    Args:
        query: SQL query to optimize
        params: Query parameters
        
    Returns:
        Tuple of (optimized query, updated parameters)
    """
    return query_optimizer.optimize_query(query, params)
