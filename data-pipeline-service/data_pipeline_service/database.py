"""
Database Module

This module provides database connectivity for the Data Pipeline Service.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List

import asyncpg
from asyncpg.pool import Pool

from common_lib.config import DatabaseConfig
from data_pipeline_service.config import get_database_config


class Database:
    """
    Database connectivity for the service.
    
    This class provides database connectivity and operations for the service.
    """
    
    def __init__(
        self,
        config: Optional[DatabaseConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the database.
        
        Args:
            config: Database configuration (if None, uses the configuration from the config manager)
            logger: Logger instance
        """
        self.config = config or get_database_config()
        self.logger = logger or logging.getLogger(__name__)
        self.pool = None
    
    async def connect(self) -> None:
        """
        Connect to the database.
        
        Raises:
            Exception: If the connection fails
        """
        if self.pool is not None:
            return
        
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                user=self.config.username,
                password=self.config.password,
                database=self.config.database,
                min_size=self.config.min_connections or 1,
                max_size=self.config.max_connections or 10
            )
            
            self.logger.info(f"Connected to database {self.config.database} at {self.config.host}:{self.config.port}")
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {str(e)}")
            raise
    
    async def close(self) -> None:
        """Close the database connection."""
        if self.pool is not None:
            await self.pool.close()
            self.pool = None
            self.logger.info("Closed database connection")
    
    async def execute(self, query: str, *args, timeout: Optional[float] = None) -> str:
        """
        Execute a query.
        
        Args:
            query: SQL query
            *args: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            Query result
            
        Raises:
            Exception: If the query fails
        """
        if self.pool is None:
            await self.connect()
        
        try:
            return await self.pool.execute(query, *args, timeout=timeout)
        except Exception as e:
            self.logger.error(f"Failed to execute query: {str(e)}")
            raise
    
    async def fetch(self, query: str, *args, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Fetch rows from the database.
        
        Args:
            query: SQL query
            *args: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            Query results
            
        Raises:
            Exception: If the query fails
        """
        if self.pool is None:
            await self.connect()
        
        try:
            rows = await self.pool.fetch(query, *args, timeout=timeout)
            return [dict(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Failed to fetch rows: {str(e)}")
            raise
    
    async def fetchrow(self, query: str, *args, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch a single row from the database.
        
        Args:
            query: SQL query
            *args: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            Query result or None if no rows are returned
            
        Raises:
            Exception: If the query fails
        """
        if self.pool is None:
            await self.connect()
        
        try:
            row = await self.pool.fetchrow(query, *args, timeout=timeout)
            return dict(row) if row else None
        except Exception as e:
            self.logger.error(f"Failed to fetch row: {str(e)}")
            raise
    
    async def fetchval(self, query: str, *args, timeout: Optional[float] = None) -> Any:
        """
        Fetch a single value from the database.
        
        Args:
            query: SQL query
            *args: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            Query result or None if no rows are returned
            
        Raises:
            Exception: If the query fails
        """
        if self.pool is None:
            await self.connect()
        
        try:
            return await self.pool.fetchval(query, *args, timeout=timeout)
        except Exception as e:
            self.logger.error(f"Failed to fetch value: {str(e)}")
            raise
    
    async def transaction(self):
        """
        Start a transaction.
        
        Returns:
            Transaction object
            
        Raises:
            Exception: If the transaction fails to start
        """
        if self.pool is None:
            await self.connect()
        
        try:
            return self.pool.transaction()
        except Exception as e:
            self.logger.error(f"Failed to start transaction: {str(e)}")
            raise


# Create a singleton instance
database = Database()
