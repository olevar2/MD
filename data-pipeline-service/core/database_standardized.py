"""
Standardized Database Module for Data Pipeline Service

This module provides database connectivity for the service using the standardized
database connectivity system from common-lib.
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List, Union
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
from common_lib.monitoring.tracing import trace_method
from common_lib.monitoring.metrics import track_execution_time
from config.standardized_config_1 import settings


from data_pipeline_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class Database:
    """
    Database connectivity for the service.
    
    This class provides database connectivity and operations for the service.
    """

    def __init__(self, database_url: Optional[str]=None, min_pool_size:
        Optional[int]=None, max_pool_size: Optional[int]=None, pool_recycle:
        Optional[int]=None, echo: Optional[bool]=None, logger: Optional[
        logging.Logger]=None):
        """
        Initialize the database.
        
        Args:
            database_url: Database URL (if None, uses the URL from settings)
            min_pool_size: Minimum connection pool size (if None, uses the value from settings)
            max_pool_size: Maximum connection pool size (if None, uses the value from settings)
            pool_recycle: Connection pool recycle time in seconds (if None, uses the value from settings)
            echo: Whether to echo SQL statements (if None, uses the value from settings)
            logger: Logger instance
        """
        self.database_url = database_url or settings.database_url
        self.min_pool_size = min_pool_size or settings.DB_POOL_SIZE
        self.max_pool_size = max_pool_size or settings.DB_MAX_OVERFLOW
        self.pool_recycle = pool_recycle or 3600
        self.echo = echo or settings.DB_ECHO
        self.logger = logger or logging.getLogger(__name__)
        self.engine: Optional[AsyncEngine] = None
        self.async_session: Optional[sessionmaker] = None

    @trace_method(name='connect')
    @track_execution_time(name='database_connect_duration_seconds',
        description='Time taken to connect to the database', labels={
        'service': 'data_pipeline_service'})
    @async_with_exception_handling
    async def connect(self) ->None:
        """
        Connect to the database.
        
        Raises:
            Exception: If the connection fails
        """
        if self.engine is not None:
            return
        try:
            self.engine = create_async_engine(self.database_url, echo=self.
                echo, pool_size=self.min_pool_size, max_overflow=self.
                max_pool_size, pool_recycle=self.pool_recycle)
            self.async_session = sessionmaker(self.engine, class_=
                AsyncSession, expire_on_commit=False)
            async with self.engine.connect() as conn:
                await conn.execute(text('SELECT 1'))
            self.logger.info(f'Connected to database at {self.database_url}')
        except Exception as e:
            self.logger.error(f'Failed to connect to database: {str(e)}')
            raise

    @trace_method(name='close')
    async def close(self) ->None:
        """Close the database connection."""
        if self.engine is not None:
            await self.engine.dispose()
            self.engine = None
            self.async_session = None
            self.logger.info('Closed database connection')

    @trace_method(name='get_session')
    async def get_session(self) ->AsyncSession:
        """
        Get a database session.
        
        Returns:
            Database session
            
        Raises:
            Exception: If the connection fails
        """
        if self.engine is None:
            await self.connect()
        if self.async_session is None:
            raise RuntimeError('Database session factory not initialized')
        return self.async_session()

    @trace_method(name='execute')
    @track_execution_time(name='database_execute_duration_seconds',
        description='Time taken to execute a database query', labels={
        'service': 'data_pipeline_service'})
    @async_with_exception_handling
    async def execute(self, query: Union[str, text], params: Optional[Dict[
        str, Any]]=None, timeout: Optional[float]=None) ->Any:
        """
        Execute a query.
        
        Args:
            query: SQL query
            params: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            Query result
            
        Raises:
            Exception: If the query fails
        """
        if isinstance(query, str):
            query = text(query)
        async with (await self.get_session()) as session:
            try:
                result = await session.execute(query, params or {},
                    execution_options={'timeout': timeout})
                await session.commit()
                return result
            except Exception as e:
                await session.rollback()
                self.logger.error(f'Failed to execute query: {str(e)}')
                raise

    @trace_method(name='fetch_all')
    @track_execution_time(name='database_fetch_all_duration_seconds',
        description='Time taken to fetch all rows from a database query',
        labels={'service': 'data_pipeline_service'})
    @async_with_exception_handling
    async def fetch_all(self, query: Union[str, text], params: Optional[
        Dict[str, Any]]=None, timeout: Optional[float]=None) ->List[Dict[
        str, Any]]:
        """
        Fetch all rows from a query.
        
        Args:
            query: SQL query
            params: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            Query results
            
        Raises:
            Exception: If the query fails
        """
        if isinstance(query, str):
            query = text(query)
        async with (await self.get_session()) as session:
            try:
                result = await session.execute(query, params or {},
                    execution_options={'timeout': timeout})
                return [dict(row._mapping) for row in result.fetchall()]
            except Exception as e:
                self.logger.error(f'Failed to fetch all rows: {str(e)}')
                raise

    @trace_method(name='fetch_one')
    @track_execution_time(name='database_fetch_one_duration_seconds',
        description='Time taken to fetch one row from a database query',
        labels={'service': 'data_pipeline_service'})
    @async_with_exception_handling
    async def fetch_one(self, query: Union[str, text], params: Optional[
        Dict[str, Any]]=None, timeout: Optional[float]=None) ->Optional[Dict
        [str, Any]]:
        """
        Fetch one row from a query.
        
        Args:
            query: SQL query
            params: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            Query result or None if no rows are returned
            
        Raises:
            Exception: If the query fails
        """
        if isinstance(query, str):
            query = text(query)
        async with (await self.get_session()) as session:
            try:
                result = await session.execute(query, params or {},
                    execution_options={'timeout': timeout})
                row = result.fetchone()
                return dict(row._mapping) if row else None
            except Exception as e:
                self.logger.error(f'Failed to fetch one row: {str(e)}')
                raise

    @trace_method(name='fetch_value')
    @track_execution_time(name='database_fetch_value_duration_seconds',
        description=
        'Time taken to fetch a single value from a database query', labels=
        {'service': 'data_pipeline_service'})
    @async_with_exception_handling
    async def fetch_value(self, query: Union[str, text], params: Optional[
        Dict[str, Any]]=None, timeout: Optional[float]=None) ->Any:
        """
        Fetch a single value from a query.
        
        Args:
            query: SQL query
            params: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            Query result or None if no rows are returned
            
        Raises:
            Exception: If the query fails
        """
        if isinstance(query, str):
            query = text(query)
        async with (await self.get_session()) as session:
            try:
                result = await session.execute(query, params or {},
                    execution_options={'timeout': timeout})
                row = result.fetchone()
                return row[0] if row else None
            except Exception as e:
                self.logger.error(f'Failed to fetch value: {str(e)}')
                raise


database = Database()
