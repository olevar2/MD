"""
Database Connection Manager Module

This module provides a connection manager for database connections.
"""

import logging
import time
from typing import Dict, Any, Optional, Union, ClassVar
from contextlib import contextmanager

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError

from common_lib.config.config_manager import ConfigManager


class DatabaseConnectionManager:
    """
    Database connection manager.
    
    This class provides a connection manager for database connections.
    """
    
    _instance: ClassVar[Optional["DatabaseConnectionManager"]] = None
    _engines: Dict[str, Engine] = {}
    _session_factories: Dict[str, sessionmaker] = {}
    
    def __new__(cls, *args, **kwargs):
        """
        Create a new instance of the database connection manager.
        
        Returns:
            Singleton instance of the database connection manager
        """
        if cls._instance is None:
            cls._instance = super(DatabaseConnectionManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the database connection manager.
        
        Args:
            config_manager: Configuration manager
            logger: Logger to use (if None, creates a new logger)
        """
        # Skip initialization if already initialized
        if getattr(self, "_initialized", False):
            return
        
        self.logger = logger or logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config_manager = config_manager or ConfigManager()
        
        self._engines = {}
        self._session_factories = {}
        
        self._initialized = True
    
    def create_engine(
        self,
        connection_name: str = "default",
        **kwargs
    ) -> Engine:
        """
        Create a database engine.
        
        Args:
            connection_name: Name of the connection
            **kwargs: Additional arguments for create_engine
            
        Returns:
            Database engine
        """
        # Check if engine already exists
        if connection_name in self._engines:
            return self._engines[connection_name]
        
        # Get database configuration
        db_config = self.config_manager.get_database_config()
        
        # Create connection URL
        connection_url = (
            f"postgresql://{db_config.username}:{db_config.password}"
            f"@{db_config.host}:{db_config.port}/{db_config.database}"
        )
        
        # Create engine
        engine = create_engine(
            connection_url,
            poolclass=QueuePool,
            pool_size=db_config.pool_size,
            max_overflow=db_config.max_overflow,
            pool_timeout=db_config.pool_timeout,
            pool_recycle=db_config.pool_recycle,
            echo=db_config.echo,
            **kwargs
        )
        
        # Store engine
        self._engines[connection_name] = engine
        
        return engine
    
    def create_session_factory(
        self,
        connection_name: str = "default",
        **kwargs
    ) -> sessionmaker:
        """
        Create a session factory.
        
        Args:
            connection_name: Name of the connection
            **kwargs: Additional arguments for sessionmaker
            
        Returns:
            Session factory
        """
        # Check if session factory already exists
        if connection_name in self._session_factories:
            return self._session_factories[connection_name]
        
        # Get engine
        engine = self.get_engine(connection_name)
        
        # Create session factory
        session_factory = sessionmaker(bind=engine, **kwargs)
        
        # Store session factory
        self._session_factories[connection_name] = session_factory
        
        return session_factory
    
    def get_engine(self, connection_name: str = "default") -> Engine:
        """
        Get a database engine.
        
        Args:
            connection_name: Name of the connection
            
        Returns:
            Database engine
            
        Raises:
            KeyError: If engine is not found
        """
        # Check if engine exists
        if connection_name not in self._engines:
            # Create engine
            self.create_engine(connection_name)
        
        return self._engines[connection_name]
    
    def get_session_factory(self, connection_name: str = "default") -> sessionmaker:
        """
        Get a session factory.
        
        Args:
            connection_name: Name of the connection
            
        Returns:
            Session factory
            
        Raises:
            KeyError: If session factory is not found
        """
        # Check if session factory exists
        if connection_name not in self._session_factories:
            # Create session factory
            self.create_session_factory(connection_name)
        
        return self._session_factories[connection_name]
    
    def get_session(self, connection_name: str = "default") -> Session:
        """
        Get a database session.
        
        Args:
            connection_name: Name of the connection
            
        Returns:
            Database session
        """
        # Get session factory
        session_factory = self.get_session_factory(connection_name)
        
        # Create session
        return session_factory()
    
    @contextmanager
    def session_scope(self, connection_name: str = "default"):
        """
        Context manager for database sessions.
        
        Args:
            connection_name: Name of the connection
            
        Yields:
            Database session
        """
        # Get session
        session = self.get_session(connection_name)
        
        try:
            # Yield session
            yield session
            
            # Commit transaction
            session.commit()
        except Exception as e:
            # Rollback transaction
            session.rollback()
            
            # Log error
            self.logger.error(f"Error in database session: {str(e)}")
            
            # Re-raise exception
            raise
        finally:
            # Close session
            session.close()
    
    def execute_with_retry(
        self,
        query: Union[str, sqlalchemy.sql.expression.ClauseElement],
        params: Optional[Dict[str, Any]] = None,
        connection_name: str = "default",
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Any:
        """
        Execute a query with retry.
        
        Args:
            query: Query to execute
            params: Query parameters
            connection_name: Name of the connection
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
            
        Returns:
            Query result
            
        Raises:
            SQLAlchemyError: If query execution fails after all retries
        """
        # Get engine
        engine = self.get_engine(connection_name)
        
        # Initialize retry counter
        retry_count = 0
        
        # Execute query with retry
        while True:
            try:
                # Execute query
                with engine.connect() as connection:
                    if params is not None:
                        result = connection.execute(query, params)
                    else:
                        result = connection.execute(query)
                    
                    # Return result
                    return result
            except SQLAlchemyError as e:
                # Increment retry counter
                retry_count += 1
                
                # Check if maximum retries reached
                if retry_count > max_retries:
                    # Log error
                    self.logger.error(
                        f"Query execution failed after {max_retries} retries: {str(e)}"
                    )
                    
                    # Re-raise exception
                    raise
                
                # Log retry
                self.logger.warning(
                    f"Query execution failed, retrying ({retry_count}/{max_retries}): {str(e)}"
                )
                
                # Wait before retrying
                time.sleep(retry_delay)
    
    def close_all_connections(self):
        """
        Close all database connections.
        """
        # Close all engines
        for connection_name, engine in self._engines.items():
            self.logger.info(f"Closing database engine: {connection_name}")
            engine.dispose()
        
        # Clear engines and session factories
        self._engines = {}
        self._session_factories = {}