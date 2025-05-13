"""
Standardized Database Module for ML Workbench Service

This module provides standardized database connectivity and ORM setup
that follows the common-lib pattern for database management.
"""
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, cast
from contextlib import asynccontextmanager
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, Boolean, Float, ForeignKey, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session, Session, relationship
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.future import select
from sqlalchemy.sql import Select
from sqlalchemy.exc import SQLAlchemyError
from common_lib.errors import DatabaseError
from ml_workbench_service.config.standardized_config import get_db_url, settings
from ml_workbench_service.logging_setup import get_logger
T = TypeVar('T')
logger = get_logger(__name__)
Base = declarative_base()
metadata = Base.metadata
engine: Optional[AsyncEngine] = None
sync_engine = None
async_session_factory = None
sync_session_factory = None


def init_db(db_url: Optional[str]=None) ->None:
    """
    Initialize database connection.

    Args:
        db_url: Database URL (defaults to the one from settings)
    """
    global engine, sync_engine, async_session_factory, sync_session_factory
    db_url = db_url or get_db_url()
    if db_url.startswith(('postgresql://', 'postgresql+asyncpg://',
        'mysql+aiomysql://')):
        if not db_url.startswith(('postgresql+asyncpg://', 'mysql+aiomysql://')
            ):
            if db_url.startswith('postgresql://'):
                db_url = db_url.replace('postgresql://',
                    'postgresql+asyncpg://')
            elif db_url.startswith('mysql://'):
                db_url = db_url.replace('mysql://', 'mysql+aiomysql://')
        engine = create_async_engine(db_url, echo=settings.DB_POOL_SIZE > 1,
            pool_size=settings.DB_POOL_SIZE, max_overflow=settings.
            DB_MAX_OVERFLOW, pool_timeout=settings.DB_POOL_TIMEOUT,
            pool_recycle=settings.DB_POOL_RECYCLE)
        async_session_factory = async_sessionmaker(engine, expire_on_commit
            =False, class_=AsyncSession)
        sync_db_url = db_url
        if 'asyncpg' in sync_db_url:
            sync_db_url = sync_db_url.replace('asyncpg', 'psycopg2')
        elif 'aiomysql' in sync_db_url:
            sync_db_url = sync_db_url.replace('aiomysql', 'pymysql')
        sync_engine = create_engine(sync_db_url, echo=settings.DB_POOL_SIZE >
            1, pool_size=settings.DB_POOL_SIZE, max_overflow=settings.
            DB_MAX_OVERFLOW, pool_timeout=settings.DB_POOL_TIMEOUT,
            pool_recycle=settings.DB_POOL_RECYCLE)
        sync_session_factory = scoped_session(sessionmaker(bind=sync_engine,
            expire_on_commit=False))
    else:
        sync_engine = create_engine(db_url, echo=settings.DB_POOL_SIZE > 1,
            pool_size=settings.DB_POOL_SIZE if db_url.startswith((
            'mysql://', 'postgresql://')) else None, max_overflow=settings.
            DB_MAX_OVERFLOW if db_url.startswith(('mysql://',
            'postgresql://')) else None, pool_timeout=settings.
            DB_POOL_TIMEOUT if db_url.startswith(('mysql://',
            'postgresql://')) else None, pool_recycle=settings.
            DB_POOL_RECYCLE if db_url.startswith(('mysql://',
            'postgresql://')) else None)
        sync_session_factory = scoped_session(sessionmaker(bind=sync_engine,
            expire_on_commit=False))
        logger.warning(
            'Using synchronous database engine. This is not recommended for production.'
            , extra={'db_url': db_url})


def get_sync_session() ->Session:
    """
    Get a synchronous database session.

    Returns:
        Database session
    """
    if sync_session_factory is None:
        init_db()
    return sync_session_factory()


@asynccontextmanager
@async_with_exception_handling
async def get_async_session() ->AsyncSession:
    """
    Get an asynchronous database session.

    Yields:
        Database session
    """
    if async_session_factory is None:
        init_db()
    if async_session_factory is None:
        raise DatabaseError(message=
            'Async database session factory is not available', service_name
            ='ml-workbench-service', operation='get_async_session')
    session = async_session_factory()
    try:
        yield session
        await session.commit()
    except SQLAlchemyError as e:
        await session.rollback()
        logger.error(f'Database error: {str(e)}', extra={'error': str(e)},
            exc_info=True)
        raise DatabaseError(message=f'Database error: {str(e)}',
            service_name='ml-workbench-service', operation='get_async_session'
            ) from e
    finally:
        await session.close()


def create_tables() ->None:
    """Create all tables defined in the models."""
    if sync_engine is None:
        init_db()
    metadata.create_all(sync_engine)


def drop_tables() ->None:
    """Drop all tables defined in the models."""
    if sync_engine is None:
        init_db()
    metadata.drop_all(sync_engine)


class BaseRepository(Generic[T]):
    """Base repository for database operations."""

    def __init__(self, model_class: Type[T]):
        """
        Initialize the repository.

        Args:
            model_class: Model class
        """
        self.model_class = model_class

    async def get_by_id(self, id: int) ->Optional[T]:
        """
        Get entity by ID.

        Args:
            id: Entity ID

        Returns:
            Entity or None if not found
        """
        async with get_async_session() as session:
            result = await session.get(self.model_class, id)
            return result

    async def get_all(self) ->List[T]:
        """
        Get all entities.

        Returns:
            List of entities
        """
        async with get_async_session() as session:
            result = await session.execute(select(self.model_class))
            return result.scalars().all()

    async def create(self, **kwargs: Any) ->T:
        """
        Create a new entity.

        Args:
            **kwargs: Entity attributes

        Returns:
            Created entity
        """
        async with get_async_session() as session:
            entity = self.model_class(**kwargs)
            session.add(entity)
            await session.commit()
            await session.refresh(entity)
            return entity

    async def update(self, id: int, **kwargs: Any) ->Optional[T]:
        """
        Update entity.

        Args:
            id: Entity ID
            **kwargs: Entity attributes to update

        Returns:
            Updated entity or None if not found
        """
        async with get_async_session() as session:
            entity = await session.get(self.model_class, id)
            if entity:
                for key, value in kwargs.items():
                    setattr(entity, key, value)
                await session.commit()
                await session.refresh(entity)
            return entity

    async def delete(self, id: int) ->bool:
        """
        Delete entity.

        Args:
            id: Entity ID

        Returns:
            True if entity was deleted, False otherwise
        """
        async with get_async_session() as session:
            entity = await session.get(self.model_class, id)
            if entity:
                await session.delete(entity)
                await session.commit()
                return True
            return False

    async def count(self) ->int:
        """
        Count entities.

        Returns:
            Number of entities
        """
        async with get_async_session() as session:
            result = await session.execute(select(self.model_class).count())
            return result.scalar_one()

    async def exists(self, id: int) ->bool:
        """
        Check if entity exists.

        Args:
            id: Entity ID

        Returns:
            True if entity exists, False otherwise
        """
        async with get_async_session() as session:
            result = await session.get(self.model_class, id)
            return result is not None

    async def execute_query(self, query: Select) ->List[T]:
        """
        Execute custom query.

        Args:
            query: SQLAlchemy query

        Returns:
            Query results
        """
        async with get_async_session() as session:
            result = await session.execute(query)
            return result.scalars().all()


init_db()
