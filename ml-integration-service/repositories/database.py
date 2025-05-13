"""
Standardized Database Module for ML Integration Service.

This module provides standardized database connectivity for the ML Integration Service,
including SQLAlchemy ORM setup, session management, and repository pattern implementation.
"""
import os
import logging
from typing import Dict, Any, Optional, List, Type, TypeVar, Generic, Callable
from contextlib import contextmanager
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Table, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship, scoped_session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_scoped_session
from sqlalchemy.orm import sessionmaker as async_sessionmaker
from sqlalchemy.future import select
from sqlalchemy.sql import text
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
from config.standardized_config_1 import get_db_url
logger = logging.getLogger(__name__)
Base = declarative_base()
engine = None
async_engine = None
SessionLocal = None
AsyncSessionLocal = None


from ml_integration_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

def init_db(db_url: Optional[str]=None) ->None:
    """
    Initialize the database connection.

    Args:
        db_url: Database URL (if None, get from config)
    """
    global engine, async_engine, SessionLocal, AsyncSessionLocal
    if db_url is None:
        db_url = get_db_url()
    logger.info(f'Initializing database connection to {db_url}')
    engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=3600,
        pool_size=5, max_overflow=10, echo=False)
    if db_url.startswith('postgresql'):
        async_db_url = db_url.replace('postgresql', 'postgresql+asyncpg')
        async_engine = create_async_engine(async_db_url, pool_pre_ping=True,
            pool_recycle=3600, pool_size=5, max_overflow=10, echo=False)
        AsyncSessionLocal = async_sessionmaker(async_engine,
            expire_on_commit=False, class_=AsyncSession)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info('Database connection initialized')


def create_tables() ->None:
    """Create all tables defined in SQLAlchemy models."""
    logger.info('Creating database tables')
    Base.metadata.create_all(bind=engine)
    logger.info('Database tables created')


@contextmanager
@with_exception_handling
def get_sync_session() ->Session:
    """
    Get a synchronous database session.

    Yields:
        Database session
    """
    if SessionLocal is None:
        init_db()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.exception(f'Database session error: {str(e)}')
        raise
    finally:
        session.close()


@async_with_exception_handling
async def get_async_session() ->AsyncSession:
    """
    Get an asynchronous database session.

    Returns:
        Asynchronous database session
    """
    if AsyncSessionLocal is None:
        init_db()
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.exception(f'Database session error: {str(e)}')
            raise


T = TypeVar('T', bound=Base)


class BaseRepository(Generic[T]):
    """Base repository for database operations."""

    def __init__(self, model: Type[T]):
        """
        Initialize the repository.

        Args:
            model: SQLAlchemy model class
        """
        self.model = model

    def get_by_id(self, session: Session, id: int) ->Optional[T]:
        """
        Get a record by ID.

        Args:
            session: Database session
            id: Record ID

        Returns:
            Record or None if not found
        """
        return session.query(self.model).filter(self.model.id == id).first()

    async def get_by_id_async(self, session: AsyncSession, id: int) ->Optional[
        T]:
        """
        Get a record by ID asynchronously.

        Args:
            session: Asynchronous database session
            id: Record ID

        Returns:
            Record or None if not found
        """
        result = await session.execute(select(self.model).filter(self.model
            .id == id))
        return result.scalars().first()

    def get_all(self, session: Session) ->List[T]:
        """
        Get all records.

        Args:
            session: Database session

        Returns:
            List of records
        """
        return session.query(self.model).all()

    async def get_all_async(self, session: AsyncSession) ->List[T]:
        """
        Get all records asynchronously.

        Args:
            session: Asynchronous database session

        Returns:
            List of records
        """
        result = await session.execute(select(self.model))
        return result.scalars().all()

    def create(self, session: Session, **kwargs) ->T:
        """
        Create a new record.

        Args:
            session: Database session
            **kwargs: Record attributes

        Returns:
            Created record
        """
        obj = self.model(**kwargs)
        session.add(obj)
        session.flush()
        return obj

    async def create_async(self, session: AsyncSession, **kwargs) ->T:
        """
        Create a new record asynchronously.

        Args:
            session: Asynchronous database session
            **kwargs: Record attributes

        Returns:
            Created record
        """
        obj = self.model(**kwargs)
        session.add(obj)
        await session.flush()
        return obj

    def update(self, session: Session, id: int, **kwargs) ->Optional[T]:
        """
        Update a record.

        Args:
            session: Database session
            id: Record ID
            **kwargs: Record attributes to update

        Returns:
            Updated record or None if not found
        """
        obj = self.get_by_id(session, id)
        if obj:
            for key, value in kwargs.items():
                setattr(obj, key, value)
            session.flush()
        return obj

    async def update_async(self, session: AsyncSession, id: int, **kwargs
        ) ->Optional[T]:
        """
        Update a record asynchronously.

        Args:
            session: Asynchronous database session
            id: Record ID
            **kwargs: Record attributes to update

        Returns:
            Updated record or None if not found
        """
        obj = await self.get_by_id_async(session, id)
        if obj:
            for key, value in kwargs.items():
                setattr(obj, key, value)
            await session.flush()
        return obj

    def delete(self, session: Session, id: int) ->bool:
        """
        Delete a record.

        Args:
            session: Database session
            id: Record ID

        Returns:
            True if deleted, False if not found
        """
        obj = self.get_by_id(session, id)
        if obj:
            session.delete(obj)
            session.flush()
            return True
        return False

    async def delete_async(self, session: AsyncSession, id: int) ->bool:
        """
        Delete a record asynchronously.

        Args:
            session: Asynchronous database session
            id: Record ID

        Returns:
            True if deleted, False if not found
        """
        obj = await self.get_by_id_async(session, id)
        if obj:
            await session.delete(obj)
            await session.flush()
            return True
        return False


class MLModel(Base):
    """ML model metadata."""
    __tablename__ = 'ml_models'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    description = Column(String(1000))
    framework = Column(String(100))
    input_schema = Column(String(2000))
    output_schema = Column(String(2000))
    metrics = Column(String(2000))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=
        datetime.utcnow)
    created_by = Column(String(100))
    is_active = Column(Boolean, default=True)
    tags = Column(String(500))


class MLModelVersion(Base):
    """ML model version."""
    __tablename__ = 'ml_model_versions'
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey('ml_models.id'), nullable=False)
    version = Column(String(50), nullable=False)
    description = Column(String(1000))
    metrics = Column(String(2000))
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(100))
    is_active = Column(Boolean, default=True)
    status = Column(String(50), default='draft')
    artifact_path = Column(String(500))
    model = relationship('MLModel', backref='versions')


class MLModelDeployment(Base):
    """ML model deployment."""
    __tablename__ = 'ml_model_deployments'
    id = Column(Integer, primary_key=True, index=True)
    model_version_id = Column(Integer, ForeignKey('ml_model_versions.id'),
        nullable=False)
    environment = Column(String(50), nullable=False)
    status = Column(String(50), default='pending')
    deployed_at = Column(DateTime, default=datetime.utcnow)
    deployed_by = Column(String(100))
    endpoint_url = Column(String(500))
    config = Column(String(2000))
    model_version = relationship('MLModelVersion', backref='deployments')


class MLModelPerformance(Base):
    """ML model performance metrics."""
    __tablename__ = 'ml_model_performance'
    id = Column(Integer, primary_key=True, index=True)
    model_version_id = Column(Integer, ForeignKey('ml_model_versions.id'),
        nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    dataset_id = Column(String(100))
    environment = Column(String(50))
    model_version = relationship('MLModelVersion', backref=
        'performance_metrics')


class MLModelFeatureImportance(Base):
    """ML model feature importance."""
    __tablename__ = 'ml_model_feature_importance'
    id = Column(Integer, primary_key=True, index=True)
    model_version_id = Column(Integer, ForeignKey('ml_model_versions.id'),
        nullable=False)
    feature_name = Column(String(100), nullable=False)
    importance_value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_version = relationship('MLModelVersion', backref='feature_importance'
        )


class MLModelRepository(BaseRepository[MLModel]):
    """Repository for ML models."""

    def __init__(self):
        """Initialize the repository."""
        super().__init__(MLModel)

    def get_by_name_and_version(self, session: Session, name: str, version: str
        ) ->Optional[MLModel]:
        """
        Get a model by name and version.

        Args:
            session: Database session
            name: Model name
            version: Model version

        Returns:
            Model or None if not found
        """
        return session.query(self.model).filter(self.model.name == name, 
            self.model.version == version).first()

    async def get_by_name_and_version_async(self, session: AsyncSession,
        name: str, version: str) ->Optional[MLModel]:
        """
        Get a model by name and version asynchronously.

        Args:
            session: Asynchronous database session
            name: Model name
            version: Model version

        Returns:
            Model or None if not found
        """
        result = await session.execute(select(self.model).filter(self.model
            .name == name, self.model.version == version))
        return result.scalars().first()


class MLModelVersionRepository(BaseRepository[MLModelVersion]):
    """Repository for ML model versions."""

    def __init__(self):
        """Initialize the repository."""
        super().__init__(MLModelVersion)

    def get_by_model_id_and_version(self, session: Session, model_id: int,
        version: str) ->Optional[MLModelVersion]:
        """
        Get a model version by model ID and version.

        Args:
            session: Database session
            model_id: Model ID
            version: Version

        Returns:
            Model version or None if not found
        """
        return session.query(self.model).filter(self.model.model_id ==
            model_id, self.model.version == version).first()

    async def get_by_model_id_and_version_async(self, session: AsyncSession,
        model_id: int, version: str) ->Optional[MLModelVersion]:
        """
        Get a model version by model ID and version asynchronously.

        Args:
            session: Asynchronous database session
            model_id: Model ID
            version: Version

        Returns:
            Model version or None if not found
        """
        result = await session.execute(select(self.model).filter(self.model
            .model_id == model_id, self.model.version == version))
        return result.scalars().first()


class MLModelDeploymentRepository(BaseRepository[MLModelDeployment]):
    """Repository for ML model deployments."""

    def __init__(self):
        """Initialize the repository."""
        super().__init__(MLModelDeployment)

    def get_by_model_version_id_and_environment(self, session: Session,
        model_version_id: int, environment: str) ->Optional[MLModelDeployment]:
        """
        Get a model deployment by model version ID and environment.

        Args:
            session: Database session
            model_version_id: Model version ID
            environment: Environment

        Returns:
            Model deployment or None if not found
        """
        return session.query(self.model).filter(self.model.model_version_id ==
            model_version_id, self.model.environment == environment).first()

    async def get_by_model_version_id_and_environment_async(self, session:
        AsyncSession, model_version_id: int, environment: str) ->Optional[
        MLModelDeployment]:
        """
        Get a model deployment by model version ID and environment asynchronously.

        Args:
            session: Asynchronous database session
            model_version_id: Model version ID
            environment: Environment

        Returns:
            Model deployment or None if not found
        """
        result = await session.execute(select(self.model).filter(self.model
            .model_version_id == model_version_id, self.model.environment ==
            environment))
        return result.scalars().first()


ml_model_repository = MLModelRepository()
ml_model_version_repository = MLModelVersionRepository()
ml_model_deployment_repository = MLModelDeploymentRepository()
