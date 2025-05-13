"""
Standardized Database Module for Monitoring Alerting Service

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
from monitoring_alerting_service.config.standardized_config import get_db_url, settings
from monitoring_alerting_service.logging_setup import get_logger
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
            ='monitoring-alerting-service', operation='get_async_session')
    session = async_session_factory()
    try:
        yield session
        await session.commit()
    except SQLAlchemyError as e:
        await session.rollback()
        logger.error(f'Database error: {str(e)}', extra={'error': str(e)},
            exc_info=True)
        raise DatabaseError(message=f'Database error: {str(e)}',
            service_name='monitoring-alerting-service', operation=
            'get_async_session') from e
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


class Alert(Base):
    """Alert model."""
    __tablename__ = 'alerts'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(String(1000), nullable=True)
    query = Column(String(1000), nullable=False)
    severity = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=text(
        'CURRENT_TIMESTAMP'))
    updated_at = Column(DateTime, nullable=False, server_default=text(
        'CURRENT_TIMESTAMP'), onupdate=text('CURRENT_TIMESTAMP'))
    last_triggered_at = Column(DateTime, nullable=True)
    silenced = Column(Boolean, nullable=False, default=False)
    silenced_until = Column(DateTime, nullable=True)
    silenced_by = Column(String(255), nullable=True)
    labels = Column(String(1000), nullable=True)
    annotations = Column(String(1000), nullable=True)


class AlertHistory(Base):
    """Alert history model."""
    __tablename__ = 'alert_history'
    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(Integer, ForeignKey('alerts.id'), nullable=False)
    status = Column(String(50), nullable=False)
    triggered_at = Column(DateTime, nullable=False, server_default=text(
        'CURRENT_TIMESTAMP'))
    resolved_at = Column(DateTime, nullable=True)
    duration = Column(Float, nullable=True)
    labels = Column(String(1000), nullable=True)
    annotations = Column(String(1000), nullable=True)
    alert = relationship('Alert', backref='history')


class Notification(Base):
    """Notification model."""
    __tablename__ = 'notifications'
    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_history_id = Column(Integer, ForeignKey('alert_history.id'),
        nullable=False)
    channel = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False)
    sent_at = Column(DateTime, nullable=False, server_default=text(
        'CURRENT_TIMESTAMP'))
    recipient = Column(String(255), nullable=False)
    message = Column(String(1000), nullable=False)
    error = Column(String(1000), nullable=True)
    alert_history = relationship('AlertHistory', backref='notifications')


class Dashboard(Base):
    """Dashboard model."""
    __tablename__ = 'dashboards'
    id = Column(Integer, primary_key=True, autoincrement=True)
    uid = Column(String(255), nullable=False, unique=True)
    title = Column(String(255), nullable=False)
    description = Column(String(1000), nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=text(
        'CURRENT_TIMESTAMP'))
    updated_at = Column(DateTime, nullable=False, server_default=text(
        'CURRENT_TIMESTAMP'), onupdate=text('CURRENT_TIMESTAMP'))
    created_by = Column(String(255), nullable=True)
    updated_by = Column(String(255), nullable=True)
    tags = Column(String(1000), nullable=True)
    data = Column(String(10000), nullable=False)


class MonitoringTarget(Base):
    """Monitoring target model."""
    __tablename__ = 'monitoring_targets'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(String(1000), nullable=True)
    type = Column(String(50), nullable=False)
    url = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=text(
        'CURRENT_TIMESTAMP'))
    updated_at = Column(DateTime, nullable=False, server_default=text(
        'CURRENT_TIMESTAMP'), onupdate=text('CURRENT_TIMESTAMP'))
    last_checked_at = Column(DateTime, nullable=True)
    labels = Column(String(1000), nullable=True)
    annotations = Column(String(1000), nullable=True)


class AlertRepository(BaseRepository[Alert]):
    """Repository for Alert model."""

    def __init__(self):
        """Initialize the repository."""
        super().__init__(Alert)

    async def get_by_name(self, name: str) ->Optional[Alert]:
        """
        Get alert by name.

        Args:
            name: Alert name

        Returns:
            Alert or None if not found
        """
        async with get_async_session() as session:
            result = await session.execute(select(Alert).where(Alert.name ==
                name))
            return result.scalar_one_or_none()

    async def get_by_status(self, status: str) ->List[Alert]:
        """
        Get alerts by status.

        Args:
            status: Alert status

        Returns:
            List of alerts
        """
        async with get_async_session() as session:
            result = await session.execute(select(Alert).where(Alert.status ==
                status))
            return result.scalars().all()

    async def get_by_severity(self, severity: str) ->List[Alert]:
        """
        Get alerts by severity.

        Args:
            severity: Alert severity

        Returns:
            List of alerts
        """
        async with get_async_session() as session:
            result = await session.execute(select(Alert).where(Alert.
                severity == severity))
            return result.scalars().all()

    async def get_silenced(self) ->List[Alert]:
        """
        Get silenced alerts.

        Returns:
            List of silenced alerts
        """
        async with get_async_session() as session:
            result = await session.execute(select(Alert).where(Alert.
                silenced == True))
            return result.scalars().all()


class AlertHistoryRepository(BaseRepository[AlertHistory]):
    """Repository for AlertHistory model."""

    def __init__(self):
        """Initialize the repository."""
        super().__init__(AlertHistory)

    async def get_by_alert_id(self, alert_id: int) ->List[AlertHistory]:
        """
        Get alert history by alert ID.

        Args:
            alert_id: Alert ID

        Returns:
            List of alert history entries
        """
        async with get_async_session() as session:
            result = await session.execute(select(AlertHistory).where(
                AlertHistory.alert_id == alert_id))
            return result.scalars().all()

    async def get_by_status(self, status: str) ->List[AlertHistory]:
        """
        Get alert history by status.

        Args:
            status: Alert history status

        Returns:
            List of alert history entries
        """
        async with get_async_session() as session:
            result = await session.execute(select(AlertHistory).where(
                AlertHistory.status == status))
            return result.scalars().all()

    async def get_unresolved(self) ->List[AlertHistory]:
        """
        Get unresolved alert history entries.

        Returns:
            List of unresolved alert history entries
        """
        async with get_async_session() as session:
            result = await session.execute(select(AlertHistory).where(
                AlertHistory.resolved_at == None))
            return result.scalars().all()


class NotificationRepository(BaseRepository[Notification]):
    """Repository for Notification model."""

    def __init__(self):
        """Initialize the repository."""
        super().__init__(Notification)

    async def get_by_alert_history_id(self, alert_history_id: int) ->List[
        Notification]:
        """
        Get notifications by alert history ID.

        Args:
            alert_history_id: Alert history ID

        Returns:
            List of notifications
        """
        async with get_async_session() as session:
            result = await session.execute(select(Notification).where(
                Notification.alert_history_id == alert_history_id))
            return result.scalars().all()

    async def get_by_channel(self, channel: str) ->List[Notification]:
        """
        Get notifications by channel.

        Args:
            channel: Notification channel

        Returns:
            List of notifications
        """
        async with get_async_session() as session:
            result = await session.execute(select(Notification).where(
                Notification.channel == channel))
            return result.scalars().all()

    async def get_by_status(self, status: str) ->List[Notification]:
        """
        Get notifications by status.

        Args:
            status: Notification status

        Returns:
            List of notifications
        """
        async with get_async_session() as session:
            result = await session.execute(select(Notification).where(
                Notification.status == status))
            return result.scalars().all()

    async def get_by_recipient(self, recipient: str) ->List[Notification]:
        """
        Get notifications by recipient.

        Args:
            recipient: Notification recipient

        Returns:
            List of notifications
        """
        async with get_async_session() as session:
            result = await session.execute(select(Notification).where(
                Notification.recipient == recipient))
            return result.scalars().all()


class DashboardRepository(BaseRepository[Dashboard]):
    """Repository for Dashboard model."""

    def __init__(self):
        """Initialize the repository."""
        super().__init__(Dashboard)

    async def get_by_uid(self, uid: str) ->Optional[Dashboard]:
        """
        Get dashboard by UID.

        Args:
            uid: Dashboard UID

        Returns:
            Dashboard or None if not found
        """
        async with get_async_session() as session:
            result = await session.execute(select(Dashboard).where(
                Dashboard.uid == uid))
            return result.scalar_one_or_none()

    async def get_by_title(self, title: str) ->Optional[Dashboard]:
        """
        Get dashboard by title.

        Args:
            title: Dashboard title

        Returns:
            Dashboard or None if not found
        """
        async with get_async_session() as session:
            result = await session.execute(select(Dashboard).where(
                Dashboard.title == title))
            return result.scalar_one_or_none()

    async def get_by_tag(self, tag: str) ->List[Dashboard]:
        """
        Get dashboards by tag.

        Args:
            tag: Dashboard tag

        Returns:
            List of dashboards
        """
        async with get_async_session() as session:
            result = await session.execute(select(Dashboard).where(
                Dashboard.tags.like(f'%{tag}%')))
            return result.scalars().all()


class MonitoringTargetRepository(BaseRepository[MonitoringTarget]):
    """Repository for MonitoringTarget model."""

    def __init__(self):
        """Initialize the repository."""
        super().__init__(MonitoringTarget)

    async def get_by_name(self, name: str) ->Optional[MonitoringTarget]:
        """
        Get monitoring target by name.

        Args:
            name: Monitoring target name

        Returns:
            Monitoring target or None if not found
        """
        async with get_async_session() as session:
            result = await session.execute(select(MonitoringTarget).where(
                MonitoringTarget.name == name))
            return result.scalar_one_or_none()

    async def get_by_type(self, type: str) ->List[MonitoringTarget]:
        """
        Get monitoring targets by type.

        Args:
            type: Monitoring target type

        Returns:
            List of monitoring targets
        """
        async with get_async_session() as session:
            result = await session.execute(select(MonitoringTarget).where(
                MonitoringTarget.type == type))
            return result.scalars().all()

    async def get_by_status(self, status: str) ->List[MonitoringTarget]:
        """
        Get monitoring targets by status.

        Args:
            status: Monitoring target status

        Returns:
            List of monitoring targets
        """
        async with get_async_session() as session:
            result = await session.execute(select(MonitoringTarget).where(
                MonitoringTarget.status == status))
            return result.scalars().all()


alert_repository = AlertRepository()
alert_history_repository = AlertHistoryRepository()
notification_repository = NotificationRepository()
dashboard_repository = DashboardRepository()
monitoring_target_repository = MonitoringTargetRepository()
init_db()
