"""
Standardized Database Connection Utilities using SQLAlchemy AsyncIO.

Provides functions to create an asynchronous SQLAlchemy engine and manage sessions.
"""

import os
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from contextlib import asynccontextmanager

# Base class for declarative models
Base = declarative_base()

# Global session maker - configure once
_async_session_maker: Optional[async_sessionmaker[AsyncSession]] = None

def create_async_db_engine(database_url: Optional[str] = None, echo: bool = False):
    """
    Creates an asynchronous SQLAlchemy engine.

    Args:
        database_url: The database connection URL. If None, attempts to read from
                      DATABASE_URL environment variable.
        echo: If True, the engine will log all statements as well as a repr()
              of their parameter lists to the default log handler.

    Returns:
        An asynchronous SQLAlchemy engine instance.

    Raises:
        ValueError: If database_url is not provided and DATABASE_URL env var is not set.
    """
    if database_url is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("Database URL must be provided either as an argument or via DATABASE_URL environment variable.")

    # Recommended settings for asyncpg:
    # https://docs.sqlalchemy.org/en/20/dialects/postgresql.html#asyncpg-execution-options
    connect_args = {}
    if database_url.startswith("postgresql+asyncpg"):
        connect_args = {
            "statement_cache_size": 0, # Disable statement caching for pgbouncer compatibility
            "prepared_statement_cache_size": 0
        }

    engine = create_async_engine(database_url, echo=echo, connect_args=connect_args)
    return engine

def configure_session_maker(engine):
    """
    Configures the global session maker. Should be called once during application startup.

    Args:
        engine: The asynchronous SQLAlchemy engine instance.
    """
    global _async_session_maker
    if _async_session_maker is not None:
        # Avoid reconfiguring if already done
        return

    _async_session_maker = async_sessionmaker(
        bind=engine,
        expire_on_commit=False,  # Recommended for async use
        class_=AsyncSession
    )

@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Provides an asynchronous session context manager.

    Yields:
        An AsyncSession instance.

    Raises:
        RuntimeError: If configure_session_maker has not been called.
    """
    if _async_session_maker is None:
        raise RuntimeError("Session maker not configured. Call configure_session_maker first.")

    async with _async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def init_db(engine):
    """
    Initialize the database (create tables). Use with caution, typically for dev/testing.
    """
    async with engine.begin() as conn:
        # await conn.run_sync(Base.metadata.drop_all) # Optional: drop existing tables
        await conn.run_sync(Base.metadata.create_all)

# Example usage (typically in your main application setup):
#
# from common_lib.db import create_async_db_engine, configure_session_maker, get_session, Base, init_db
# from sqlalchemy import Column, Integer, String
#
# class Item(Base):
    """
    Item class that inherits from Base.
    
    Attributes:
        Add attributes here
    """

#     __tablename__ = 'items'
#     id = Column(Integer, primary_key=True)
#     name = Column(String)
#
# async def main():
    """
    Main.
    
    """

#     db_url = "postgresql+asyncpg://user:password@host:port/dbname"
#     engine = create_async_db_engine(db_url)
#     configure_session_maker(engine)
#
#     # Initialize DB (optional)
#     # await init_db(engine)
#
#     # Use session
#     async with get_session() as session:
#         new_item = Item(name="Example Item")
#         session.add(new_item)
#         # await session.commit() is handled by context manager
#
#     await engine.dispose() # Clean up engine connections
#
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())

